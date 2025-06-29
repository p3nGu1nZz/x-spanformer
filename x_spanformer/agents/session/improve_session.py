from pathlib import Path
from typing import Optional, Tuple, Dict
import jinja2
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_fixed
import re

from ..config_loader import load_selfcrit_config
from ..dialogue import DialogueManager
from ..ollama_client import chat

c = Console()

class ImproveSession:
    """Session for improving text segments using AI."""
    
    def __init__(self, config: Optional[Dict] = None, config_name="selfcrit.yaml", template_name="segment_improve.j2", quiet=False):
        if config:
            self.cfg = config
        else:
            self.cfg = load_selfcrit_config(config_name, quiet=quiet)
        self.template = self._load_template(template_name)
        if not quiet and not config:
            c.print(f"[bold blue]🔧 ImproveSession initialized[/] with config: [yellow]{config_name}[/yellow]")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    def _load_template(self, template_name: str) -> jinja2.Template:
        """Load the improvement template."""
        template_path = Path(__file__).parent.parent / "templates" / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Improvement template not found at: {template_path}")
            
        try:
            with template_path.open("r", encoding="utf-8") as f:
                return jinja2.Template(f.read())
        except Exception as e:
            c.print(f"[red]Error loading template {template_path}: {e}[/red]")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    async def improve(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Improve text using AI.
        
        Returns:
            Tuple of (improved_text, content_type) where:
            - improved_text: The improved version of the text, or None if no improvement
            - content_type: Classification as "Natural", "Code", or "Mixed", or None
        """
        c.print(f"[white]🔧 Improving text[/] (len={len(text)}): [dim]{text[:80]}{'…' if len(text) > 80 else ''}")
        
        try:
            # Use DialogueManager for conversation management
            dialogue = DialogueManager(max_turns=3)
            
            # Render the improvement prompt
            prompt = self.template.render(text=text)
            dialogue.add("user", prompt)
            
            # Get model configuration
            model_name = self.cfg["model"]["name"]
            temperature = self.cfg["model"].get("temperature", 0.3)
            
            c.print(f"[dim]🔧 Calling improvement model: {model_name}[/dim]")
            
            # Call the LLM
            response = await chat(
                model=model_name,
                conversation=dialogue.as_messages(),
                temperature=temperature
            )
            
            if not response or not response.strip():
                c.print(f"[yellow]⚠ Empty response from improvement model[/yellow]")
                return None, None
            
            # Parse the response
            improved_text, content_type = self._parse_response(response, text)
            
            if improved_text:
                c.print(f"[green]✔ Improvement successful[/] — type: {content_type}, length: {len(improved_text)}")
            else:
                c.print(f"[yellow]⚠ No improvement generated[/] — type: {content_type}")
            
            return improved_text, content_type
            
        except Exception as e:
            # Escape potential Rich markup in error messages
            error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
            c.print(f"[red]🔧 Error during improvement: {error_msg}[/red]")
            return None, None

    def _parse_response(self, response: str, original_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the LLM response to extract improved text and content type."""
        # Escape potential Rich markup for safe logging
        safe_response = response.replace('[', '\\[').replace(']', '\\]')
        c.print(f"[dim]🔧 Parsing improvement response ({len(response)} chars):[/dim] {safe_response[:200]}{'...' if len(safe_response) > 200 else ''}")
        
        content_type = None
        improved_text = None
        
        # Handle DISCARD case first
        if response.strip().upper().startswith("DISCARD"):
            c.print(f"[yellow]🔧 LLM recommended DISCARD[/yellow]")
            return None, None
        
        lines = response.strip().split('\n')

        # Regex for labels, allowing for optional markdown bolding (both **Label:** and Label:)
        classification_re = re.compile(r"\*\*classification:\*\*|classification:", re.IGNORECASE)
        improved_text_re = re.compile(r"\*\*improved\s+text:\*\*|improved\s+text:", re.IGNORECASE)
        
        # Look for classification
        for line in lines:
            line_clean = line.strip()
            if classification_re.search(line_clean):
                type_value = classification_re.sub("", line_clean).strip().lower()
                if "natural" in type_value:
                    content_type = "Natural"
                elif "code" in type_value:
                    content_type = "Code"
                elif "mixed" in type_value:
                    content_type = "Mixed"
                c.print(f"[dim]🔧 Found classification: {content_type}[/dim]")
                break
        
        # Look for improved text
        improved_start = False
        improved_lines = []
        for line in lines:
            line_clean = line.strip()
            if improved_text_re.search(line_clean):
                improved_start = True
                # Check if the improved text is on the same line
                remainder = improved_text_re.sub("", line_clean).strip()
                if remainder:
                    improved_lines.append(remainder)
                continue
            if improved_start:
                improved_lines.append(line)
        
        if improved_lines:
            improved_text = '\n'.join(improved_lines).strip()
            # Clean up any markdown artifacts
            if improved_text.endswith("```"):
                improved_text = improved_text[:-3].strip()
            if improved_text.startswith("```"):
                improved_text = improved_text[3:].strip()
            c.print(f"[dim]🔧 Extracted improved text ({len(improved_text)} chars)[/dim]")
        
        # Validate the improved text
        if improved_text and len(improved_text) > 10 and improved_text != original_text:
            return improved_text, content_type
        elif content_type:
            # Return classification even if no improvement
            return None, content_type
        else:
            # If we couldn't parse anything meaningful, raise an exception to trigger retry
            c.print(f"[yellow]⚠ Failed to parse response, will retry[/yellow]")
            raise ValueError("Could not parse improvement response")
