import asyncio
import subprocess
import re
from typing import Optional, List

from ollama import AsyncClient
from rich.console import Console

from x_spanformer.agents.constants import DEFAULT_SYSTEM

c = Console()
Message = dict[str, str]

async def check_ollama_running() -> bool:
	"""
	Check if Ollama service is running by testing 'ollama ps' command.
	Returns True if Ollama is running, False otherwise.
	"""
	try:
		result = subprocess.run(
			["ollama", "ps"],
			capture_output=True,
			text=True,
			encoding='utf-8',
			errors='replace',
			timeout=5
		)
		return result.returncode == 0
	except subprocess.TimeoutExpired:
		c.print(f"[red]❌ ollama ps command timed out[/red]")
		return False
	except FileNotFoundError:
		c.print(f"[red]❌ ollama command not found - is Ollama installed?[/red]")
		return False
	except Exception as e:
		c.print(f"[red]❌ Error checking ollama status: {str(e)}[/red]")
		return False

def check_model_loaded(model: str, ps_output: str) -> bool:
	"""
	Check if a specific model is loaded by parsing ollama ps output.
	Returns True if model is found in the output, False otherwise.
	"""
	# Pattern matches: "phi4-mini" or "phi4-mini:latest" or "phi4-mini:anything"
	model_pattern = re.compile(rf"^{re.escape(model)}(:|$|\s)", re.MULTILINE)
	return bool(model_pattern.search(ps_output))

async def check_ollama_connection(model: str) -> bool:
	"""
	Test if Ollama is running and the specified model is loaded.
	Returns True if model is loaded and accessible, False otherwise.
	"""
	try:
		# First check if Ollama is running
		if not await check_ollama_running():
			c.print(f"[red]❌ Ollama service is not running[/red]")
			return False
		
		# Get current ollama ps output
		result = subprocess.run(
			["ollama", "ps"],
			capture_output=True,
			text=True,
			encoding='utf-8',
			errors='replace',
			timeout=5
		)
		
		if result.returncode != 0:
			c.print(f"[red]❌ ollama ps failed: {result.stderr}[/red]")
			return False
		
		# Check if model is loaded
		if check_model_loaded(model, result.stdout):
			c.print(f"[green]✅ Model {model} is loaded and ready[/green]")
			return True
		else:
			c.print(f"[red]❌ Model {model} is not loaded. Please run: ollama run {model}[/red]")
			return False
			
	except Exception as e:
		c.print(f"[red]❌ Error checking ollama connection: {str(e)}[/red]")
		return False

async def chat(
	model: str,
	conversation: List[Message],
	system: Optional[str] = None,
	temperature: float = 0.2
) -> str:
	client = AsyncClient()
	
	# Build messages list with system prompt if provided
	messages: List[Message] = []
	if system:
		messages.append({"role": "system", "content": system})
	messages.extend(conversation)

	# Reduced logging for better performance - only show essential info
	c.print(f"[bold blue]Sending to {model} (T={temperature}):[/bold blue]")
	for i, msg in enumerate(messages):
		role_color = "yellow" if msg["role"] == "system" else "cyan" if msg["role"] == "user" else "green"
		preview = (msg['content'][:60] + '...') if len(msg['content']) > 60 else msg['content']
		c.print(f"[{role_color}]Message {i+1} ({msg['role']}):[/{role_color}] [dim]{preview}[/dim]")

	response = await client.chat(
		model=model,
		messages=messages,
		options={"temperature": temperature},
		stream=False
	)
	content = response["message"]["content"]
	
	# Reduced response logging for performance
	preview = (content[:60] + '...') if len(content) > 60 else content
	c.print(f"[bold green]Response from {model}:[/bold green] [dim]{preview}[/dim]")
	c.print()
	
	return content