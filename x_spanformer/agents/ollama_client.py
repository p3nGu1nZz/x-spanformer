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

async def load_model(model: str) -> bool:
	"""
	Attempt to load a model using 'ollama run' command.
	Returns True if successful, False otherwise.
	"""
	try:
		c.print(f"[cyan]🔄 Running: ollama run {model}[/cyan]")
		load_result = subprocess.run(
			["ollama", "run", model],
			capture_output=True,
			text=True,
			encoding='utf-8',  # Explicitly set UTF-8 encoding
			errors='replace',  # Replace problematic characters instead of failing
			timeout=60,  # Give it more time to load the model
			input="\n"  # Send a newline to exit the interactive mode
		)
		
		if load_result.returncode == 0:
			c.print(f"[green]✅ Successfully loaded model {model}[/green]")
			return True
		else:
			# Only show stderr if it's not empty and doesn't contain encoding issues
			error_msg = load_result.stderr.strip() if load_result.stderr else "Unknown error"
			c.print(f"[red]❌ Failed to load model {model}: {error_msg}[/red]")
			return False
			
	except subprocess.TimeoutExpired:
		c.print(f"[red]❌ Timeout loading model {model}[/red]")
		return False
	except Exception as e:
		c.print(f"[red]❌ Error loading model {model}: {str(e)}[/red]")
		return False

async def verify_model_loaded(model: str) -> bool:
	"""
	Verify that a model is loaded by checking ollama ps output.
	Returns True if model is found in ps output, False otherwise.
	"""
	try:
		verify_result = subprocess.run(
			["ollama", "ps"],
			capture_output=True,
			text=True,
			encoding='utf-8',
			errors='replace',
			timeout=5
		)
		
		if verify_result.returncode == 0:
			return check_model_loaded(model, verify_result.stdout)
		else:
			return False
	except Exception:
		return False

async def check_ollama_connection(model: str) -> bool:
	"""
	Test if Ollama is running and the specified model is loaded.
	If model is not loaded, automatically try to load it with 'ollama run'.
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
		
		# Check if model is already loaded
		if check_model_loaded(model, result.stdout):
			c.print(f"[green]✅ Model {model} is loaded and ready[/green]")
			return True
		
		# Model not loaded, try to load it automatically
		c.print(f"[yellow]⚠ Model {model} is not loaded. Attempting to load it...[/yellow]")
		
		if await load_model(model):
			# Verify the model is now loaded
			if await verify_model_loaded(model):
				c.print(f"[green]✅ Model {model} is now loaded and ready[/green]")
				return True
			else:
				c.print(f"[yellow]⚠ Model {model} loaded but not showing in ps output[/yellow]")
				return False
		else:
			c.print(f"[yellow]💡 Please manually run: ollama run {model}[/yellow]")
			return False
			
	except Exception as e:
		c.print(f"[red]❌ Error checking ollama connection: {str(e)}[/red]")
		return False

async def wait_for_ollama_recovery(model: str, max_wait_time: int = 300):
	"""
	Wait for Ollama to recover from a crash or disconnection.
	Checks every 20 seconds up to max_wait_time seconds.
	"""
	wait_interval = 20  # seconds
	elapsed_time = 0
	
	c.print(f"[yellow]⏳ Ollama connection lost. Waiting for recovery...[/yellow]")
	
	while elapsed_time < max_wait_time:
		await asyncio.sleep(wait_interval)
		elapsed_time += wait_interval
		
		c.print(f"[cyan]🔄 Checking Ollama connection... ({elapsed_time}s elapsed)[/cyan]")
		
		if await check_ollama_connection(model):
			c.print(f"[green]✅ Ollama connection restored! Resuming processing...[/green]")
			return True
		
		c.print(f"[yellow]⏳ Ollama still unavailable. Please reload Ollama to continue... (next check in {wait_interval}s)[/yellow]")
	
	c.print(f"[red]❌ Ollama recovery timeout after {max_wait_time}s. Exiting...[/red]")
	return False

async def chat(
	model: str,
	conversation: List[Message],
	system: Optional[str] = None,
	temperature: float = 0.2
) -> str:
	client = AsyncClient()
	messages: List[Message] = conversation  # conversation already includes system prompt from DialogueManager

	# Reduced logging for better performance - only show essential info
	c.print(f"[bold blue]Sending to {model} (T={temperature}):[/bold blue]")
	for i, msg in enumerate(messages):
		role_color = "yellow" if msg["role"] == "system" else "cyan" if msg["role"] == "user" else "green"
		preview = (msg['content'][:60] + '...') if len(msg['content']) > 60 else msg['content']
		c.print(f"[{role_color}]Message {i+1} ({msg['role']}):[/{role_color}] [dim]{preview}[/dim]")

	response = await client.chat(
		model=model,
		messages=messages,
		options={"temperature": temperature}
	)
	content = response["message"]["content"]
	
	# Reduced response logging for performance
	preview = (content[:60] + '...') if len(content) > 60 else content
	c.print(f"[bold green]Response from {model}:[/bold green] [dim]{preview}[/dim]")
	c.print()
	
	return content