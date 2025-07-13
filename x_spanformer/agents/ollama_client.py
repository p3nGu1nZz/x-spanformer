from typing import Optional, List

from ollama import AsyncClient
from rich.console import Console

from x_spanformer.agents.constants import DEFAULT_SYSTEM

c = Console()
Message = dict[str, str]

async def chat(
	model: str,
	conversation: List[Message],
	system: Optional[str] = None,
	temperature: float = 0.2
) -> str:
	client = AsyncClient()
	messages: List[Message] = conversation.copy()  # Make a copy to avoid modifying original
	
	# Handle empty conversation by adding system message if provided
	if not messages and system:
		messages = [{"role": "system", "content": system}]
	elif not messages:
		# If completely empty, create a minimal conversation
		messages = [{"role": "system", "content": "You are a helpful assistant."}]

	c.print(f"[bold blue]Sending to {model} (T={temperature}):[/bold blue]")
	for i, msg in enumerate(messages):
		role_color = "yellow" if msg["role"] == "system" else "cyan" if msg["role"] == "user" else "green"
		c.print(f"[{role_color}]Message {i+1} ({msg['role']}):[/{role_color}]")
		c.print(f"[white]{(msg['content'][:77] + '...') if len(msg['content']) > 80 else msg['content']}[/white]")
		c.print()

	try:
		response = await client.chat(
			model=model,
			messages=messages,
			options={"temperature": temperature}
		)
		
		content = response.message.content
		
		if not content:
			c.print(f"[red]Empty response from {model}[/red]")
			raise ValueError(f"Empty response from model {model}")
		
		c.print(f"[bold green]Response from {model}:[/bold green]")
		c.print(f"[white]{(content[:77] + '...') if len(content) > 80 else content}[/white]")
		c.print()
		
		return content
		
	except Exception as e:
		c.print(f"[red]Error in ollama chat: {e}[/red]")
		c.print(f"[red]Error type: {type(e).__name__}[/red]")
		raise