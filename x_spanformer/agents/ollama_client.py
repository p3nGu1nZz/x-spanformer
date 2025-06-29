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
	messages: List[Message] = conversation  # conversation already includes system prompt from DialogueManager

	c.print(f"[bold blue]Sending to {model} (T={temperature}):[/bold blue]")
	for i, msg in enumerate(messages):
		role_color = "yellow" if msg["role"] == "system" else "cyan" if msg["role"] == "user" else "green"
		c.print(f"[{role_color}]Message {i+1} ({msg['role']}):[/{role_color}]")
		c.print(f"[white]{msg['content'][:80]}[/white]")
		c.print()

	response = await client.chat(
		model=model,
		messages=messages,
		options={"temperature": temperature}
	)
	content = response["message"]["content"]
	
	c.print(f"[bold green]Response from {model}:[/bold green]")
	c.print(f"[white]{content[:80]}[/white]")
	c.print()
	
	return content