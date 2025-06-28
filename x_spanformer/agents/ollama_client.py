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
	messages: List[Message] = [{"role": "system", "content": system or DEFAULT_SYSTEM}]
	messages.extend(conversation)

	c.print(f"[bold blue]💬 chat()[/] sending [cyan]{len(messages)}[/] messages to [magenta]{model}[/] @ T={temperature}")
	response = await client.chat(
		model=model,
		messages=messages,
		options={"temperature": temperature}
	)
	content = response["message"]["content"]
	c.print(f"[green]✔ Reply received[/] — [dim]{len(content)} chars[/dim]")
	return content