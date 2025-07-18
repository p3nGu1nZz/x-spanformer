from typing import Literal

from rich.console import Console

from x_spanformer.agents.constants import DEFAULT_SYSTEM

c = Console()

class DialogueManager:
	def __init__(self, system_prompt: str = DEFAULT_SYSTEM, max_turns: int = 12):
		self.history: list[dict[str, str]] = []
		self.system = system_prompt
		self.max_turns = max_turns
		c.print(f"[dim]DialogueManager initialized — max_turns={max_turns}[/dim]")

	def add(self, role: Literal["user", "assistant"], content: str):
		self.history.append({"role": role, "content": content})
		self.history = self.history[-2 * self.max_turns:]
		c.print(f"[dim]Message added: role={role}, total turns={len(self.history)}[/dim]")

	def as_messages(self) -> list[dict[str, str]]:
		return [{"role": "system", "content": self.system}] + self.history