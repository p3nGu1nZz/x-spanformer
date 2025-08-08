import logging
from typing import Literal

from x_spanformer.agents.constants import DEFAULT_SYSTEM

logger = logging.getLogger(__name__)

class DialogueManager:
	def __init__(self, system_prompt: str = DEFAULT_SYSTEM, max_turns: int = 12):
		self.history: list[dict[str, str]] = []
		self.system = system_prompt
		self.max_turns = max_turns
		logger.debug(f"DialogueManager initialized with max_turns={max_turns}")

	def add(self, role: Literal["user", "assistant"], content: str):
		self.history.append({"role": role, "content": content})
		self.history = self.history[-2 * self.max_turns:]
		logger.info(f"[DIALOGUE] {role.upper()}:\n{content}")
		logger.debug(f"Message added: role={role}, total turns={len(self.history)}")

	def as_messages(self) -> list[dict[str, str]]:
		return [{"role": "system", "content": self.system}] + self.history