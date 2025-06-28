import re
from rich.console import Console
from rich.panel import Panel
from tenacity import retry, stop_after_attempt, wait_fixed

from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.ollama_client import chat
from x_spanformer.agents.prompts import render_prompt

c = Console()

class CritiqueSession:
	def __init__(self, config_name="selfcrit.yaml"):
		self.cfg = load_selfcrit_config(config_name)
		self.system = render_prompt(self.cfg["templates"]["system"])
		self.regex_filters = [re.compile(rx["pattern"]) for rx in self.cfg.get("regex_filters", [])]
		self.pattern = re.compile(
			r"Score:\s*(?P<score>[0-9.]+)\s*"
			r"Status:\s*(?P<status>\w+)\s*"
			r"Reason:\s*(?P<reason>.+)",
			re.IGNORECASE | re.DOTALL
		)
		c.print(f"[bold cyan]🧪 CritiqueSession initialized[/] with config: [yellow]{config_name}[/yellow]")

	@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
	async def evaluate(self, text: str) -> dict:
		c.print(f"[white]🔍 Evaluating text[/] (len={len(text)}): [dim]{text[:80]}{'…' if len(text) > 80 else ''}")
		for rx in self.regex_filters:
			if rx.search(text):
				c.print(f"[red]❌ Regex match:[/] pattern=[dim]{rx.pattern}[/] — auto-discarded")
				return {"score": 0.1, "status": "discard", "reason": "regex filter triggered"}

		passes = self.cfg["evaluation"]["passes"]
		model = self.cfg["model"]["name"]
		temp = self.cfg["model"]["temperature"]
		max_turns = self.cfg["dialogue"]["max_turns"]

		results = []
		for i in range(passes):
			dm = DialogueManager(system_prompt=self.system, max_turns=max_turns)
			dm.add("user", render_prompt(self.cfg["templates"]["score"], text=text))
			c.print(f"[cyan]• Pass {i+1}/{passes}[/] — model=[magenta]{model}[/], T={temp}")

			reply = await chat(model=model, conversation=dm.as_messages(), system=self.system, temperature=temp)
			parsed = self.parse(reply)
			results.append(parsed)

			c.print(f"[green]↳ Response {i+1}:[/] [white]{parsed['status']}[/], score={parsed['score']} — [dim]{parsed['reason']}[/]")

			if parsed["status"] in ("keep", "discard"):
				c.print(f"[bold magenta]✓ Early exit[/] — decisive response: [bold]{parsed['status']}[/]")
				return parsed

		c.print(f"[yellow]⚖ No decisive agreement — applying consensus resolution[/]")
		return self.resolve(results)

	def parse(self, text: str) -> dict:
		m = self.pattern.search(text)
		if not m:
			c.print(Panel.fit(f"[bold yellow]⚠ Could not parse model output:[/]\n[dim]{text.strip()[:160]}", title="Parse Failure", border_style="yellow"))
			return {"score": 0.5, "status": "revise", "reason": "unparseable"}
		return {
			"score": float(m["score"]),
			"status": m["status"].strip().lower(),
			"reason": m["reason"].strip()
		}

	def resolve(self, all_votes: list[dict]) -> dict:
		scores = [r["score"] for r in all_votes]
		statuses = [r["status"] for r in all_votes]
		reasons = sorted({r["reason"] for r in all_votes})

		final = {
			"score": round(sum(scores) / len(scores), 3),
			"status": max(set(statuses), key=statuses.count),
			"reason": " / ".join(reasons)
		}

		c.print(Panel.fit(
			f"[green]Status:[/] {final['status']}\n"
			f"[cyan]Score:[/] {final['score']}\n"
			f"[dim]Reason:[/] {final['reason']}",
			title="🧠 Consensus Verdict", border_style="cyan"
		))
		return final