import re

from rich.console import Console
from rich.panel import Panel
from tenacity import retry, stop_after_attempt, wait_fixed

from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.ollama_client import chat
from x_spanformer.agents.prompts import render_prompt

c = Console()
cfg = load_selfcrit_config()

RE_FLAGGED = [re.compile(rx["pattern"]) for rx in cfg.get("regex_filters", [])]

SCORE_PATTERN = re.compile(
	r"Score:\s*(?P<score>[0-9.]+)\s*"
	r"Status:\s*(?P<status>\w+)\s*"
	r"Reason:\s*(?P<reason>.+)",
	re.IGNORECASE | re.DOTALL
)

def parse_response(raw: str) -> dict:
	match = SCORE_PATTERN.search(raw)
	if not match:
		c.print(Panel.fit(f"[bold yellow]⚠ Unparseable model response:[/]\n[dim]{raw.strip()[:160]}...", border_style="yellow"))
		return {"score": 0.5, "status": "revise", "reason": "unparseable"}
	return {
		"score": float(match["score"]),
		"status": match["status"].strip().lower(),
		"reason": match["reason"].strip()
	}

@retry(stop=stop_after_attempt(cfg["evaluation"]["max_retries"]), wait=wait_fixed(0.5))
async def judge_segment(text: str) -> dict:
	c.print(f"[white]🔎 Evaluating span (len={len(text)}):[/] [dim]{text[:96]}{'…' if len(text) > 96 else ''}[/]")

	for rx in RE_FLAGGED:
		if rx.search(text):
			c.print(f"[red]❌ Regex filter matched:[/] [dim]{rx.pattern}[/] — span discarded")
			return {"score": 0.1, "status": "discard", "reason": "regex filter triggered"}

	system = render_prompt(cfg["templates"]["system"], text=text)
	model = cfg["model"]["name"]
	temp = cfg["model"].get("temperature", 0.2)
	passes = cfg["evaluation"]["passes"]
	consensus = []

	for i in range(passes):
		dm = DialogueManager(system_prompt=system, max_turns=cfg["dialogue"]["max_turns"])
		dm.add("user", render_prompt(cfg["templates"]["score"], text=text))
		c.print(f"[cyan]• Pass {i+1}/{passes}[/] — requesting judgment...")

		reply = await chat(
			model=model,
			conversation=dm.as_messages(),
			system=system,
			temperature=temp
		)

		result = parse_response(reply)
		consensus.append(result)
		c.print(f"[green]↳ Response {i+1}:[/] [white]{result['status']}[/], [dim]score={result['score']}[/] — {result['reason']}")

		if result["status"] in {"keep", "discard"}:
			c.print(f"[bold magenta]✓ Early consensus:[/] Decisive result: [bold]{result['status']}[/]")
			return result

	c.print(f"[yellow]⚖ Fallback triggered:[/] using majority + mean")
	statuses = [r["status"] for r in consensus]
	scores = [r["score"] for r in consensus]
	reasons = sorted({r["reason"] for r in consensus})

	final = {
		"score": round(sum(scores) / len(scores), 3),
		"status": max(set(statuses), key=statuses.count),
		"reason": " / ".join(reasons)
	}

	c.print(Panel.fit(
		f"[bold white]Final Consensus[/]\n"
		f"[green]Status:[/] {final['status']}\n"
		f"[cyan]Score:[/] {final['score']}\n"
		f"[dim]Reason:[/] {final['reason']}",
		title="🧠 Aggregated Judgment", border_style="cyan"
	))
	return final