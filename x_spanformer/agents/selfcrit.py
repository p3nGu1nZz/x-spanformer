import re

from rich.console import Console
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
		c.print(f"[yellow]⚠ Unparseable model response:[/yellow] {raw.strip()[:160]}...")
		return {"score": 0.5, "status": "revise", "reason": "unparseable"}
	return {
		"score": float(match["score"]),
		"status": match["status"].strip().lower(),
		"reason": match["reason"].strip()
	}

@retry(stop=stop_after_attempt(cfg["evaluation"]["max_retries"]), wait=wait_fixed(0.5))
async def judge_segment(text: str) -> dict:
	c.print(f"[dim]Evaluating text segment ({len(text)} chars)[/dim]")

	for rx in RE_FLAGGED:
		if rx.search(text):
			c.print(f"[red]Regex filter matched: {rx.pattern} — discarding[/red]")
			return {"score": 0.1, "status": "discard", "reason": "regex filter triggered"}

	system = render_prompt(cfg["templates"]["system"], text=text)
	model = cfg["model"]["name"]
	temp = cfg["model"].get("temperature", 0.2)
	passes = cfg["evaluation"]["passes"]
	consensus = []

	for i in range(passes):
		dm = DialogueManager(system_prompt=system, max_turns=cfg["dialogue"]["max_turns"])
		dm.add("user", render_prompt(cfg["templates"]["score"], text=text))
		c.print(f"[cyan]Pass {i+1}/{passes} — requesting judgment...[/cyan]")

		reply = await chat(
			model=model,
			conversation=dm.as_messages(),
			temperature=temp
		)

		result = parse_response(reply)
		consensus.append(result)
		c.print(f"[green]Response {i+1}: {result['status']}, score={result['score']} — {result['reason']}[/green]")

		if result["status"] in {"keep", "discard"}:
			c.print(f"[magenta]Early consensus: {result['status']}[/magenta]")
			return result

	c.print(f"[yellow]Using majority vote from {len(consensus)} responses[/yellow]")
	statuses = [r["status"] for r in consensus]
	scores = [r["score"] for r in consensus]
	reasons = sorted({r["reason"] for r in consensus})

	final = {
		"score": round(sum(scores) / len(scores), 3),
		"status": max(set(statuses), key=statuses.count),
		"reason": " / ".join(reasons)
	}

	c.print(f"[bold green]Final: {final['status']}, score={final['score']}[/bold green]")
	c.print(f"[dim]Reason: {final['reason']}[/dim]")
	return final