import re

from tenacity import retry, stop_after_attempt, wait_fixed

from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.ollama_client import chat
from x_spanformer.agents.prompts import render_prompt
from x_spanformer.agents.agent_utils import console, rich_log

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
		rich_log(
			{"error": "Unparseable model response", "raw": raw.strip()},
			title="Parse Error",
			color="yellow",
		)
		return {"score": 0.5, "status": "revise", "reason": "unparseable"}
	return {
		"score": float(match["score"]),
		"status": match["status"].strip().lower(),
		"reason": match["reason"].strip()
	}

@retry(stop=stop_after_attempt(cfg["judge"]["max_retries"]), wait=wait_fixed(0.5))
async def judge_segment(text: str) -> dict:
	"""
	Judge role: Authoritative decision maker for record removal from dataframe.
	Uses 3-judge consensus with majority rules to decide keep/discard/revise.
	Only the judge can remove records - critics only update confidence scores.
	"""
	console.print(f"[bold blue]⚖️ JUDGE EVALUATION[/] — text segment ({len(text)} chars)")

	# Regex filters for immediate discard (no judge voting needed)
	for rx in RE_FLAGGED:
		if rx.search(text):
			console.print(f"[red]❌ Regex filter matched: {rx.pattern} — DISCARD (bypassing judges)[/red]")
			return {"score": 0.1, "status": "discard", "reason": "regex filter triggered"}

	system = render_prompt(cfg["templates"]["system"], text=text)
	model = cfg["judge"]["model_name"]
	temp = cfg["judge"]["temperature"]
	passes = cfg["judge"]["passes"]  # Use judge-specific passes (should be 3)
	consensus = []

	console.print(f"[cyan]🏛️ Convening {passes} judges for consensus vote...[/cyan]")

	for i in range(passes):
		dm = DialogueManager(system_prompt=system, max_turns=cfg["dialogue"]["max_turns"])
		dm.add("user", render_prompt(cfg["templates"]["score"], text=text))
		console.print(f"[cyan]Judge {i+1}/{passes} — requesting judgment...[/cyan]")

		reply = await chat(
			model=model,
			conversation=dm.as_messages(),
			temperature=temp
		)

		result = parse_response(reply)
		rich_log(result, title=f"Judge {i+1}/{passes} Verdict", color="green")
		consensus.append(result)

		# Early exit if unanimous decision on keep/discard
		if result["status"] in {"keep", "discard"}:
			console.print(f"[magenta]⚡ Judge {i+1} decisive: {result['status'].upper()}[/magenta]")
			if i == 0:  # If first judge is decisive, continue to get more opinions
				continue
			return result

	console.print(f"[yellow]⚖️ MAJORITY RULE: Processing {len(consensus)} judge votes[/yellow]")
	statuses = [r["status"] for r in consensus]
	scores = [r["score"] for r in consensus]
	reasons = sorted({r["reason"] for r in consensus})

	final = {
		"score": round(sum(scores) / len(scores), 3),
		"status": max(set(statuses), key=statuses.count),  # Majority rules
		"reason": " / ".join(reasons)
	}

	rich_log(final, title="🏛️ JUDGE CONSENSUS", color="bold green")
	
	# Apply judge thresholds for final decision
	judge_threshold = cfg.get("judge", {}).get("threshold", 0.69)
	discard_threshold = cfg.get("judge", {}).get("discard_threshold", 0.25)
	
	# First check discard threshold (lowest priority, most strict)
	if final["score"] < discard_threshold:
		console.print(f"[red]🗑️ JUDGE OVERRIDE: Score {final['score']:.3f} below discard threshold {discard_threshold:.3f} — REMOVING RECORD[/red]")
		final["status"] = "discard"
		final["reason"] = f"judge discard threshold: score {final['score']:.3f} < {discard_threshold:.3f}"
	# Then check keep threshold (if not already discarded)
	elif final["score"] >= judge_threshold and final["status"] != "discard":
		console.print(f"[green]✅ JUDGE CONFIRMATION: Score {final['score']:.3f} meets keep threshold {judge_threshold:.3f}[/green]")
		if final["status"] == "revise":
			final["status"] = "keep"  # Upgrade revise to keep if score is high enough
			final["reason"] = f"judge threshold: score {final['score']:.3f} >= {judge_threshold:.3f}"
	
	status_emoji = {"keep": "✅", "discard": "🗑️", "revise": "🔄"}
	console.print(f"[bold]{status_emoji.get(final['status'], '❓')} FINAL JUDGE DECISION: {final['status'].upper()}[/bold]")
	
	return final

def update_confidence_score(text: str, score: float) -> dict:
	"""
	Critic role: Only updates confidence scores, cannot remove records from dataframe.
	The critic provides quality assessment but defers to judge for removal decisions.
	"""
	console.print(f"[dim]📊 CRITIC: Updating confidence score to {score:.3f}[/dim]")
	return {
		"confidence": score,
		"role": "critic",
		"action": "confidence_update_only"
	}

async def process_segment_cycle(text: str, max_cycles: int | None = None) -> dict:
	"""
	Main processing cycle: Judge evaluates -> if revise, improve -> repeat until limits.
	Only judge can remove records from dataframe. Process continues with improved text.
	"""
	cycles_limit = max_cycles if max_cycles is not None else cfg["dialogue"]["max_turns"]
	
	console.print(f"[bold yellow]🔄 Starting segment processing cycle (max {cycles_limit} cycles)[/bold yellow]")
	
	current_text = text
	cycle_count = 0
	
	while cycle_count < cycles_limit:
		cycle_count += 1
		console.print(f"[bold cyan]━━━ CYCLE {cycle_count}/{cycles_limit} ━━━[/bold cyan]")
		
		# Judge evaluation (authoritative decision maker)
		judge_result = await judge_segment(current_text)
		
		if judge_result["status"] == "discard":
			console.print(f"[red]🗑️ CYCLE {cycle_count}: Judge decided to DISCARD - removing from dataframe[/red]")
			return judge_result
		
		elif judge_result["status"] == "keep":
			console.print(f"[green]✅ CYCLE {cycle_count}: Judge decided to KEEP - adding to dataframe[/green]")
			return {**judge_result, "final_text": current_text, "cycles_completed": cycle_count}
		
		elif judge_result["status"] == "revise":
			console.print(f"[yellow]🔄 CYCLE {cycle_count}: Judge decided to REVISE - attempting improvement[/yellow]")
			
			# Import here to avoid circular dependency
			from x_spanformer.agents.session.improve_session import ImproveSession
			improver = ImproveSession()
			
			try:
				improved_result = await improver.improve(current_text, judge_result.get("reason", "needs improvement"))
				if improved_result and len(improved_result) >= 2 and improved_result[0]:
					current_text = improved_result[0]  # Get improved text from tuple
					console.print(f"[green]✨ CYCLE {cycle_count}: Text improved, continuing to next cycle[/green]")
				else:
					console.print(f"[yellow]⚠️ CYCLE {cycle_count}: Improvement failed, keeping original[/yellow]")
					break
			except Exception as e:
				console.print(f"[red]❌ CYCLE {cycle_count}: Improvement error: {e}[/red]")
				break
		
		else:
			console.print(f"[red]❓ CYCLE {cycle_count}: Unknown status '{judge_result['status']}' - breaking cycle[/red]")
			break
	
	# Reached max cycles without resolution
	console.print(f"[yellow]⏰ Reached maximum cycles ({cycles_limit}) - defaulting to current state[/yellow]")
	return {
		"score": judge_result.get("score", 0.5),
		"status": "keep",  # Default to keep if we've gone through all cycles
		"reason": f"max cycles reached ({cycle_count})",
		"final_text": current_text,
		"cycles_completed": cycle_count
	}