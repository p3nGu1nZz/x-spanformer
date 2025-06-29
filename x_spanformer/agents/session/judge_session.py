import re
from rich.console import Console
from rich.panel import Panel
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Dict, Optional

from ..config_loader import load_selfcrit_config
from ..dialogue import DialogueManager
from ..ollama_client import chat
from ..prompts import render_prompt

c = Console()

class JudgeSession:
    """Session for evaluating text segments for training data quality."""
    
    def __init__(self, config: Optional[Dict] = None, config_name="selfcrit.yaml", quiet=False):
        if config:
            self.cfg = config
        else:
            self.cfg = load_selfcrit_config(config_name, quiet=quiet)

        self.system = render_prompt(self.cfg["templates"]["system"])
        self.regex_filters = [re.compile(rx["pattern"]) for rx in self.cfg.get("regex_filters", [])]
        self.pattern = re.compile(
            r"(\*\*score:\*\*|score:)\s*(?P<score>[0-9.]+)\s*"
            r"(\*\*status:\*\*|status:)\s*(?P<status>\w+)\s*"
            r"(\*\*reason:\*\*|reason:)\s*(?P<reason>.+)",
            re.IGNORECASE | re.DOTALL
        )
        if not quiet and not config:
            c.print(f"[bold green]⚖️ JudgeSession initialized[/] with config: [yellow]{config_name}[/yellow]")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    async def evaluate(self, text: str) -> dict:
        """Evaluate a text segment for training data suitability."""
        c.print(f"[white]⚖️ Judging text[/] (len={len(text)}): [dim]{text[:80]}{'…' if len(text) > 80 else ''}")
        
        # Apply regex filters first
        for rx in self.regex_filters:
            if rx.search(text):
                c.print(f"[red]❌ Regex filter triggered:[/] pattern=[dim]{rx.pattern}[/] — auto-discarded")
                return {"score": 0.1, "status": "discard", "reason": "regex filter triggered"}

        passes = self.cfg["evaluation"]["passes"]
        model = self.cfg["model"]["name"]
        temp = self.cfg["model"]["temperature"]
        max_turns = self.cfg["dialogue"]["max_turns"]

        results = []
        for i in range(passes):
            dm = DialogueManager(system_prompt=self.system, max_turns=max_turns)
            dm.add("user", render_prompt(self.cfg["templates"]["score"], text=text))
            c.print(f"[cyan]• Judge Pass {i+1}/{passes}[/] — model=[magenta]{model}[/], T={temp}")

            reply = await chat(model=model, conversation=dm.as_messages(), temperature=temp)
            parsed = self.parse(reply)
            results.append(parsed)

            c.print(f"[green]↳ Judge Response {i+1}:[/] [white]{parsed['status']}[/], score={parsed['score']} — [dim]{parsed['reason']}[/]")

            # Early exit for decisive responses
            if parsed["status"] in ("keep", "discard"):
                c.print(f"[bold magenta]✓ Judge early exit[/] — decisive response: [bold]{parsed['status']}[/]")
                return parsed

        c.print(f"[yellow]⚖️ No decisive judge agreement — applying consensus resolution[/]")
        return self.resolve(results)

    def parse(self, text: str) -> dict:
        """Parse LLM response into structured judgment."""
        m = self.pattern.search(text)
        if not m:
            c.print(Panel.fit(f"[bold yellow]⚠ Could not parse judge output:[/]\n[dim]{text.strip()[:160]}", title="Judge Parse Failure", border_style="yellow"))
            return {"score": 0.5, "status": "revise", "reason": "unparseable"}
        return {
            "score": float(m["score"]),
            "status": m["status"].strip().lower(),
            "reason": m["reason"].strip()
        }

    def resolve(self, all_votes: list[dict]) -> dict:
        """Resolve multiple judge votes into a consensus."""
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
            title="⚖️ Judge Consensus", border_style="green"
        ))
        return final
