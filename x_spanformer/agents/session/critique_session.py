import re

from tenacity import retry, stop_after_attempt, wait_fixed

from x_spanformer.agents.config_loader import load_selfcrit_config
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.ollama_client import chat
from x_spanformer.agents.prompts import render_prompt
from x_spanformer.agents.agent_utils import console, rich_log


class CritiqueSession:
    def __init__(self, config_name="selfcrit.yaml"):
        self.cfg = load_selfcrit_config(config_name)
        self.system = render_prompt(self.cfg["templates"]["system"])
        self.regex_filters = [
            re.compile(rx["pattern"]) for rx in self.cfg.get("regex_filters", [])
        ]
        self.pattern = re.compile(
            r"Score:\s*(?P<score>[0-9.]+)\s*"
            r"Status:\s*(?P<status>\w+)\s*"
            r"Reason:\s*(?P<reason>.+)",
            re.IGNORECASE | re.DOTALL,
        )
        console.print(
            f"[bold cyan]🧪 CritiqueSession initialized[/] with config: [yellow]{config_name}[/yellow]"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
    async def evaluate(self, text: str) -> dict:
        console.print(
            f"[white]🔍 Evaluating text[/] (len={len(text)}): [dim]{text[:80]}{'…' if len(text) > 80 else ''}"
        )
        for rx in self.regex_filters:
            if rx.search(text):
                console.print(
                    f"[red]❌ Regex match:[/] pattern=[dim]{rx.pattern}[/] — auto-discarded"
                )
                return {
                    "score": 0.1,
                    "status": "discard",
                    "reason": "regex filter triggered",
                }

        passes = self.cfg["critique"]["passes"]
        model = self.cfg["judge"]["model_name"]
        temp = self.cfg["judge"]["temperature"]
        max_turns = self.cfg["dialogue"]["max_turns"]

        results = []
        for i in range(passes):
            dm = DialogueManager(system_prompt=self.system, max_turns=max_turns)
            dm.add("user", render_prompt(self.cfg["templates"]["score"], text=text))
            console.print(
                f"[cyan]• Pass {i+1}/{passes}[/] — model=[magenta]{model}[/], T={temp}"
            )

            reply = await chat(
                model=model,
                conversation=dm.as_messages(),
                system=self.system,
                temperature=temp,
            )
            parsed = self.parse(reply)
            results.append(parsed)

            rich_log(parsed, title=f"Response {i+1}", color="green")

            if parsed["status"] in ("keep", "discard"):
                console.print(
                    f"[bold magenta]✓ Early exit[/] — decisive response: [bold]{parsed['status']}[/]"
                )
                return parsed

        console.print(f"[yellow]⚖ No decisive agreement — applying consensus resolution[/]")
        return self.resolve(results)

    def parse(self, text: str) -> dict:
        m = self.pattern.search(text)
        if not m:
            rich_log(
                {"error": "Could not parse model output", "text": text.strip()[:160]},
                title="Parse Failure",
                color="yellow",
            )
            return {"score": 0.5, "status": "revise", "reason": "unparseable"}
        return {
            "score": float(m["score"]),
            "status": m["status"].strip().lower(),
            "reason": m["reason"].strip(),
        }

    def resolve(self, all_votes: list[dict]) -> dict:
        scores = [r["score"] for r in all_votes]
        statuses = [r["status"] for r in all_votes]
        reasons = sorted({r["reason"] for r in all_votes})

        final = {
            "score": round(sum(scores) / len(scores), 3),
            "status": max(set(statuses), key=statuses.count),
            "reason": " / ".join(reasons),
        }

        rich_log(final, title="Consensus Verdict", color="cyan")
        # Apply discard threshold after consensus
        discard_threshold = self.cfg.get("critique", {}).get("discard_threshold", 0.25)
        if final["score"] < discard_threshold:
            console.print(
                f"[red]🗑️ Score {final['score']:.3f} below discard threshold {discard_threshold:.3f}, overriding status to discard[/red]"
            )
            final["status"] = "discard"
            final["reason"] = (
                f"score {final['score']:.3f} below discard threshold {discard_threshold:.3f}"
            )

        return final
