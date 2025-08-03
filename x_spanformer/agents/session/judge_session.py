import re
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Dict, Optional

from x_spanformer.config.judge_config_loader import load_judge_config
from ..dialogue import DialogueManager
from ..ollama_client import chat
from ..prompts import render_prompt

logger = logging.getLogger(__name__)

class JudgeSession:
    """Session for evaluating text segments for training data quality."""
    
    def __init__(self, config: Optional[Dict] = None, config_name="judge.yaml", quiet=False):
        if config:
            self.cfg = config
        else:
            self.cfg = load_judge_config(config_name, quiet=quiet)

        self.system = render_prompt(self.cfg["templates"]["system"])
        self.regex_filters = [re.compile(rx["pattern"]) for rx in self.cfg.get("regex_filters", [])]
        self.pattern = re.compile(
            r"(\*\*score:\*\*|score:)\s*(?P<score>[0-9.]+)\s*"
            r"(\*\*status:\*\*|status:)\s*(?P<status>\w+)\s*"
            r"(\*\*type:\*\*|type:)\s*(?P<type>\w+)\s*"
            r"(\*\*reason:\*\*|reason:)\s*(?P<reason>.+)",
            re.IGNORECASE | re.DOTALL
        )
        if not quiet and not config:
            logger.info(f"JudgeSession initialized with config: {config_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.05))  # Further reduced wait time to 0.05s for optimal performance
    async def evaluate(self, text: str) -> dict:
        """Evaluate a text segment for training data suitability with a single judge."""
        text_preview = (text[:80] + '…') if len(text) > 80 else text
        logger.info(f"Judging text (len={len(text)}): {text_preview}")
        
        # Apply regex filters first
        for rx in self.regex_filters:
            if rx.search(text):
                logger.info(f"Regex filter triggered: pattern={rx.pattern} — auto-discarded")
                return {"score": 0.1, "status": "discard", "reason": "regex filter triggered"}

        model = self.cfg["judge"]["model_name"]
        temp = self.cfg["judge"]["temperature"]
        max_turns = self.cfg["dialogue"]["max_turns"]

        dm = DialogueManager(system_prompt=self.system, max_turns=max_turns)
        dm.add("user", render_prompt(self.cfg["templates"]["judge"], text=text))
        logger.debug(f"Judge evaluation — model={model}, T={temp}")

        reply = await chat(model=model, conversation=dm.as_messages(), temperature=temp)
        result = self.parse(reply)

        logger.info(f"Judge response: {result['status']}, score={result['score']} — {result['reason']}")
        
        # Apply discard threshold - use judge's own threshold
        discard_threshold = self.cfg.get("judge", {}).get("discard_threshold", 0.25)
        if result["score"] < discard_threshold:
            logger.info(f"Judge threshold: Score {result['score']:.3f} below discard threshold {discard_threshold:.3f} — status changed to discard")
            result["status"] = "discard"
            result["reason"] = f"judge threshold: score {result['score']:.3f} < {discard_threshold:.3f}"
        
        return result

    def parse(self, text: str) -> dict:
        """Parse LLM response into structured judgment."""
        m = self.pattern.search(text)
        if not m:
            logger.warning(f"Could not parse judge output: {text.strip()[:160]}")
            return {"score": 0.5, "status": "discard", "type": "natural", "reason": "unparseable"}
        
        # Normalize status to expected values (only keep/discard)
        status = m["status"].strip().lower()
        if status not in ["keep", "discard"]:
            status = "discard"  # Default fallback - be conservative
        
        # Normalize type to expected values
        content_type = m["type"].strip().lower()
        if content_type not in ["natural", "code", "mixed"]:
            content_type = "natural"  # Default fallback
            
        return {
            "score": float(m["score"]),
            "status": status,
            "type": content_type,
            "reason": m["reason"].strip()
        }

    def resolve(self, all_votes: list[dict]) -> dict:
        """Resolve multiple judge votes into a consensus."""
        scores = [r["score"] for r in all_votes]
        statuses = [r["status"] for r in all_votes]
        types = [r.get("type", "natural") for r in all_votes]
        reasons = sorted({r["reason"] for r in all_votes})

        final = {
            "score": round(sum(scores) / len(scores), 3),
            "status": max(set(statuses), key=statuses.count),
            "type": max(set(types), key=types.count),  # Most common type
            "reason": " / ".join(reasons)
        }

        logger.info(f"Judge consensus - Status: {final['status']}, Score: {final['score']}, Type: {final['type']}, Reason: {final['reason']}")
        return final
