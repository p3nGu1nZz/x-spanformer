import asyncio
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from x_spanformer.agents import (
    config_loader,
    ollama_client,
    prompts,
    selfcrit,
)
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.session import (
    CritiqueSession,
    ImproveSession,
    JudgeSession,
)

# Mock the rich console to prevent printing during tests
from rich.console import Console

console = Console()
console.print = lambda *args, **kwargs: None


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestAgents(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent / "test_agent_data"
        self.test_dir.mkdir(exist_ok=True)
        self.config_dir = self.test_dir / "config"
        self.config_dir.mkdir(exist_ok=True)
        self.templates_dir = self.test_dir / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        self.dummy_config = {
            "agent_name": "test_agent",
            "model": {"name": "test_model", "temperature": 0.1},
            "evaluation": {"passes": 1, "max_retries": 1},
            "dialogue": {"max_turns": 2},
            "regex_filters": [{"pattern": "badword"}],
            "templates": {"system": "system_prompt", "score": "score_prompt"},
        }
        with (self.config_dir / "test.yaml").open("w") as f:
            yaml.dump(self.dummy_config, f)
        with (self.templates_dir / "test_prompt.j2").open("w") as f:
            f.write("Hello {{name}}")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir)

    @patch("x_spanformer.agents.config_loader.Path")
    def test_config_loader(self, mock_path):
        mock_path.return_value.parent.__truediv__.return_value = self.config_dir
        cfg = config_loader.load_selfcrit_config("test.yaml")
        self.assertEqual(cfg["agent_name"], "test_agent")

    def test_dialogue_manager(self):
        dm = DialogueManager(max_turns=1)
        dm.add("user", "1")
        dm.add("assistant", "2")
        dm.add("user", "3")
        self.assertEqual(len(dm.history), 2)
        self.assertEqual(dm.history[0]["content"], "2")
        self.assertEqual(dm.history[1]["content"], "3")

    @patch("x_spanformer.agents.prompts.env")
    def test_render_prompt(self, mock_env):
        mock_env.get_template.return_value.render = MagicMock(return_value="Hello World")
        result = prompts.render_prompt("test_template", name="World")
        self.assertEqual(result, "Hello World")
        mock_env.get_template.assert_called_with("test_template.j2")

    @patch("x_spanformer.agents.ollima_client.AsyncClient")
    def test_ollima_client_chat(self, mock_client):
        mock_response = {"message": {"content": "response"}}
        mock_client.return_value.chat = AsyncMock(return_value=mock_response)
        result = asyncio.run(
            ollama_client.chat("model", [{"role": "user", "content": "hi"}])
        )
        self.assertEqual(result, "response")

    @patch("x_spanformer.agents.selfcrit.chat", new_callable=AsyncMock)
    @patch("x_spanformer.agents.selfcrit.RE_FLAGGED", [re.compile("bad")])
    def test_judge_segment(self, mock_chat):
        mock_chat.return_value = "Score: 0.9\nStatus: keep\nReason: ok"
        result = asyncio.run(selfcrit.judge_segment("good text"))
        self.assertEqual(result["status"], "keep")

        result = asyncio.run(selfcrit.judge_segment("bad text"))
        self.assertEqual(result["status"], "discard")
        self.assertEqual(result["reason"], "regex filter triggered")

    def test_selfcrit_parse_response(self):
        parsed = selfcrit.parse_response(
            "Score: 0.8\nStatus: keep\nReason: looks good"
        )
        self.assertEqual(parsed["score"], 0.8)
        self.assertEqual(parsed["status"], "keep")
        self.assertEqual(parsed["reason"], "looks good")
        bad_raw = "invalid response"
        parsed = selfcrit.parse_response(bad_raw)
        self.assertEqual(parsed["status"], "revise")
