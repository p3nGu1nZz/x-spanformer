import asyncio
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import yaml
from x_spanformer.agents import (
    ollama_client,
    prompts,
)
from x_spanformer.config import judge_config_loader
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.session import (
    JudgeSession,
)

# Mock the rich console to prevent printing during tests
from rich.console import Console

console = Console()
console.print = lambda *args, **kwargs: None


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestAgents(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with mock data."""
        self.dummy_config = {
            "agent_type": "test_agent",
            "model": {"name": "test_model", "temperature": 0.1},
            "judge": {
                "model_name": "test_model", 
                "temperature": 0.1, 
                "judges": 5,  # Updated to match real config
                "threshold": 0.7, 
                "discard_threshold": 0.25,
                "max_retries": 3
            },
            "dialogue": {"max_turns": 2},
            "regex_filters": [{"pattern": "badword"}],
            "templates": {"system": "system_prompt", "judge": "judge_prompt"},
        }

    @patch("x_spanformer.config.judge_config_loader.Path")
    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_config_loader(self, mock_yaml_load, mock_file_open, mock_path):
        """Test config loader with mocked file system."""
        # Mock YAML loading to return our test config
        mock_yaml_load.return_value = self.dummy_config
        
        # Mock path existence check
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value.parent.parent.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_path_instance
        
        # Suppress console output
        with patch("x_spanformer.config.judge_config_loader.c"):
            cfg = judge_config_loader.load_judge_config("test.yaml", quiet=True)
            
        self.assertEqual(cfg["agent_type"], "test_agent")
        self.assertEqual(cfg["model"]["name"], "test_model")
        self.assertEqual(cfg["judge"]["judges"], 5)

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

    @patch("x_spanformer.agents.ollama_client.AsyncClient")
    def test_ollama_client_chat(self, mock_client):
        mock_response = {"message": {"content": "response"}}
        mock_client.return_value.chat = AsyncMock(return_value=mock_response)
        result = asyncio.run(
            ollama_client.chat("model", [{"role": "user", "content": "hi"}])
        )
        self.assertEqual(result, "response")

    # Legacy selfcrit tests - OBSOLETE (removed with simplification)
    # The judge_segment functionality is now handled directly in the pipeline
    # with a 5-judge consensus system
    
    # @patch("x_spanformer.agents.selfcrit.chat", new_callable=AsyncMock)
    # @patch("x_spanformer.agents.selfcrit.RE_FLAGGED", [re.compile("bad")])
    # def test_judge_segment(self, mock_chat, mock_cfg):
    #     mock_cfg.__getitem__ = lambda _, key: self.dummy_config[key]
    #     mock_cfg.get = lambda key, default=None: self.dummy_config.get(key, default)
    #     mock_chat.return_value = "Score: 0.9\nStatus: keep\nReason: ok"
    #     result = asyncio.run(selfcrit.judge_segment("good text"))
    #     self.assertEqual(result["status"], "keep")

    #     result = asyncio.run(selfcrit.judge_segment("bad text"))
    #     self.assertEqual(result["status"], "discard")
    #     self.assertEqual(result["reason"], "regex filter triggered")

    # def test_selfcrit_parse_response(self):
    #     parsed = selfcrit.parse_response(
    #         "Score: 0.8\nStatus: keep\nReason: looks good"
    #     )
    #     self.assertEqual(parsed["score"], 0.8)
    #     self.assertEqual(parsed["status"], "keep")
    #     self.assertEqual(parsed["reason"], "looks good")
    #     bad_raw = "invalid response"
    #     parsed = selfcrit.parse_response(bad_raw)
    #     self.assertEqual(parsed["status"], "revise")
