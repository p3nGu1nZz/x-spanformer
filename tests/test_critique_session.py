import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from x_spanformer.agents.critique_session import CritiqueSession


class TestCritiqueSession(unittest.TestCase):
    def setUp(self):
        # No setup needed as CritiqueSession is stateless in these tests
        pass

    @patch("x_spanformer.agents.critique_session.load_selfcrit_config")
    @patch("x_spanformer.agents.critique_session.chat", new_callable=AsyncMock)
    def test_run_session_early_exit(self, mock_chat, mock_load_config):
        # Test that the session exits early if the model returns a decisive status
        mock_load_config.return_value = {
            "templates": {"system": "system prompt", "score": "score prompt"},
            "evaluation": {"passes": 1},
            "model": {"name": "test_model", "temperature": 0.5},
            "dialogue": {"max_turns": 3},
        }
        mock_chat.side_effect = [
            "Score: 0.9\nStatus: keep\nReason: Looks good.",
        ]
        session = CritiqueSession()
        result = asyncio.run(session.evaluate("Test document"))
        self.assertEqual(result["status"], "keep")
        self.assertEqual(mock_chat.call_count, 1)

    @patch("x_spanformer.agents.critique_session.load_selfcrit_config")
    @patch("x_spanformer.agents.critique_session.chat", new_callable=AsyncMock)
    def test_run_session_max_turns(self, mock_chat, mock_load_config):
        # Test that the session stops after max_turns is reached
        mock_load_config.return_value = {
            "templates": {"system": "system prompt", "score": "score prompt"},
            "evaluation": {"passes": 3},
            "model": {"name": "test_model", "temperature": 0.5},
            "dialogue": {"max_turns": 3},
        }
        mock_chat.side_effect = [
            "Score: 0.6\nStatus: revise\nReason: Needs more work.",
            "Score: 0.7\nStatus: revise\nReason: Still not quite there.",
            "Score: 0.8\nStatus: revise\nReason: Almost there.",
        ]
        session = CritiqueSession()
        result = asyncio.run(session.evaluate("Test document"))
        self.assertEqual(result["status"], "revise")
        self.assertEqual(mock_chat.call_count, 3)

    @patch("x_spanformer.agents.critique_session.load_selfcrit_config")
    @patch("x_spanformer.agents.critique_session.chat", new_callable=AsyncMock)
    def test_run_session_unparseable_response(self, mock_chat, mock_load_config):
        # Test how the session handles an unparseable response from the model
        mock_load_config.return_value = {
            "templates": {"system": "system prompt", "score": "score prompt"},
            "evaluation": {"passes": 2},
            "model": {"name": "test_model", "temperature": 0.5},
            "dialogue": {"max_turns": 3},
        }
        mock_chat.side_effect = [
            "This is not a valid response.",
            "Score: 0.9\nStatus: keep\nReason: Looks good.",
        ]
        session = CritiqueSession()
        result = asyncio.run(session.evaluate("Test document"))
        self.assertEqual(result["status"], "keep")
        self.assertEqual(mock_chat.call_count, 2)

    @patch("x_spanformer.agents.critique_session.load_selfcrit_config")
    @patch("x_spanformer.agents.critique_session.chat", new_callable=AsyncMock)
    def test_evaluate_regex_filter(self, mock_chat, mock_load_config):
        # Test that the regex filter is applied and the session auto-discards the text
        mock_load_config.return_value = {
            "templates": {"system": "system prompt", "score": "score prompt"},
            "evaluation": {"passes": 1},
            "model": {"name": "test_model", "temperature": 0.5},
            "dialogue": {"max_turns": 3},
            "regex_filters": [{"pattern": "bad word"}],
        }
        mock_chat.side_effect = [
            "Score: 0.9\nStatus: keep\nReason: Looks good.",
        ]
        session = CritiqueSession()
        result = asyncio.run(session.evaluate("this is a bad word"))
        self.assertEqual(result["status"], "discard")
        self.assertEqual(result["reason"], "regex filter triggered")
        mock_chat.assert_not_called()

    @patch("x_spanformer.agents.critique_session.load_selfcrit_config")
    @patch("x_spanformer.agents.critique_session.chat", new_callable=AsyncMock)
    def test_evaluate_consensus(self, mock_chat, mock_load_config):
        # Test that consensus is reached when no decisive response is returned
        mock_load_config.return_value = {
            "templates": {"system": "system prompt", "score": "score prompt"},
            "evaluation": {"passes": 3},
            "model": {"name": "test_model", "temperature": 0.5},
            "dialogue": {"max_turns": 3},
        }
        mock_chat.side_effect = [
            "Score: 0.6\nStatus: revise\nReason: not sure",
            "Score: 0.7\nStatus: revise\nReason: maybe",
            "Score: 0.5\nStatus: revise\nReason: one more time",
        ]
        session = CritiqueSession()
        result = asyncio.run(session.evaluate("this is a text for consensus"))
        self.assertEqual(result["status"], "revise")
        self.assertEqual(mock_chat.call_count, 3)

    def test_parse(self):
        session = CritiqueSession()
        parsed = session.parse("Score: 0.8\nStatus: keep\nReason: looks good")
        self.assertEqual(parsed["score"], 0.8)
        self.assertEqual(parsed["status"], "keep")
        self.assertEqual(parsed["reason"], "looks good")

        bad_raw = "invalid response"
        parsed = session.parse(bad_raw)
        self.assertEqual(parsed["status"], "revise")

    def test_resolve(self):
        session = CritiqueSession()
        votes = [
            {"score": 0.6, "status": "revise", "reason": "not sure"},
            {"score": 0.7, "status": "revise", "reason": "maybe"},
            {"score": 0.8, "status": "keep", "reason": "looks good"},
        ]
        result = session.resolve(votes)
        self.assertEqual(result["status"], "revise")
        self.assertAlmostEqual(result["score"], 0.7)
