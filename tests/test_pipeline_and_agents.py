import asyncio
import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import yaml
from pydantic import ValidationError
from jinja2 import Environment, FileSystemLoader, meta
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama
from x_spanformer.pipelines import csv2jsonl
from x_spanformer.agents import config_loader, critique_session, dialogue, ollama_client, prompts, selfcrit
from x_spanformer.schema import (
    identifier, metadata, pretrain_record, scoring, source, span, status,
    tokenization, typing, validation, xbar
)


class TestSchemas(unittest.TestCase):
    def test_record_id(self):
        rec_id = identifier.RecordID()
        self.assertIsInstance(rec_id.id, str)
        with self.assertRaises(ValidationError):
            rec_id.id = "new-id"

    def test_record_meta(self):
        meta = metadata.RecordMeta(tags=["test"], doc_language="en", confidence=0.5, extracted_by="test", source_file="test.txt", notes="test")
        self.assertEqual(meta.tags, ["test"])
        self.assertEqual(meta.doc_language, "en")
        self.assertEqual(meta.confidence, 0.5)
        self.assertIsNone(metadata.RecordMeta(doc_language="en", extracted_by="test", source_file="test.txt", notes="test", confidence=None).confidence)

    def test_pretrain_record(self):
        record = pretrain_record.PretrainRecord(raw="test text")
        self.assertEqual(record.raw, "test text")
        self.assertIsInstance(record.id, identifier.RecordID)
        self.assertIsInstance(record.meta, metadata.RecordMeta)

    def test_span_label(self):
        label = span.SpanLabel(span=(0, 1), label="noun", text="test", role="noun")
        self.assertEqual(label.span, (0, 1))
        self.assertEqual(label.label, "noun")

    def test_xp_span(self):
        xp = xbar.XPSpan(span=(0, 2), category="XP", role="specifier", label="NP", text="test")
        self.assertEqual(xp.category, "XP")
        with self.assertRaises(ValidationError):
            xbar.XPSpan.model_validate({"span": (0, 1), "category": "invalid", "role": "specifier", "label": "NP", "text": "test"})

    def test_tokenized_input(self):
        tok_input = tokenization.TokenizedInput(input=["a", "b"], tokenizer="test")
        self.assertEqual(tok_input.input, ["a", "b"])
        self.assertTrue(tok_input.preserve_whitespace)

    def test_record_type(self):
        rec_type = typing.RecordType(type="code")
        self.assertEqual(rec_type.type, "code")
        with self.assertRaises(ValidationError):
            typing.RecordType.model_validate({"type": "invalid"})

    def test_source_info(self):
        src = source.SourceInfo(filename="test.txt", page_number=1, document_id="test", filetype="txt", line_number=1, section="test", source_url="http://test.com")
        self.assertEqual(src.filename, "test.txt")
        self.assertEqual(src.page_number, 1)

    def test_record_status(self):
        stat = status.RecordStatus(stages=["csv_ingested", "validated"])
        self.assertEqual(stat.stages, ["csv_ingested", "validated"])
        with self.assertRaises(ValidationError):
            status.RecordStatus.model_validate({"stages": ["invalid_stage"]})

    def test_entropy_profile(self):
        profile = scoring.EntropyProfile(token_entropy=1.5, span_overlap=0.5, structure_variance=0.5, fluency_score=0.5)
        self.assertEqual(profile.token_entropy, 1.5)

    def test_validation_result(self):
        issue = validation.ValidationIssue(field="test", message="error", severity="error")
        res = validation.ValidationResult(is_valid=False, issues=[issue])
        self.assertFalse(res.is_valid)
        self.assertEqual(len(res.issues), 1)


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
            "templates": {"system": "system_prompt", "score": "score_prompt"}
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
        dm = dialogue.DialogueManager(max_turns=1)
        dm.add("user", "1")
        dm.add("assistant", "2")
        dm.add("user", "3")
        self.assertEqual(len(dm.history), 2)
        self.assertEqual(dm.history[0]["content"], "2")
        self.assertEqual(dm.history[1]["content"], "3")

    @patch("x_spanformer.agents.prompts.env")
    def test_render_prompt(self, mock_env):
        mock_env.get_template.return_value.render = MagicMock(return_value="Hello World")
        result = prompts.render_prompt("test_template", user_name="World")
        self.assertEqual(result, "Hello World")
        mock_env.get_template.assert_called_with("test_template.j2")

    @patch("x_spanformer.agents.ollama_client.AsyncClient")
    def test_ollama_client_chat(self, mock_client):
        mock_response = {"message": {"content": "response"}}
        mock_client.return_value.chat = AsyncMock(return_value=mock_response)
        result = asyncio.run(ollama_client.chat("model", [{"role": "user", "content": "hi"}]))
        self.assertEqual(result, "response")

    def test_selfcrit_parse_response(self):
        raw = "Score: 0.8\nStatus: keep\nReason: looks good"
        parsed = selfcrit.parse_response(raw)
        self.assertEqual(parsed["score"], 0.8)
        self.assertEqual(parsed["status"], "keep")
        self.assertEqual(parsed["reason"], "looks good")
        bad_raw = "invalid response"
        parsed = selfcrit.parse_response(bad_raw)
        self.assertEqual(parsed["status"], "revise")

    @patch.dict(
        "x_spanformer.agents.selfcrit.cfg",
        {
            "evaluation": {"max_retries": 1, "passes": 1},
            "model": {"name": "test"},
            "dialogue": {"max_turns": 1},
            "templates": {"system": "test_prompt", "score": "test_prompt"},
            "regex_filters": [{"pattern": "bad"}],
        },
    )
    @patch("x_spanformer.agents.selfcrit.chat", new_callable=AsyncMock)  
    @patch("x_spanformer.agents.prompts.env")
    def test_judge_segment(self, mock_env, mock_chat):
        with patch("x_spanformer.agents.prompts.env.loader") as mock_loader:
            mock_loader.searchpath = [str(self.templates_dir)]
            mock_chat.return_value = "Score: 0.9\nStatus: keep\nReason: ok"
            result = asyncio.run(selfcrit.judge_segment("good text"))
            self.assertEqual(result["status"], "keep")
            
            # Test regex filter behavior by patching RE_FLAGGED directly inside the test
            with patch("x_spanformer.agents.selfcrit.RE_FLAGGED", [re.compile("bad")]):
                result = asyncio.run(selfcrit.judge_segment("bad text"))
                self.assertEqual(result["status"], "discard")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        self.output_dir = self.test_data_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def test_ollama_gpu_usage(self):
        try:
            client = ollama.Client()
            ps_response = client.ps()
            self.assertIn("models", ps_response)
            gpu_in_use = any('gpu' in model.get('details', {}).get('parameter_size', '') for model in ps_response['models'])
            if not gpu_in_use:
                print("\n[Warning] Ollama does not appear to be using the GPU for any loaded models.")
        except Exception as e:
            self.fail(f"Ollama client failed to connect or query: {e}")

    @patch("x_spanformer.pipelines.csv2jsonl.judge_segment", new_callable=AsyncMock)
    def test_csv2jsonl_pipeline(self, mock_judge_segment):
        csv_path = self.test_data_dir / "test.csv"
        test_rows = [{"text": "The quick brown fox."}, {"text": "This is a test."}, {"text": ""}]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text"])
            writer.writeheader()
            writer.writerows(test_rows)
        def mock_judge_side_effect(text):
            if "fox" in text:
                return {"score": 0.9, "status": "keep", "reason": "good example"}
            return {"score": 0.4, "status": "discard", "reason": "too simple"}
        mock_judge_segment.side_effect = mock_judge_side_effect
        csv2jsonl.run(i=csv_path, o=self.output_dir, f="text", pretty=True, n="test_dataset", w=1)
        output_file = self.output_dir / "test_dataset.jsonl"
        self.assertTrue(output_file.exists())
        records = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(pretrain_record.PretrainRecord.model_validate(json.loads(line)))
        
        records.sort(key=lambda r: r.raw)
        
        self.assertEqual(len(records), 2)
        record1, record2 = records
        self.assertEqual(record1.raw, "The quick brown fox.")
        self.assertEqual(record1.meta.confidence, 0.9)
        self.assertEqual(record1.meta.tags, [])
        self.assertEqual(record2.raw, "This is a test.")
        self.assertEqual(record2.meta.confidence, 0.4)
        self.assertEqual(record2.meta.tags, ["discard"])
        csv_path.unlink()
        output_file.unlink()
        (self.output_dir / "test_dataset.json").unlink()


if __name__ == "__main__":
    unittest.main()
