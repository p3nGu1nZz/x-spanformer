import asyncio
import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama
from x_spanformer.pipelines import csv2jsonl
from x_spanformer.schema import pretrain_record


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

    @patch("x_spanformer.pipelines.csv2jsonl.judge_segment", new_callable=AsyncMock)
    def test_csv2jsonl_pipeline(self, mock_judge_segment):
        csv_path = self.test_data_dir / "test.csv"
        test_rows = [
            {"text": "The quick brown fox."},
            {"text": "This is a test."},
            {"text": ""},
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text"])
            writer.writeheader()
            writer.writerows(test_rows)

        def mock_judge_side_effect(text):
            if "fox" in text:
                return {"score": 0.9, "status": "keep", "reason": "good example"}
            return {"score": 0.4, "status": "discard", "reason": "too simple"}

        mock_judge_segment.side_effect = mock_judge_side_effect
        csv2jsonl.run(
            i=csv_path, o=self.output_dir, f="text", pretty=True, n="test_dataset", w=1
        )
        output_file = self.output_dir / "test_dataset.jsonl"
        self.assertTrue(output_file.exists())
        records = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(
                    pretrain_record.PretrainRecord.model_validate(json.loads(line))
                )
        self.assertEqual(len(records), 2)

        # Sort records by raw text to ensure deterministic order for assertions
        records.sort(key=lambda r: r.raw)

        record1, record2 = records
        self.assertEqual(record1.raw, "The quick brown fox.")
        self.assertEqual(record1.meta.tags, [])
        self.assertEqual(record2.raw, "This is a test.")
        self.assertEqual(record2.meta.tags, ["discard"])

    def test_ollama_gpu_usage(self):
        # This test is a placeholder to demonstrate checking GPU usage.
        try:
            client = ollama.Client()
            ps_response = client.ps()
            self.assertIn("models", ps_response)
            gpu_in_use = any(
                "gpu"
                in model.get("details", {}).get("parameter_size", "")
                for model in ps_response["models"]
            )
            if not gpu_in_use:
                print(
                    "\n[Warning] Ollama does not appear to be using the GPU for any loaded models."
                )
        except Exception as e:
            self.fail(f"Ollama client failed to connect or query: {e}")


if __name__ == "__main__":
    unittest.main()
