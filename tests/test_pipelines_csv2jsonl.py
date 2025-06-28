import asyncio
import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from x_spanformer.pipelines.csv2jsonl import run as csv2jsonl_run


class TestCsv2JsonlPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.test_dir))

        self.input_dir = Path(self.test_dir) / "input"
        self.input_dir.mkdir()

        self.output_dir = Path(self.test_dir) / "output"
        self.output_dir.mkdir()

        # Create a dummy CSV file
        self.csv_file_path = self.input_dir / "test.csv"
        with open(self.csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow(["This is a good sentence."])
            writer.writerow(["This is a bad sentence."])

    @patch("x_spanformer.pipelines.csv2jsonl.load_selfcrit_config")
    @patch("x_spanformer.pipelines.csv2jsonl.judge_segment", new_callable=AsyncMock)
    def test_csv_to_jsonl_conversion(self, mock_judge_segment, mock_load_config):
        """
        Test that the CSV to JSONL conversion works as expected, with mocked self-crit.
        """
        # Mock config
        mock_load_config.return_value = {
            "model": {"name": "mock-model", "temperature": 0},
            "evaluation": {"passes": 1, "max_retries": 1},
            "logging": {
                "log_queries": False,
                "log_responses": False,
                "track_consensus": False,
            },
            "templates": {"some_template": "template_text"},
        }

        # Configure the mock to return different results for different inputs
        async def side_effect(text):
            if "good" in text:
                return {"score": 0.9, "status": "keep", "reason": "good"}
            else:
                return {"score": 0.1, "status": "discard", "reason": "bad"}

        mock_judge_segment.side_effect = side_effect

        # Run the pipeline
        csv2jsonl_run(
            i=self.csv_file_path,
            o=self.output_dir,
            f="text",
            pretty=False,
            n="output",
            w=1,
        )

        output_file = self.output_dir / "output.jsonl"
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

            record1 = json.loads(lines[0])
            record2 = json.loads(lines[1])

            # Check that the good sentence was kept and the bad one was discarded (by checking tags)
            self.assertEqual(record1["raw"], "This is a good sentence.")
            self.assertEqual(record1["meta"]["tags"], [])  # keep means no tags

            self.assertEqual(record2["raw"], "This is a bad sentence.")
            self.assertEqual(record2["meta"]["tags"], ["discard"])


if __name__ == "__main__":
    unittest.main()
