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
            i=csv_path, o=self.output_dir, f="text", pretty=True, n="test_dataset", w=1, save_interval=0
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

    @patch("x_spanformer.pipelines.csv2jsonl.judge_segment", new_callable=AsyncMock)
    def test_csv2jsonl_incremental_save(self, mock_judge_segment):
        """Test incremental saving functionality"""
        csv_path = self.test_data_dir / "test_incremental.csv"
        test_rows = [
            {"text": f"Sample text {i}"} for i in range(15)  # 15 rows to test multiple saves
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text"])
            writer.writeheader()
            writer.writerows(test_rows)

        # Mock all segments to be "keep" status
        mock_judge_segment.return_value = {"score": 0.8, "status": "keep", "reason": "test segment"}
        
        # Run with save_interval=5
        csv2jsonl.run(
            i=csv_path, o=self.output_dir, f="text", pretty=False, n="incremental_test", w=1, save_interval=5
        )
        
        # Check that final output exists
        output_file = self.output_dir / "incremental_test.jsonl"
        self.assertTrue(output_file.exists())
        
        # Verify all records are in final output
        records = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        
        self.assertEqual(len(records), 15)
        
        # Verify no temporary files remain
        temp_files = list(self.output_dir.glob("incremental_test_temp_*.jsonl"))
        self.assertEqual(len(temp_files), 0, "Temporary files should be cleaned up")

    @patch("x_spanformer.pipelines.csv2jsonl.judge_segment", new_callable=AsyncMock)
    def test_csv2jsonl_no_incremental_save(self, mock_judge_segment):
        """Test that incremental saving can be disabled"""
        csv_path = self.test_data_dir / "test_no_incremental.csv"
        test_rows = [{"text": f"Sample text {i}"} for i in range(5)]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text"])
            writer.writeheader()
            writer.writerows(test_rows)

        mock_judge_segment.return_value = {"score": 0.8, "status": "keep", "reason": "test segment"}
        
        # Run with save_interval=0 (disabled)
        csv2jsonl.run(
            i=csv_path, o=self.output_dir, f="text", pretty=False, n="no_incremental", w=1, save_interval=0
        )
        
        # Check that final output exists
        output_file = self.output_dir / "no_incremental.jsonl"
        self.assertTrue(output_file.exists())
        
        # Verify no temporary files were created
        temp_files = list(self.output_dir.glob("no_incremental_temp_*.jsonl"))
        self.assertEqual(len(temp_files), 0, "No temporary files should be created when incremental saving is disabled")

    def test_manifest_function(self):
        """Test the manifest loading functionality"""
        # Create a test CSV
        csv_path = self.test_data_dir / "test_manifest.csv"
        csv_path.touch()
        
        # Test without manifest
        src, tool = csv2jsonl.manifest(csv_path)
        self.assertEqual(src, "test_manifest.csv")
        self.assertEqual(tool, "unknown")
        
        # Create a manifest file
        manifest_dir = self.test_data_dir / "test_manifest"
        manifest_dir.mkdir(exist_ok=True)
        manifest_file = manifest_dir / "test_manifest.json"
        
        manifest_data = {"csv": "custom_name.csv"}
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)
        
        # Test with manifest
        src, tool = csv2jsonl.manifest(csv_path)
        self.assertEqual(src, "custom_name.csv")
        self.assertEqual(tool, "pdf2seg (manifest v1)")

    def test_show_summary(self):
        """Test the summary display function"""
        from collections import Counter
        import io
        from contextlib import redirect_stdout
        
        stats = Counter({"keep": 5, "revise": 3, "discard": 2})
        reasons = ["good quality", "too short", "good quality", "noisy", "good quality"]
        
        # Capture output (this is just to ensure the function doesn't crash)
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            csv2jsonl.show_summary("test.csv", stats, reasons)
        
        # The function should execute without errors
        self.assertTrue(True)  # If we get here, the function didn't crash

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

    @patch("x_spanformer.pipelines.csv2jsonl.judge_segment", new_callable=AsyncMock)
    def test_single_file_incremental_saving(self, mock_judge):
        """Test that incremental saving writes to a single file, not multiple temp files."""
        mock_judge.side_effect = [
            {"score": 0.8, "status": "keep", "reason": "good text"},
            {"score": 0.3, "status": "discard", "reason": "poor quality"},
            {"score": 0.7, "status": "keep", "reason": "acceptable"},
            {"score": 0.9, "status": "keep", "reason": "excellent"},
        ]

        csv_content = """text
"Text one"
"Text two"
"Text three"
"Text four"
"""
        csv_file = self.test_data_dir / "test.csv"
        csv_file.write_text(csv_content)

        # Process with save_interval=2 (save after every 2 segments)
        result = csv2jsonl.rows(
            csv_file, "text", 1, {"regex_filters": []}, 
            save_interval=2, output_path=self.output_dir, base_name="test_dataset"
        )

        # Should have processed all 4 records
        self.assertEqual(len(result), 4)

        # Check that only a single dataset.jsonl file exists (no temp files)
        dataset_file = self.output_dir / "test_dataset.jsonl"
        self.assertTrue(dataset_file.exists())
        
        # Verify no temporary files were left behind
        temp_files = list(self.output_dir.glob("test_dataset_temp_*.jsonl"))
        self.assertEqual(len(temp_files), 0, "No temporary files should exist")
        
        # Verify the content of the single file
        records = []
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(
                        pretrain_record.PretrainRecord.model_validate(json.loads(line))
                    )
        
        # Should have all 4 records in the single file
        self.assertEqual(len(records), 4)

    @patch("x_spanformer.pipelines.csv2jsonl.judge_segment", new_callable=AsyncMock)
    def test_incremental_saving_preserves_final_file(self, mock_judge):
        """Test that when incremental saving is used, the final file write preserves all data."""
        mock_judge.side_effect = [
            {"score": 0.8, "status": "keep", "reason": "good text"},
            {"score": 0.7, "status": "keep", "reason": "acceptable"},
        ]

        csv_content = """text
"Sample text one"
"Sample text two"
"""
        csv_file = self.test_data_dir / "test.csv"
        csv_file.write_text(csv_content)

        # Run the full pipeline with incremental saving
        csv2jsonl.run(
            csv_file, self.output_dir, "text", False, "test_dataset", 1, save_interval=1
        )

        # Check that the final file contains all records
        dataset_file = self.output_dir / "test_dataset.jsonl"
        self.assertTrue(dataset_file.exists())
        
        records = []
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(
                        pretrain_record.PretrainRecord.model_validate(json.loads(line))
                    )
        
        # Should have both records in the final file
        self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
