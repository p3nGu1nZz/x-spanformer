import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ollama
from x_spanformer.pipelines import pdf2jsonl
from x_spanformer.schema import pretrain_record


class TestPdf2JsonlPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.tmp_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    @patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock)
    def test_pdf2jsonl_pipeline(self, mock_judge_segment):
        pdf_path = self.tmp_dir / "test.pdf"
        pdf_path.touch()
        
        csv_content = """text
"The quick brown fox."
"This is a test."
"""
        temp_csv_dir = self.output_dir / "temp_csv"
        temp_csv_dir.mkdir(parents=True, exist_ok=True)
        csv_file = temp_csv_dir / "test.csv"
        csv_file.write_text(csv_content)
        
        def mock_judge_side_effect(text):
            if "fox" in text:
                return {"score": 0.9, "status": "keep", "reason": "good example"}
            return {"score": 0.4, "status": "discard", "reason": "too simple"}

        mock_judge_segment.side_effect = mock_judge_side_effect
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", return_value=csv_file):
            pdf2jsonl.run(
                i=pdf_path, o=self.output_dir, f="text", pretty=True, n="test_dataset", w=1, save_interval=0
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

        records.sort(key=lambda r: r.raw)

        record1, record2 = records
        self.assertEqual(record1.raw, "The quick brown fox.")
        self.assertEqual(record1.meta.tags, [])
        self.assertEqual(record2.raw, "This is a test.")
        self.assertEqual(record2.meta.tags, ["discard"])

    def test_run_pdf2seg_success(self):
        pdf_path = self.tmp_dir / "test.pdf"
        pdf_path.touch()
        output_dir = self.tmp_dir / "csv_output"
        output_dir.mkdir()
        
        csv_file = output_dir / "test.csv"
        csv_file.write_text("text\n\"sample text\"")
        
        # Mock the pdf2seg module at import time
        mock_pdf2seg = MagicMock()
        mock_pdf2seg.load.return_value = MagicMock()
        mock_pdf2seg.pdf.return_value = None
        mock_pdf2seg.extract.return_value = []
        mock_pdf2seg.save_csv.return_value = None
        
        with patch.dict('sys.modules', {'pdf2seg': mock_pdf2seg}):
            result = pdf2jsonl.run_pdf2seg(pdf_path, output_dir)
            
            self.assertEqual(result, csv_file)
            mock_pdf2seg.load.assert_called_once_with("en")
            mock_pdf2seg.pdf.assert_called_once()
            mock_pdf2seg.extract.assert_called_once()
            mock_pdf2seg.save_csv.assert_called_once()

    def test_run_pdf2seg_failure(self):
        pdf_path = self.tmp_dir / "test.pdf"
        pdf_path.touch()
        output_dir = self.tmp_dir / "csv_output"
        output_dir.mkdir()
        
        mock_pdf2seg = MagicMock()
        mock_pdf2seg.load.side_effect = Exception("Processing error")
        
        with patch.dict('sys.modules', {'pdf2seg': mock_pdf2seg}):
            result = pdf2jsonl.run_pdf2seg(pdf_path, output_dir)
            
            self.assertIsNone(result)

    def test_run_pdf2seg_import_error(self):
        pdf_path = self.tmp_dir / "test.pdf"
        pdf_path.touch()
        output_dir = self.tmp_dir / "csv_output"
        output_dir.mkdir()
        
        # Mock importlib to raise ImportError only for pdf2seg
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'pdf2seg':
                raise ImportError("No module named 'pdf2seg'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = pdf2jsonl.run_pdf2seg(pdf_path, output_dir)
            self.assertIsNone(result)

    @patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock)
    def test_rows_function(self, mock_judge):
        mock_judge.side_effect = [
            {"score": 0.8, "status": "keep", "reason": "good text"},
            {"score": 0.3, "status": "discard", "reason": "poor quality"},
        ]

        csv_content = """text
"Sample text one"
"Sample text two"
"""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = pdf2jsonl.rows(
            csv_file, "text", 1, {"regex_filters": []}
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].raw, "Sample text one")
        self.assertEqual(result[0].meta.tags, [])
        self.assertEqual(result[1].raw, "Sample text two")
        self.assertEqual(result[1].meta.tags, ["discard"])

    @patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock)
    def test_single_file_incremental_saving(self, mock_judge):
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
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = pdf2jsonl.rows(
            csv_file, "text", 1, {"regex_filters": []}, 
            save_interval=2, output_path=self.output_dir, base_name="test_dataset"
        )

        self.assertEqual(len(result), 4)

        dataset_file = self.output_dir / "test_dataset.jsonl"
        self.assertTrue(dataset_file.exists())
        
        temp_files = list(self.output_dir.glob("test_dataset_temp_*.jsonl"))
        self.assertEqual(len(temp_files), 0, "No temporary files should exist")
        
        records = []
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(
                        pretrain_record.PretrainRecord.model_validate(json.loads(line))
                    )
        
        self.assertEqual(len(records), 4)

    def test_incremental_saving_preserves_final_file(self):
        with patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock) as mock_judge:
            mock_judge.side_effect = [
                {"score": 0.8, "status": "keep", "reason": "good text"},
                {"score": 0.7, "status": "keep", "reason": "acceptable"},
            ]

            csv_content = """text
"Sample text one"
"Sample text two"
"""
            csv_file = self.tmp_dir / "test.csv"
            csv_file.write_text(csv_content)

            with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", return_value=csv_file):
                pdf_path = self.tmp_dir / "test.pdf"
                pdf_path.touch()
                
                pdf2jsonl.run(
                    pdf_path, self.output_dir, "text", False, "test_dataset", 1, save_interval=1
                )

                dataset_file = self.output_dir / "test_dataset.jsonl"
                self.assertTrue(dataset_file.exists())
                
                records = []
                with dataset_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            records.append(
                                pretrain_record.PretrainRecord.model_validate(json.loads(line))
                            )
                
                self.assertEqual(len(records), 2)

    def test_manifest_loading(self):
        test_dir = self.tmp_dir / "test_doc"
        test_dir.mkdir()
        
        manifest_file = test_dir / "test_doc.json"
        manifest_data = {"csv": "original.csv", "other": "data"}
        with manifest_file.open("w") as f:
            json.dump(manifest_data, f)
        
        csv_file = test_dir.parent / "test_doc.csv"
        csv_file.touch()
        
        source, tool = pdf2jsonl.manifest(csv_file)
        self.assertEqual(source, "original.csv")
        self.assertEqual(tool, "pdf2seg (manifest v1)")

    def test_manifest_fallback(self):
        csv_file = self.tmp_dir / "no_manifest.csv"
        csv_file.touch()
        
        source, tool = pdf2jsonl.manifest(csv_file)
        self.assertEqual(source, "no_manifest.csv")
        self.assertEqual(tool, "unknown")

    def test_show_summary(self):
        from collections import Counter
        from io import StringIO
        import sys
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        stats = Counter({"keep": 5, "discard": 2, "revise": 1})
        reasons = ["good quality", "good quality", "poor format", "good quality"]
        
        pdf2jsonl.show_summary("test.csv", stats, reasons)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        self.assertIn("keep", output.lower())
        self.assertIn("discard", output.lower())
        self.assertIn("revise", output.lower())

    def test_multiple_pdfs(self):
        pdf_dir = self.tmp_dir / "pdfs"
        pdf_dir.mkdir()
        
        pdf1 = pdf_dir / "doc1.pdf"
        pdf2 = pdf_dir / "doc2.pdf"
        pdf1.touch()
        pdf2.touch()
        
        with patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock) as mock_judge:
            mock_judge.side_effect = [
                {"score": 0.8, "status": "keep", "reason": "good"},
                {"score": 0.9, "status": "keep", "reason": "excellent"},
            ]
            
            temp_csv_dir = self.output_dir / "temp_csv"
            temp_csv_dir.mkdir(parents=True)
            
            csv1 = temp_csv_dir / "doc1.csv"
            csv2 = temp_csv_dir / "doc2.csv"
        csv1.write_text('text\n"Text from doc1"')
        csv2.write_text('text\n"Text from doc2"')
        
        def mock_pdf2seg(pdf_file, output_dir):
            if pdf_file.name == "doc1.pdf":
                return csv1
            elif pdf_file.name == "doc2.pdf":
                return csv2
            return None
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", side_effect=mock_pdf2seg):
            pdf2jsonl.run(pdf_dir, self.output_dir, "text", False, "test_dataset", 1, save_interval=0)
        
        output_file = self.output_dir / "test_dataset.jsonl"
        self.assertTrue(output_file.exists())
        
        records = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        self.assertEqual(len(records), 2)

    def test_ollama_integration(self):
        try:
            client = ollama.Client()
            ps_response = client.ps()
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
            self.skipTest(f"Ollama client failed to connect or query: {e}")

    @patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock)
    def test_end_to_end_pdf_to_jsonl_workflow(self, mock_judge):
        """Test the complete workflow from PDF to JSONL."""
        pdf_path = self.tmp_dir / "document.pdf"
        pdf_path.touch()
        
        mock_judge.side_effect = [
            {"score": 0.9, "status": "keep", "reason": "high quality content"},
            {"score": 0.2, "status": "discard", "reason": "low quality content"},
            {"score": 0.8, "status": "keep", "reason": "good content"},
        ]
        
        temp_csv_dir = self.output_dir / "temp_csv"
        temp_csv_dir.mkdir(parents=True)
        generated_csv = temp_csv_dir / "document.csv"
        generated_csv.write_text('''text
"This is high quality text content."
"Bad text here."
"Another good piece of content."
''')
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", return_value=generated_csv):
            pdf2jsonl.run(pdf_path, self.output_dir, "text", True, "final_dataset", 1, save_interval=0)
        
        jsonl_file = self.output_dir / "final_dataset.jsonl"
        json_file = self.output_dir / "final_dataset.json"
        
        self.assertTrue(jsonl_file.exists(), "JSONL output file should exist")
        self.assertTrue(json_file.exists(), "Pretty JSON output file should exist")
        
        records = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = pretrain_record.PretrainRecord.model_validate(json.loads(line))
                    records.append(record)
        
        self.assertEqual(len(records), 3, "Should have 3 records")
        
        kept_records = [r for r in records if not r.meta.tags]
        discarded_records = [r for r in records if r.meta.tags]
        
        self.assertEqual(len(kept_records), 2, "Should have 2 kept records")
        self.assertEqual(len(discarded_records), 1, "Should have 1 discarded record")
        
        with json_file.open("r", encoding="utf-8") as f:
            pretty_data = json.load(f)
            self.assertEqual(len(pretty_data), 3, "Pretty JSON should have same number of records")

    def test_no_pdfs_found(self):
        """Test behavior when no PDF files are found in the input directory."""
        empty_dir = self.tmp_dir / "empty"
        empty_dir.mkdir()
        
        from io import StringIO
        import sys
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            pdf2jsonl.run(empty_dir, self.output_dir, "text", False, "dataset", 1, save_interval=0)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("No PDFs found", output)

    def test_single_pdf_processing(self):
        """Test processing a single PDF file rather than a directory."""
        pdf_file = self.tmp_dir / "single.pdf"
        pdf_file.touch()
        
        csv_content = '''text
"Single PDF text content."
'''
        temp_csv_dir = self.output_dir / "temp_csv"
        temp_csv_dir.mkdir(parents=True)
        generated_csv = temp_csv_dir / "single.csv"
        generated_csv.write_text(csv_content)
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", return_value=generated_csv):
            with patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock) as mock_judge:
                mock_judge.return_value = {"score": 0.8, "status": "keep", "reason": "good"}
                
                pdf2jsonl.run(pdf_file, self.output_dir, "text", False, "single_dataset", 1, save_interval=0)
        
        output_file = self.output_dir / "single_dataset.jsonl"
        self.assertTrue(output_file.exists())
        
        with output_file.open("r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 1)

    @patch("x_spanformer.pipelines.pdf2jsonl.judge_segment", new_callable=AsyncMock)
    def test_partial_pdf2seg_failure(self, mock_judge):
        """Test behavior when pdf2seg fails for some PDFs but succeeds for others."""
        pdf_dir = self.tmp_dir / "mixed_pdfs"
        pdf_dir.mkdir()
        
        good_pdf = pdf_dir / "good.pdf"
        bad_pdf = pdf_dir / "bad.pdf"
        good_pdf.touch()
        bad_pdf.touch()
        
        mock_judge.return_value = {"score": 0.8, "status": "keep", "reason": "good"}
        
        temp_csv_dir = self.output_dir / "temp_csv"
        temp_csv_dir.mkdir(parents=True)
        good_csv = temp_csv_dir / "good.csv"
        good_csv.write_text('text\n"Good PDF content"')
        
        def mock_pdf2seg_mixed(pdf_file, output_dir):
            if pdf_file.name == "good.pdf":
                return good_csv
            return None
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", side_effect=mock_pdf2seg_mixed):
            pdf2jsonl.run(pdf_dir, self.output_dir, "text", False, "mixed_dataset", 1, save_interval=0)
        
        output_file = self.output_dir / "mixed_dataset.jsonl"
        self.assertTrue(output_file.exists())
        
        with output_file.open("r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 1, "Should only process the successful PDF")


if __name__ == "__main__":
    unittest.main()
