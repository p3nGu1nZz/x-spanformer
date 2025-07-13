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

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    def test_pdf2jsonl_pipeline(self, mock_judge_session_class):
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
        
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        async def mock_evaluate(text):
            if "fox" in text:
                return {"score": 0.9, "status": "keep", "reason": "good example"}
            return {"score": 0.4, "status": "discard", "reason": "too simple"}
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock ImproveSession - REMOVED (no longer used in simplified pipeline)
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", return_value=csv_file):
            pdf2jsonl.run(
                i=pdf_path, o=self.output_dir, f="text", pretty=True, n="test_dataset", w=1, save_interval=1
            )
        
        output_file = self.output_dir / "jsonl" / "test_dataset.jsonl"
        self.assertTrue(output_file.exists())
        records = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(
                        pretrain_record.PretrainRecord.model_validate(json.loads(line))
                    )
        self.assertEqual(len(records), 1)  # Only "fox" text meets threshold in 5-judge consensus

        # Should only have the "fox" text since it's the only one above threshold
        self.assertIn("fox", records[0].raw)

        # Check the single record that met the threshold
        record = records[0]
        self.assertEqual(record.raw, "The quick brown fox.")
        self.assertEqual(record.meta.tags, [])
        self.assertEqual(record.meta.source_file, "test.pdf")  # Should be original PDF name

    def test_run_pdf2seg_success(self):
        pdf_path = self.tmp_dir / "test.pdf"
        pdf_path.touch()
        output_dir = self.tmp_dir / "csv_output"
        output_dir.mkdir()
        
        # The function uses hash-based naming: hash_name("test.pdf") = "056c935e"
        expected_csv_file = output_dir / "056c935e.csv"
        # Don't create the CSV file beforehand - let the function create it
        
        # Mock the pdf2seg module at import time
        mock_pdf2seg = MagicMock()
        mock_pdf2seg.load.return_value = MagicMock()
        mock_pdf2seg.pdf.return_value = None
        mock_pdf2seg.extract.return_value = []
        mock_pdf2seg.save_csv.return_value = None
        
        with patch.dict('sys.modules', {'pdf2seg': mock_pdf2seg}):
            result = pdf2jsonl.run_pdf2seg(pdf_path, output_dir)
    
            self.assertEqual(result, expected_csv_file)
            mock_pdf2seg.load.assert_called_once_with("en_core_web_sm")

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

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    def test_process_all_csvs_function(self, mock_judge_session_class):
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Create a deterministic mock that returns different results based on text content
        def mock_evaluate(text):
            if "one" in text:
                return {"score": 0.8, "status": "keep", "reason": "good text"}
            else:  # "two" in text
                return {"score": 0.3, "status": "discard", "reason": "poor quality"}
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock ImproveSession - REMOVED (no longer used in simplified pipeline)

        csv_content = """text
"Sample text one"
"Sample text two"
"""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = pdf2jsonl.process_all_csvs(
            [csv_file], "text", 1, {}, save_interval=0
        )

        self.assertEqual(len(result), 2)  # All processed records are returned
        
        # Sort results by raw text to avoid dependency on async execution order
        result.sort(key=lambda x: x.raw)
        
        # Check both records exist but with different statuses
        self.assertEqual(result[0].raw, "Sample text one")
        self.assertEqual(result[0].meta.status, "keep")  # Above threshold
        self.assertEqual(result[0].meta.tags, [])  # "keep" records have no tags
        
        self.assertEqual(result[1].raw, "Sample text two") 
        self.assertEqual(result[1].meta.status, "discard")  # Below threshold
        self.assertEqual(result[1].meta.tags, ["discard"])  # "discard" records get tagged
        self.assertEqual(result[1].meta.tags, ["discard"])

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    # @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession") - REMOVED
    def test_single_file_incremental_saving(self, mock_judge_session_class):
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock judge results - each segment gets 5 judge calls for consensus
        def mock_evaluate(text):
            if "one" in text:
                return {"score": 0.8, "status": "keep", "reason": "good text"}
            elif "two" in text:
                return {"score": 0.3, "status": "discard", "reason": "poor quality"}
            elif "three" in text:
                return {"score": 0.7, "status": "keep", "reason": "acceptable"}
            elif "four" in text:
                return {"score": 0.9, "status": "keep", "reason": "excellent"}
            return {"score": 0.5, "status": "discard", "reason": "default"}
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock ImproveSession
        # mock_improve_session = MagicMock() - REMOVED
        # mock_improve_session_class.return_value = ... - REMOVED
        # mock_improve_session.improve = ... - REMOVED

        csv_content = """text
"Text one"
"Text two"
"Text three"
"Text four"
"""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = pdf2jsonl.process_all_csvs(
            [csv_file], "text", 1, {}, 
            save_interval=2, output_path=self.output_dir, base_name="test_dataset"
        )

        self.assertEqual(len(result), 4)  # All processed records are returned
        
        # Check status of all records (sorted alphabetically by raw text)
        result.sort(key=lambda x: x.raw)  # Sort for consistent ordering
        
        # After sorting: "Text four", "Text one", "Text three", "Text two"
        # "Text four" -> 0.9 >= 0.69 -> keep
        self.assertEqual(result[0].meta.status, "keep")
        # "Text one" -> 0.8 >= 0.69 -> keep
        self.assertEqual(result[1].meta.status, "keep")
        # "Text three" -> 0.7 >= 0.69 -> keep
        self.assertEqual(result[2].meta.status, "keep") 
        # "Text two" -> 0.3 < 0.69 -> discard
        self.assertEqual(result[3].meta.status, "discard")

        dataset_file = self.output_dir / "jsonl" / "test_dataset.jsonl"
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
        
        self.assertEqual(len(records), 3)  # Only the 3 "keep" records should be saved

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    # @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession") - REMOVED
    def test_incremental_saving_preserves_final_file(self, mock_judge_session_class):
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock judge results - each segment gets 5 judge calls for consensus
        def mock_evaluate(text):
            if "one" in text:
                return {"score": 0.8, "status": "keep", "reason": "good text"}
            elif "two" in text:
                return {"score": 0.7, "status": "keep", "reason": "acceptable"}
            return {"score": 0.5, "status": "discard", "reason": "default"}
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock ImproveSession
        # mock_improve_session = MagicMock() - REMOVED
        # mock_improve_session_class.return_value = ... - REMOVED
        # mock_improve_session.improve = ... - REMOVED

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

            dataset_file = self.output_dir / "jsonl" / "test_dataset.jsonl"
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
        
        # Create CSV file in the parent directory
        csv_file = self.tmp_dir / "test_doc.csv"
        csv_file.touch()
        
        # Create manifest file in the same directory as the CSV
        manifest_file = self.tmp_dir / "test_doc.json"
        manifest_data = {"csv": "original.csv", "other": "data"}
        with manifest_file.open("w") as f:
            json.dump(manifest_data, f)
        
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
        
        stats = Counter({"keep": 5, "discard": 3})
        reasons = ["good quality", "good quality", "poor format", "good quality"]
        
        # Use the actual display function from rich_utils
        from x_spanformer.agents.rich_utils import display_summary_panel
        display_summary_panel("test.csv", stats, reasons)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        self.assertIn("keep", output.lower())
        self.assertIn("discard", output.lower())
        self.assertIn("discard", output.lower())

    def test_multiple_pdfs(self):
        pdf_dir = self.tmp_dir / "pdfs"
        pdf_dir.mkdir()

        pdf1 = pdf_dir / "doc1.pdf"
        pdf2 = pdf_dir / "doc2.pdf"
        pdf1.touch()
        pdf2.touch()

        with patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession") as mock_judge_session_class:
            # with patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession") as mock_improve_session_class: - REMOVED
                mock_judge_session_class.return_value = MagicMock()
                # mock_improve_session_class.return_value = ... - REMOVED

                # Mock judge results - each segment gets 5 judge calls for consensus
                def mock_evaluate(text):
                    if "doc1" in text:
                        return {"score": 0.8, "status": "keep", "reason": "good"}
                    elif "doc2" in text:
                        return {"score": 0.9, "status": "keep", "reason": "excellent"}
                    return {"score": 0.5, "status": "discard", "reason": "default"}
                
                mock_judge_session_class.return_value.evaluate = AsyncMock(side_effect=mock_evaluate)
                # mock_improve_session_class.return_value.improve = AsyncMock(return_value=(None, "Natural")) - REMOVED

                temp_csv_dir = self.output_dir / "temp_csv"
                temp_csv_dir.mkdir(parents=True)

                csv1 = temp_csv_dir / "doc1.csv"
                csv2 = temp_csv_dir / "doc2.csv"
                csv1.write_text('text\n"Text from doc1"')
                csv2.write_text('text\n"Text from doc2"')

                def mock_pdf2seg(pdf_file, output_dir, force_regenerate=False):
                    if pdf_file.name == "doc1.pdf":
                        return csv1
                    elif pdf_file.name == "doc2.pdf":
                        return csv2
                    return None

                with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", side_effect=mock_pdf2seg):
                    pdf2jsonl.run(pdf_dir, self.output_dir, "text", False, "test_dataset", 1, save_interval=1)

        output_file = self.output_dir / "jsonl" / "test_dataset.jsonl"
        self.assertTrue(output_file.exists())
        
        records = []
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        self.assertEqual(len(records), 2)

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
            with patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession") as mock_judge_session_class:
                mock_judge_session_class.return_value = MagicMock()
                mock_judge_session_class.return_value.evaluate = AsyncMock(
                    return_value={"score": 0.8, "status": "keep", "reason": "good"}
                )
                
                pdf2jsonl.run(pdf_file, self.output_dir, "text", False, "single_dataset", 1, save_interval=1)
        
        output_file = self.output_dir / "jsonl" / "single_dataset.jsonl"
        self.assertTrue(output_file.exists())
        
        with output_file.open("r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 1)

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    # @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession") - REMOVED
    def test_end_to_end_pdf_to_jsonl_workflow(self, mock_judge_session_class):
        """Test the complete workflow from PDF to JSONL."""
        pdf_path = self.tmp_dir / "document.pdf"
        pdf_path.touch()
        
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock judge results - each segment gets 5 judge calls for consensus
        def mock_evaluate(text):
            if "high quality" in text:
                return {"score": 0.9, "status": "keep", "reason": "high quality content"}
            elif "Bad text" in text:
                return {"score": 0.2, "status": "discard", "reason": "low quality content"}
            elif "good piece" in text:
                return {"score": 0.8, "status": "keep", "reason": "good content"}
            return {"score": 0.5, "status": "discard", "reason": "default"}
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock ImproveSession
        # mock_improve_session = MagicMock() - REMOVED
        # mock_improve_session_class.return_value = ... - REMOVED
        # mock_improve_session.improve = ... - REMOVED
        
        temp_csv_dir = self.output_dir / "temp_csv"
        temp_csv_dir.mkdir(parents=True)
        generated_csv = temp_csv_dir / "document.csv"
        generated_csv.write_text('''text
"This is high quality text content."
"Bad text here."
"Another good piece of content."
''')
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", return_value=generated_csv):
            pdf2jsonl.run(pdf_path, self.output_dir, "text", True, "final_dataset", 1, save_interval=1)
        
        jsonl_file = self.output_dir / "jsonl" / "final_dataset.jsonl"
        json_file = self.output_dir / "jsonl" / "final_dataset.json"
        
        self.assertTrue(jsonl_file.exists(), "JSONL output file should exist")
        self.assertTrue(json_file.exists(), "Pretty JSON output file should exist")
        
        records = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = pretrain_record.PretrainRecord.model_validate(json.loads(line))
                    records.append(record)
        
        self.assertEqual(len(records), 2, "Should have 2 records (only the 'keep' ones)")
        
        # All saved records should be "keep" status
        for record in records:
            self.assertEqual(record.meta.status, "keep")
            self.assertEqual(record.meta.tags, [])  # "keep" records have no tags
        
        # Since only "keep" records are saved, there should be no discarded records in the file
        
        with json_file.open("r", encoding="utf-8") as f:
            pretty_data = json.load(f)
            self.assertEqual(len(pretty_data), 3, "Pretty JSON should have same number of records")

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    # @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession") - REMOVED
    def test_partial_pdf2seg_failure(self, mock_judge_session_class):
        """Test behavior when pdf2seg fails for some PDFs but succeeds for others."""
        pdf_dir = self.tmp_dir / "mixed_pdfs"
        pdf_dir.mkdir()
        
        good_pdf = pdf_dir / "good.pdf"
        bad_pdf = pdf_dir / "bad.pdf"
        good_pdf.touch()
        bad_pdf.touch()
        
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        mock_judge_session.evaluate = AsyncMock(return_value={"score": 0.8, "status": "keep", "reason": "good"})
        
        # Mock ImproveSession
        # mock_improve_session = MagicMock() - REMOVED
        # mock_improve_session_class.return_value = ... - REMOVED
        # mock_improve_session.improve = ... - REMOVED
        
        temp_csv_dir = self.output_dir / "temp_csv"
        temp_csv_dir.mkdir(parents=True)
        good_csv = temp_csv_dir / "good.csv"
        good_csv.write_text('text\n"Good PDF content"')
        
        def mock_pdf2seg_mixed(pdf_file, output_dir, force_regenerate=False):
            if pdf_file.name == "good.pdf":
                return good_csv
            return None
        
        with patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg", side_effect=mock_pdf2seg_mixed):
            pdf2jsonl.run(pdf_dir, self.output_dir, "text", False, "mixed_dataset", 1, save_interval=1)
        
        output_file = self.output_dir / "jsonl" / "mixed_dataset.jsonl"
        self.assertTrue(output_file.exists())
        
        with output_file.open("r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
            self.assertEqual(len(lines), 1, "Should only process the successful PDF")

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    # @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession") - REMOVED
    def test_pdf_to_csv_mapping(self, mock_judge_session_class):
        """Test that original PDF filenames are correctly mapped to source_file fields."""
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        mock_judge_session.evaluate = AsyncMock(side_effect=[
            {"score": 0.8, "status": "keep", "reason": "good text"},
            {"score": 0.7, "status": "keep", "reason": "acceptable text"},
        ])
        
        # Mock ImproveSession
        # mock_improve_session = MagicMock() - REMOVED
        # mock_improve_session_class.return_value = ... - REMOVED
        # mock_improve_session.improve = ... - REMOVED

        # Create test CSV content
        csv_content = """text
"Text from first PDF"
"Text from second PDF"
"""
        csv_file1 = self.tmp_dir / "hash1.csv"
        csv_file2 = self.tmp_dir / "hash2.csv"
        csv_file1.write_text('text\n"Text from first PDF"')
        csv_file2.write_text('text\n"Text from second PDF"')

        # Create PDF mapping (simulating the hash mapping from real PDFs)
        pdf_mapping = {
            "hash1.csv": "document1.pdf",
            "hash2.csv": "document2.pdf"
        }

        result = pdf2jsonl.process_all_csvs(
            [csv_file1, csv_file2], "text", 1, {},
            save_interval=0, pdf_mapping=pdf_mapping
        )

        self.assertEqual(len(result), 2)
        
        # Verify that source_file contains original PDF filenames, not CSV filenames
        source_files = [record.meta.source_file for record in result]
        self.assertIn("document1.pdf", source_files)
        self.assertIn("document2.pdf", source_files)
        
        # Verify no CSV filenames in source_file
        self.assertNotIn("hash1.csv", source_files)
        self.assertNotIn("hash2.csv", source_files)
