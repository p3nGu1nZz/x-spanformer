import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.pipelines import pdf2jsonl


class TestImprovedFieldBug(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession")
    def test_improved_field_not_null_when_improvement_made(self, mock_improve_session_class, mock_judge_session_class):
        """Test that improved field is not null when real improvement iterations occurred."""
        # Mock JudgeSession to return different scores for original vs improved text
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        original_text = "This is the original text."
        improved_text = "This is the improved and enhanced text with better clarity."
        
        def mock_evaluate(text):
            if text == original_text:
                return {"score": 0.6, "status": "revise", "reason": "needs improvement"}
            elif text == improved_text:
                return {"score": 0.8, "status": "keep", "reason": "good improvement"}
            else:
                return {"score": 0.6, "status": "revise", "reason": "needs improvement"}
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock ImproveSession to return actual improved text
        mock_improve_session = MagicMock()
        mock_improve_session_class.return_value = mock_improve_session
        mock_improve_session.improve = AsyncMock(return_value=(improved_text, "Natural"))

        # Create test CSV
        csv_content = f'text\n"{original_text}"'
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)

        # Process the CSV
        result = pdf2jsonl.process_all_csvs(
            [csv_file], "text", 1, {}, save_interval=0
        )

        # Verify results
        self.assertEqual(len(result), 1)
        record = result[0]
        
        # The improved field should NOT be null when improvement was made
        self.assertIsNotNone(record.improved, "improved field should not be null when improvement was made")
        self.assertEqual(record.improved, improved_text, "improved field should contain the improved text")
        self.assertEqual(record.raw, original_text, "raw field should contain the original text")
        
        # Check that improvement iterations were recorded
        self.assertIsNotNone(record.meta.notes)
        if record.meta.notes:
            self.assertIn("Improvement iterations:", record.meta.notes)

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession")
    def test_improved_field_null_when_no_improvement_made(self, mock_improve_session_class, mock_judge_session_class):
        """Test that improved field is null when no improvement was actually made."""
        # Mock JudgeSession to return "keep" status (no improvement needed)
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        mock_judge_session.evaluate = AsyncMock(return_value={
            "score": 0.8, "status": "keep", "reason": "good text"
        })
        
        # Mock ImproveSession (shouldn't be called in this case)
        mock_improve_session = MagicMock()
        mock_improve_session_class.return_value = mock_improve_session
        mock_improve_session.improve = AsyncMock(return_value=(None, "Natural"))

        # Create test CSV
        original_text = "This is good text that needs no improvement."
        csv_content = f'text\n"{original_text}"'
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)

        # Process the CSV
        result = pdf2jsonl.process_all_csvs(
            [csv_file], "text", 1, {}, save_interval=0
        )

        # Verify results
        self.assertEqual(len(result), 1)
        record = result[0]
        
        # The improved field should be null when no improvement was made
        self.assertIsNone(record.improved, "improved field should be null when no improvement was made")
        self.assertEqual(record.raw, original_text, "raw field should contain the original text")
        
        # Check that no improvement iterations were recorded
        if record.meta.notes:
            self.assertNotIn("Improvement iterations:", record.meta.notes)

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    @patch("x_spanformer.pipelines.pdf2jsonl.ImproveSession")
    def test_improved_field_null_when_improvement_returns_same_text(self, mock_improve_session_class, mock_judge_session_class):
        """Test that improved field is null when improvement returns the same text as original."""
        # Mock JudgeSession to return "revise" status to trigger improvement attempts
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        mock_judge_session.evaluate = AsyncMock(return_value={
            "score": 0.6, "status": "revise", "reason": "needs improvement"
        })
        
        # Mock ImproveSession to return the same text (no actual improvement)
        mock_improve_session = MagicMock()
        mock_improve_session_class.return_value = mock_improve_session
        
        original_text = "This text cannot be improved."
        # Improvement returns the same text
        mock_improve_session.improve = AsyncMock(return_value=(original_text, "Natural"))

        # Create test CSV
        csv_content = f'text\n"{original_text}"'
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)

        # Process the CSV
        result = pdf2jsonl.process_all_csvs(
            [csv_file], "text", 1, {}, save_interval=0
        )

        # Verify results
        self.assertEqual(len(result), 1)
        record = result[0]
        
        # The improved field should be null when improvement returns same text
        self.assertIsNone(record.improved, "improved field should be null when improvement returns same text")
        self.assertEqual(record.raw, original_text, "raw field should contain the original text")
        
        # Check that improvement iterations were attempted but no useful improvement
        self.assertIsNotNone(record.meta.notes)
        if record.meta.notes:
            self.assertIn("Improvement iterations:", record.meta.notes)


if __name__ == "__main__":
    unittest.main()
