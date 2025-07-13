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

    @patch("x_spanformer.pipelines.pdf2jsonl.process_segment_cycle")
    def test_improved_field_not_null_when_improvement_made(self, mock_process_segment_cycle):
        """Test that improved field is not null when real improvement iterations occurred."""
        
        original_text = "This is the original text."
        improved_text = "This is the improved and enhanced text with better clarity."
        
        # Mock process_segment_cycle to simulate improvement cycle with final improved text
        mock_process_segment_cycle.return_value = {
            "score": 0.8,
            "status": "keep", 
            "reason": "good improvement",
            "final_text": improved_text,
            "cycles_completed": 2
        }

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

    @patch("x_spanformer.pipelines.pdf2jsonl.process_segment_cycle")
    def test_improved_field_null_when_no_improvement_made(self, mock_process_segment_cycle):
        """Test that improved field is null when no improvement was actually made."""
        
        original_text = "This is good text that needs no improvement."
        
        # Mock process_segment_cycle to return "keep" status (no improvement needed)
        mock_process_segment_cycle.return_value = {
            "score": 0.8,
            "status": "keep", 
            "reason": "good text",
            "final_text": original_text,
            # Don't include cycles_completed to simulate immediate keep without improvement attempts
        }

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
        
        # Check that no improvement iterations were recorded when cycles_completed is not present
        if record.meta.notes:
            self.assertNotIn("Improvement iterations:", record.meta.notes)

    @patch("x_spanformer.pipelines.pdf2jsonl.process_segment_cycle")
    def test_improved_field_null_when_improvement_returns_same_text(self, mock_process_segment_cycle):
        """Test that improved field is null when improvement returns the same text as original."""
        
        original_text = "This text cannot be improved."
        
        # Mock process_segment_cycle to simulate improvement that returns same text
        mock_process_segment_cycle.return_value = {
            "score": 0.8,
            "status": "keep",
            "reason": "improvement completed", 
            "final_text": original_text,  # Same as original
            "cycles_completed": 2
        }

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
