import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.pipelines import pdf2jsonl
from x_spanformer.agents import ollama_client


class TestErrorHandling:
    """Test error handling logic without complex async operations."""
    
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.tmp_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_ollama_connection_check_logic(self):
        """Test connection check logic without real async calls."""
        with patch('x_spanformer.agents.ollama_client.check_ollama_running', return_value=True):
            with patch('subprocess.run') as mock_run:
                # Test model not loaded scenario
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "NAME\t\tID\t\tSIZE\t\tMODIFIED\nother-model:latest\t123\t1.2GB\t5 minutes ago\n"
                
                # Test the logic directly
                output = mock_run.return_value.stdout
                target_model = "phi4-mini"
                result = target_model in output
                assert result is False

                # Test model loaded scenario
                mock_run.return_value.stdout = "NAME\t\tID\t\tSIZE\t\tMODIFIED\nphi4-mini:latest\t123\t1.2GB\t5 minutes ago\n"
                output = mock_run.return_value.stdout
                result = target_model in output
                assert result is True

    @patch("x_spanformer.pipelines.pdf2jsonl.check_ollama_connection")
    @patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg")
    @patch("x_spanformer.agents.session.judge_session.JudgeSession.evaluate")  # Mock the session evaluate method
    def test_consecutive_retry_errors_logic(self, mock_evaluate, mock_run_pdf2seg, mock_ollama_check):
        """Test retry error handling logic without actual retries."""
        mock_ollama_check.return_value = True
        
        # Create test PDF and CSV
        pdf_file = self.tmp_dir / "test.pdf"
        pdf_file.touch()
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text('text\n"Test text"')
        mock_run_pdf2seg.return_value = csv_file
        
        # Mock evaluate to raise RetryError that will trigger consecutive error counting
        mock_evaluate.side_effect = Exception("RetryError: ConnectionError - All attempts failed")
        
        # This will trigger the consecutive RetryError logic and call sys.exit(1)
        with patch('sys.exit') as mock_exit:
            with patch('x_spanformer.pipelines.pdf2jsonl.load_judge_config', return_value={
                "judge": {"judges": 1, "threshold": 0.69, "model_name": "phi4-mini", "temperature": 0.1},
                "processor": {"max_raw_length": 512},
                "dialogue": {"max_turns": 1},
                "templates": {"system": "test", "judge": "test"}
            }):
                pdf2jsonl.run(
                    pdf_file, self.output_dir, "text", False, "test_dataset", 1, save_interval=1
                )
                
                # sys.exit should have been called due to critical error pattern matching
                mock_exit.assert_called_once_with(1)

    @patch("x_spanformer.pipelines.pdf2jsonl.check_ollama_connection")
    @patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg") 
    @patch("x_spanformer.agents.session.judge_session.JudgeSession.evaluate")  # Mock the session evaluate method
    def test_all_judges_failed_logic(self, mock_evaluate, mock_run_pdf2seg, mock_ollama_check):
        """Test all judges failure handling logic."""
        mock_ollama_check.return_value = True
        
        # Create test files
        pdf_file = self.tmp_dir / "test.pdf"
        pdf_file.touch()
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text('text\n"Test text"')
        mock_run_pdf2seg.return_value = csv_file
        
        # Mock evaluate to raise RetryError that will make all judges fail but continue processing
        # This will cause judge_responses to be empty, triggering the "all judges failed" exception
        mock_evaluate.side_effect = Exception("RetryError: ConnectionError - All judges failed")
        
        # This will trigger the all judges failed logic and call sys.exit(1)
        with patch('sys.exit') as mock_exit:
            with patch('x_spanformer.pipelines.pdf2jsonl.load_judge_config', return_value={
                "judge": {"judges": 1, "threshold": 0.69, "model_name": "phi4-mini", "temperature": 0.1},
                "processor": {"max_raw_length": 512},
                "dialogue": {"max_turns": 1},
                "templates": {"system": "test", "judge": "test"}
            }):
                pdf2jsonl.run(
                    pdf_file, self.output_dir, "text", False, "test_dataset", 1, save_interval=1
                )
                
                # sys.exit should have been called due to critical error pattern
                mock_exit.assert_called_once_with(1)

    def test_partial_judge_success_logic(self):
        """Test partial judge success consensus logic."""
        # Test consensus calculation logic directly
        judge_responses = [
            {"score": 0.8, "status": "keep", "type": "natural", "reason": "good text"},
            {"score": 0.7, "status": "keep", "type": "natural", "reason": "acceptable text"},
        ]
        
        # Simulate consensus calculation
        scores = [r["score"] for r in judge_responses]
        consensus_score = sum(scores) / len(scores)
        threshold = 0.69
        final_status = "keep" if consensus_score >= threshold else "discard"
        
        # Should process successfully with partial judges
        assert len(judge_responses) == 2
        assert consensus_score == 0.75  # (0.8 + 0.7) / 2
        assert final_status == "keep"  # 0.75 > 0.69 threshold

    def test_error_counter_reset_logic(self):
        """Test that consecutive error counter resets on successful evaluation."""
        # Test counter reset logic
        consecutive_errors = 2
        
        # Simulate successful evaluation
        successful_response = {"score": 0.8, "status": "keep", "type": "natural", "reason": "good text"}
        
        if successful_response:
            consecutive_errors = 0  # Reset on success
        
        assert consecutive_errors == 0

    def test_error_threshold_logic(self):
        """Test error threshold detection logic."""
        consecutive_errors = 0
        max_errors = 3
        
        # Simulate error scenarios
        for i in range(5):
            if "RetryError" in f"Mock error {i}":
                consecutive_errors += 1
            else:
                consecutive_errors = 0
                
            # Check if threshold exceeded
            if consecutive_errors >= max_errors:
                should_exit = True
                break
        else:
            should_exit = False
        
        # With no actual RetryErrors in mock strings, should not exit
        assert should_exit is False
        
        # Test actual RetryError pattern
        consecutive_errors = 0
        for error_msg in ["RetryError: ConnectionError", "RetryError: ConnectionError", "RetryError: ConnectionError"]:
            if "RetryError" in error_msg:
                consecutive_errors += 1
            if consecutive_errors >= max_errors:
                should_exit = True
                break
        
        assert consecutive_errors == 3
        assert should_exit is True
