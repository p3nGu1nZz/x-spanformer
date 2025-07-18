import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from ollama._types import ResponseError
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.pipelines import pdf2jsonl
from x_spanformer.agents import ollama_client


class TestErrorHandling:
    """Test error handling improvements in ollama_client and pdf2jsonl."""
    
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.tmp_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    @pytest.mark.asyncio
    async def test_ollama_connection_check_with_model_not_loaded(self):
        """Test that check_ollama_connection properly detects when model is not loaded."""
        with patch('x_spanformer.agents.ollama_client.check_ollama_running', return_value=True):
            with patch('subprocess.run') as mock_run:
                # Mock ollama ps output without the target model
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "NAME\t\tID\t\tSIZE\t\tMODIFIED\nother-model:latest\t123\t1.2GB\t5 minutes ago\n"
                
                result = await ollama_client.check_ollama_connection("phi4-mini")
                assert result is False

    @pytest.mark.asyncio
    async def test_ollama_connection_check_with_model_loaded(self):
        """Test that check_ollama_connection properly detects when model is loaded."""
        with patch('x_spanformer.agents.ollama_client.check_ollama_running', return_value=True):
            with patch('subprocess.run') as mock_run:
                # Mock ollama ps output with the target model
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "NAME\t\tID\t\tSIZE\t\tMODIFIED\nphi4-mini:latest\t123\t1.2GB\t5 minutes ago\n"
                
                result = await ollama_client.check_ollama_connection("phi4-mini")
                assert result is True

    @patch("x_spanformer.pipelines.pdf2jsonl.check_ollama_connection")
    @patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg")
    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    def test_consecutive_retry_errors_exit(self, mock_judge_session_class, mock_run_pdf2seg, mock_ollama_check):
        """Test that 3 consecutive RetryErrors cause immediate exit."""
        mock_ollama_check.return_value = True
        
        # Create test PDF and CSV
        pdf_file = self.tmp_dir / "test.pdf"
        pdf_file.touch()
        csv_content = "text\n\"Test text\""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)
        mock_run_pdf2seg.return_value = csv_file
        
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock evaluate to always raise an exception that looks like a RetryError
        def mock_retry_error(*args):
            # Create an exception that will match our error detection logic
            raise Exception("RetryError: ConnectionError: Connection refused")
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_retry_error)
        
        # Mock the JudgeSession class completely to avoid initialization issues
        mock_judge_session_class.side_effect = lambda *args, **kwargs: mock_judge_session
        
        # Test that the function exits with sys.exit(1) after 3 consecutive RetryErrors
        with patch('sys.exit') as mock_exit:
            with patch('x_spanformer.pipelines.pdf2jsonl.load_judge_config', return_value={
                "judge": {"judges": 5, "threshold": 0.69, "model_name": "phi4-mini", "temperature": 0.1},
                "processor": {"max_raw_length": 512},
                "dialogue": {"max_turns": 1},
                "templates": {"system": "test", "judge": "test"}
            }):
                pdf2jsonl.run(
                    pdf_file, self.output_dir, "text", False, "test_dataset", 1, save_interval=1
                )
                
                # sys.exit should have been called
                mock_exit.assert_called_once_with(1)

    @patch("x_spanformer.pipelines.pdf2jsonl.check_ollama_connection")
    @patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg")
    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    def test_all_judges_failed_exit(self, mock_judge_session_class, mock_run_pdf2seg, mock_ollama_check):
        """Test that when all judges fail for a segment, system exits immediately."""
        mock_ollama_check.return_value = True
        
        # Create test PDF and CSV
        pdf_file = self.tmp_dir / "test.pdf"
        pdf_file.touch()
        csv_content = "text\n\"Test text\""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)
        mock_run_pdf2seg.return_value = csv_file
        
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock evaluate to always raise non-retry errors
        timeout_error = asyncio.TimeoutError("Judge evaluation timeout")
        mock_judge_session.evaluate = AsyncMock(side_effect=timeout_error)
        
        # Mock the JudgeSession class completely to avoid initialization issues
        mock_judge_session_class.side_effect = lambda *args, **kwargs: mock_judge_session
        
        # Test that the function exits when all judges fail
        with patch('sys.exit') as mock_exit:
            with patch('x_spanformer.pipelines.pdf2jsonl.load_judge_config', return_value={
                "judge": {"judges": 5, "threshold": 0.69, "model_name": "phi4-mini", "temperature": 0.1},
                "processor": {"max_raw_length": 512},
                "dialogue": {"max_turns": 1},
                "templates": {"system": "test", "judge": "test"}
            }):
                pdf2jsonl.run(
                    pdf_file, self.output_dir, "text", False, "test_dataset", 1, save_interval=1
                )
                
                # sys.exit should have been called
                mock_exit.assert_called_once_with(1)

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    def test_partial_judge_success_continues(self, mock_judge_session_class):
        """Test that if some judges succeed, processing continues with partial consensus."""
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock evaluate to succeed on some calls and fail on others
        call_count = 0
        def mock_evaluate(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First 2 judges succeed
                return {"score": 0.8, "type": "natural", "reason": "good text"}
            else:  # Remaining judges fail with RetryError
                raise Exception("RetryError: ConnectionError: Connection refused")
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock the JudgeSession class completely to avoid initialization issues
        mock_judge_session_class.side_effect = lambda *args, **kwargs: mock_judge_session
        
        # Create test CSV
        csv_content = "text\n\"Test text\""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)
        
        # Mock config to have 5 judges
        mock_config = {
            "judge": {"judges": 5, "threshold": 0.69, "model_name": "phi4-mini", "temperature": 0.1},
            "processor": {"max_raw_length": 512},
            "dialogue": {"max_turns": 1},
            "templates": {"system": "test", "judge": "test"}
        }
        
        with patch('x_spanformer.pipelines.pdf2jsonl.load_judge_config', return_value=mock_config):
            result = pdf2jsonl.process_all_csvs(
                [csv_file], "text", 1, {}, save_interval=0
            )
            
            # Should process successfully with partial judges
            assert len(result) == 1
            assert result[0].meta.status == "keep"  # 0.8 score > 0.69 threshold
            assert result[0].raw == "Test text"

    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    def test_error_counter_reset_on_success(self, mock_judge_session_class):
        """Test that consecutive error counter resets on successful evaluation."""
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock evaluate to fail twice, then succeed
        call_count = 0
        def mock_evaluate(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First 2 calls fail
                raise Exception("RetryError: ConnectionError: Connection refused")
            else:  # Subsequent calls succeed
                return {"score": 0.8, "type": "natural", "reason": "good text"}
        
        mock_judge_session.evaluate = AsyncMock(side_effect=mock_evaluate)
        
        # Mock the JudgeSession class completely to avoid initialization issues
        mock_judge_session_class.side_effect = lambda *args, **kwargs: mock_judge_session
        
        # Create test CSV
        csv_content = "text\n\"Test text\""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)
        
        # Mock config to have 5 judges
        mock_config = {
            "judge": {"judges": 5, "threshold": 0.69, "model_name": "phi4-mini", "temperature": 0.1},
            "processor": {"max_raw_length": 512},
            "dialogue": {"max_turns": 1},
            "templates": {"system": "test", "judge": "test"}
        }
        
        with patch('x_spanformer.pipelines.pdf2jsonl.load_judge_config', return_value=mock_config):
            result = pdf2jsonl.process_all_csvs(
                [csv_file], "text", 1, {}, save_interval=0
            )
            
            # Should process successfully - error counter resets after success
            assert len(result) == 1
            assert result[0].meta.status == "keep"
            assert result[0].raw == "Test text"

    @patch("x_spanformer.pipelines.pdf2jsonl.check_ollama_connection")
    @patch("x_spanformer.pipelines.pdf2jsonl.run_pdf2seg")
    @patch("x_spanformer.pipelines.pdf2jsonl.JudgeSession")
    def test_non_retry_error_increments_counter(self, mock_judge_session_class, mock_run_pdf2seg, mock_ollama_check):
        """Test that non-RetryError exceptions also increment the consecutive error counter."""
        mock_ollama_check.return_value = True
        
        # Create test PDF and CSV
        pdf_file = self.tmp_dir / "test.pdf"
        pdf_file.touch()
        csv_content = "text\n\"Test text\""
        csv_file = self.tmp_dir / "test.csv"
        csv_file.write_text(csv_content)
        mock_run_pdf2seg.return_value = csv_file
        
        # Mock JudgeSession
        mock_judge_session = MagicMock()
        mock_judge_session_class.return_value = mock_judge_session
        
        # Mock evaluate to always raise timeout errors (non-retry errors)
        timeout_error = asyncio.TimeoutError("Judge evaluation timeout")
        mock_judge_session.evaluate = AsyncMock(side_effect=timeout_error)
        
        # Mock the JudgeSession class completely to avoid initialization issues
        mock_judge_session_class.side_effect = lambda *args, **kwargs: mock_judge_session
        
        # Test that the function exits immediately on non-retry errors
        with patch('sys.exit') as mock_exit:
            with patch('x_spanformer.pipelines.pdf2jsonl.load_judge_config', return_value={
                "judge": {"judges": 5, "threshold": 0.69, "model_name": "phi4-mini", "temperature": 0.1},
                "processor": {"max_raw_length": 512},
                "dialogue": {"max_turns": 1},
                "templates": {"system": "test", "judge": "test"}
            }):
                pdf2jsonl.run(
                    pdf_file, self.output_dir, "text", False, "test_dataset", 1, save_interval=1
                )
                
                # sys.exit should have been called
                mock_exit.assert_called_once_with(1)
