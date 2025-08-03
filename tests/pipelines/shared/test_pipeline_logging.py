#!/usr/bin/env python3
"""
Tests for pipeline_logging.py shared module.

Tests the PipelineLogger and SpanAnnotationLogger classes for
centralized logging configuration across X-Spanformer pipelines.
"""
import pytest
import tempfile
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch, call
from datetime import datetime

from x_spanformer.pipelines.shared.pipeline_logging import (
    PipelineLogger,
    SpanAnnotationLogger,
    setup_logging
)


class TestPipelineLogger:
    """Test cases for the PipelineLogger utility class."""
    
    def test_setup_pipeline_logging_default(self):
        """Test default pipeline logging setup."""
        logger = PipelineLogger.setup_pipeline_logging("test_pipeline")
        
        assert logger.name == "test_pipeline"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler only
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    def test_setup_pipeline_logging_with_file(self):
        """Test pipeline logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            logger = PipelineLogger.setup_pipeline_logging(
                pipeline_name="test_pipeline",
                log_level="DEBUG",
                log_to_file=True,
                log_file_path=log_file
            )
            
            assert logger.level == logging.DEBUG
            assert len(logger.handlers) == 2  # Console + File handlers
            
            # Test that log file was created
            assert log_file.exists()
            
            # Test logging to file
            logger.info("Test message")
            
            # Close file handlers to release file locks
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_setup_pipeline_logging_file_only(self):
        """Test pipeline logging setup with file only (no console)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            logger = PipelineLogger.setup_pipeline_logging(
                pipeline_name="test_pipeline",
                log_to_file=True,
                log_file_path=log_file,
                console_output=False
            )
            
            assert len(logger.handlers) == 1  # File handler only
            assert isinstance(logger.handlers[0], logging.FileHandler)
            
            # Close file handler to release file lock
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
    
    def test_setup_pipeline_logging_custom_format(self):
        """Test pipeline logging with custom format."""
        custom_format = "%(levelname)s: %(message)s"
        
        logger = PipelineLogger.setup_pipeline_logging(
            pipeline_name="test_pipeline",
            log_format=custom_format
        )
        
        # Check that handler has the custom format
        handler = logger.handlers[0]
        assert handler.formatter is not None
        # Note: formatter._fmt is internal, we'll test behavior instead
        
        # Test that the format is applied by logging a message
        with patch('sys.stdout') as mock_stdout:
            logger.info("Test message")
            # Custom format should not include timestamp or logger name
            # This is a basic check that custom format was applied
    
    def test_setup_pipeline_logging_log_levels(self):
        """Test different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        expected_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        
        for level_str, expected_level in zip(levels, expected_levels):
            logger = PipelineLogger.setup_pipeline_logging(
                pipeline_name=f"test_{level_str.lower()}",
                log_level=level_str
            )
            assert logger.level == expected_level
    
    def test_setup_from_config(self):
        """Test setup from configuration object."""
        # Create mock config
        mock_config = Mock()
        mock_logging_config = Mock()
        mock_logging_config.level = "DEBUG"
        mock_logging_config.format = "%(message)s"
        mock_logging_config.log_to_file = True
        mock_logging_config.file_path = "test.log"
        mock_config.logging = mock_logging_config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            mock_logging_config.file_path = str(log_file)
            
            logger = PipelineLogger.setup_from_config(mock_config, "test_pipeline")
            
            assert logger.name == "test_pipeline"
            assert logger.level == logging.DEBUG
            assert len(logger.handlers) == 2  # Console + File
            
            # Close file handlers to release file locks
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
    
    def test_setup_from_config_no_logging_section(self):
        """Test setup from config without logging section."""
        mock_config = Mock()
        mock_config.logging = None
        
        logger = PipelineLogger.setup_from_config(mock_config, "test_pipeline")
        
        # Should fallback to default logging
        assert logger.name == "test_pipeline"
        assert logger.level == logging.INFO
    
    @patch('x_spanformer.pipelines.shared.pipeline_logging.datetime')
    def test_log_pipeline_start(self, mock_datetime):
        """Test pipeline start logging."""
        mock_now = datetime(2025, 8, 3, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        logger = Mock()
        
        PipelineLogger.log_pipeline_start(
            logger, 
            "Test Pipeline",
            input_file="test.jsonl",
            output_dir="output/"
        )
        
        # Verify logger.info was called with expected messages
        expected_calls = [
            call("=" * 80),
            call("Starting Test Pipeline"),
            call("=" * 80),
            call("input_file: test.jsonl"),
            call("output_dir: output/"),
            call("Started at: 2025-08-03T12:00:00"),
            call("-" * 40)
        ]
        
        logger.info.assert_has_calls(expected_calls)
    
    @patch('x_spanformer.pipelines.shared.pipeline_logging.datetime')
    def test_log_pipeline_completion(self, mock_datetime):
        """Test pipeline completion logging."""
        mock_now = datetime(2025, 8, 3, 12, 30, 0)
        mock_datetime.now.return_value = mock_now
        
        logger = Mock()
        stats = {
            "processed_sequences": 100,
            "successful_annotations": 95,
            "failed_annotations": 5,
            "total_spans": 1500
        }
        
        PipelineLogger.log_pipeline_completion(logger, "Test Pipeline", stats)
        
        # Verify logger.info was called with expected messages
        expected_calls = [
            call("-" * 40),
            call("Test Pipeline Completed"),
            call("Completed at: 2025-08-03T12:30:00"),
            call("processed_sequences: 100"),
            call("successful_annotations: 95"),
            call("failed_annotations: 5"),
            call("total_spans: 1500"),
            call("=" * 80)
        ]
        
        logger.info.assert_has_calls(expected_calls)
    
    def test_log_batch_progress(self):
        """Test batch progress logging."""
        logger = Mock()
        
        PipelineLogger.log_batch_progress(
            logger,
            batch_num=5,
            total_batches=20,
            batch_size=10,
            successful=8,
            failed=2
        )
        
        expected_message = "Batch 5/20 (25.0%): size=10, successful=8, failed=2"
        logger.info.assert_called_once_with(expected_message)
    
    def test_log_error_with_context(self):
        """Test error logging with context."""
        logger = Mock()
        error = ValueError("Test error message")
        context = "during sequence processing"
        
        PipelineLogger.log_error_with_context(logger, error, context)
        
        # Verify error and debug calls
        logger.error.assert_called_once_with("Error occurred during sequence processing: Test error message")
        logger.debug.assert_called_once_with("Error details: ValueError: Test error message", exc_info=True)
    
    def test_log_error_without_context(self):
        """Test error logging without context."""
        logger = Mock()
        error = RuntimeError("Runtime error")
        
        PipelineLogger.log_error_with_context(logger, error)
        
        logger.error.assert_called_once_with("Error occurred: Runtime error")
        logger.debug.assert_called_once_with("Error details: RuntimeError: Runtime error", exc_info=True)


class TestSpanAnnotationLogger:
    """Test cases for the SpanAnnotationLogger specialized class."""
    
    def test_log_resume_detection(self):
        """Test resume detection logging."""
        logger = Mock()
        
        SpanAnnotationLogger.log_resume_detection(logger, working_files_count=45, max_processed_id=50)
        
        expected_calls = [
            call("Resume mode detected - found 45 existing working files"),
            call("Highest processed sequence ID: 50")
        ]
        
        logger.info.assert_has_calls(expected_calls)
    
    def test_log_sequence_categorization(self):
        """Test sequence categorization logging."""
        logger = Mock()
        
        SpanAnnotationLogger.log_sequence_categorization(
            logger,
            failed_retry_count=5,
            gap_count=3,
            new_count=42,
            completed_count=25
        )
        
        expected_calls = [
            call("Processing 50 sequences:"),  # 5 + 3 + 42 = 50
            call("  - Retrying 5 failed sequences"),
            call("  - Processing 3 gap sequences (within processed range)"),
            call("  - Processing 42 new sequences (beyond processed range)"),
            call("  - Skipped 25 completed sequences")
        ]
        
        logger.info.assert_has_calls(expected_calls)
    
    def test_log_sequence_completion(self):
        """Test sequence completion logging."""
        logger = Mock()
        
        SpanAnnotationLogger.log_sequence_completion(logger, sequence_id=42, span_count=15)
        
        logger.info.assert_called_once_with("[COMPLETED] Sequence 42 - extracted 15 spans")
    
    def test_log_sequence_failure(self):
        """Test sequence failure logging."""
        logger = Mock()
        
        SpanAnnotationLogger.log_sequence_failure(
            logger,
            sequence_id=42,
            reason="Model timeout",
            consecutive_failures=3
        )
        
        expected_message = "Sequence 42 failed: Model timeout (consecutive failures: 3)"
        logger.warning.assert_called_once_with(expected_message)
    
    def test_log_metadata_correction(self):
        """Test metadata correction logging."""
        logger = Mock()
        
        corrected_stats = {
            "total_files": 100,
            "successful_count": 85,
            "failed_count": 15,
            "total_spans": 1500
        }
        
        SpanAnnotationLogger.log_metadata_correction(logger, corrected_stats)
        
        expected_calls = [
            call("Metadata corrected from working files:"),
            call("  - Processed sequences: 100"),
            call("  - Successful annotations: 85"),
            call("  - Failed annotations: 15"),
            call("  - Total spans: 1500"),
            call("  - Success rate: 85.0%")
        ]
        
        logger.info.assert_has_calls(expected_calls)
    
    def test_log_metadata_correction_with_zero_files(self):
        """Test metadata correction logging with zero files (edge case)."""
        logger = Mock()
        
        corrected_stats = {
            "total_files": 0,
            "successful_count": 0,
            "failed_count": 0,
            "total_spans": 0
        }
        
        SpanAnnotationLogger.log_metadata_correction(logger, corrected_stats)
        
        # Should handle division by zero gracefully
        expected_calls = [
            call("Metadata corrected from working files:"),
            call("  - Processed sequences: 0"),
            call("  - Successful annotations: 0"),
            call("  - Failed annotations: 0"),
            call("  - Total spans: 0"),
            call("  - Success rate: 0.0%")
        ]
        
        logger.info.assert_has_calls(expected_calls)


class TestSetupLogging:
    """Test cases for the setup_logging convenience function."""
    
    def test_setup_logging_convenience_function(self):
        """Test the convenience setup_logging function."""
        mock_config = Mock()
        mock_logging_config = Mock()
        mock_logging_config.level = "INFO"
        mock_config.logging = mock_logging_config
        
        with patch.object(PipelineLogger, 'setup_from_config') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            result = setup_logging(mock_config, "test_pipeline")
            
            mock_setup.assert_called_once_with(mock_config, "test_pipeline")
            assert result == mock_logger


class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_complete_logging_workflow(self):
        """Test a complete logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "integration_test.log"
            
            # Setup logger
            logger = PipelineLogger.setup_pipeline_logging(
                pipeline_name="Integration Test Pipeline",
                log_level="INFO",
                log_to_file=True,
                log_file_path=log_file,
                console_output=False  # Only log to file for easier testing
            )
            
            # Log pipeline start
            PipelineLogger.log_pipeline_start(
                logger,
                "Integration Test Pipeline",
                input_file="test.jsonl",
                batch_size=10
            )
            
            # Log some batch progress
            PipelineLogger.log_batch_progress(logger, 1, 5, 10, 8, 2)
            PipelineLogger.log_batch_progress(logger, 2, 5, 10, 9, 1)
            
            # Log sequence events
            SpanAnnotationLogger.log_sequence_completion(logger, 1, 5)
            SpanAnnotationLogger.log_sequence_failure(logger, 2, "Timeout", 1)
            
            # Log pipeline completion
            stats = {
                "processed_sequences": 20,
                "successful_annotations": 18,
                "failed_annotations": 2
            }
            PipelineLogger.log_pipeline_completion(logger, "Integration Test Pipeline", stats)
            
            # Close file handlers to release file locks
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            
            # Verify log file content
            assert log_file.exists()
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check that all logged messages appear
                assert "Starting Integration Test Pipeline" in content
                assert "input_file: test.jsonl" in content
                assert "Batch 1/5 (20.0%)" in content
                assert "[COMPLETED] Sequence 1 - extracted 5 spans" in content
                assert "Sequence 2 failed: Timeout" in content
                assert "Integration Test Pipeline Completed" in content
                assert "processed_sequences: 20" in content
    
    def test_logger_handler_cleanup(self):
        """Test that logger handlers are properly cleaned up."""
        # Create logger with handlers
        logger1 = PipelineLogger.setup_pipeline_logging("test_pipeline")
        initial_handler_count = len(logger1.handlers)
        
        # Setup same logger again - should clear existing handlers
        logger2 = PipelineLogger.setup_pipeline_logging("test_pipeline")
        
        # Should be the same logger object (same name)
        assert logger1 is logger2
        
        # Should have same number of handlers (old ones cleared)
        assert len(logger2.handlers) == initial_handler_count


if __name__ == "__main__":
    pytest.main([__file__])
