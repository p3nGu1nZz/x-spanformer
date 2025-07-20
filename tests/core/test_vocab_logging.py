#!/usr/bin/env python3
"""
Test suite for x_spanformer.vocab.vocab_logging module.

Tests centralized logging configuration including:
- Rich handler setup and configuration
- File and console logging coordination
- Logger creation and management
- Error handling for logging setup
"""

import pytest
import tempfile
import logging
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.vocab.vocab_logging import setup_vocab_logging, get_vocab_logger


class TestVocabLogging:
    """Test vocabulary logging functionality."""
    
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.tmp_dir / "logs"
        
        # Clear any existing handlers before each test
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Reset logger states
        logging.getLogger('jsonl2vocab').handlers.clear()
        logging.getLogger('test_logger').handlers.clear()
        
    def teardown_method(self):
        # Close and remove all handlers before cleanup to avoid file locks on Windows
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
            
        # Also clean up logger-specific handlers
        for logger_name in ['jsonl2vocab', 'test_logger']:
            logger_obj = logging.getLogger(logger_name)
            for handler in logger_obj.handlers[:]:
                handler.close()
                logger_obj.removeHandler(handler)
        
        import shutil
        import time
        if self.tmp_dir.exists():
            try:
                shutil.rmtree(self.tmp_dir)
            except PermissionError:
                # On Windows, sometimes we need to wait a bit for file handles to be released
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.tmp_dir)
                except PermissionError:
                    # If still fails, skip cleanup - test tmp dirs will be cleaned by OS
                    pass
    
    def test_setup_vocab_logging_creates_directory(self):
        """Test that setup_vocab_logging creates output directory."""
        logger = setup_vocab_logging(self.log_dir, 'test_logger')
        
        assert self.log_dir.exists()
        assert (self.log_dir / "vocab.log").exists()
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_logger'
    
    def test_setup_vocab_logging_configures_handlers(self):
        """Test that setup_vocab_logging configures file and console handlers."""
        logger = setup_vocab_logging(self.log_dir, 'test_logger')
        
        # Check that root logger has handlers
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2
        
        # Check handler types
        handler_types = [type(h).__name__ for h in root_logger.handlers]
        assert 'FileHandler' in handler_types
        assert 'RichHandler' in handler_types
    
    def test_setup_vocab_logging_file_handler_level(self):
        """Test that file handler captures DEBUG level."""
        logger = setup_vocab_logging(self.log_dir, 'test_logger')
        
        file_handler = None
        for handler in logging.root.handlers:
            if handler.__class__.__name__ == 'FileHandler':
                file_handler = handler
                break
                
        assert file_handler is not None
        assert file_handler.level == logging.DEBUG
    
    def test_setup_vocab_logging_console_handler_level(self):
        """Test that console handler captures INFO level."""
        logger = setup_vocab_logging(self.log_dir, 'test_logger')
        
        console_handler = None
        for handler in logging.root.handlers:
            if handler.__class__.__name__ == 'RichHandler':
                console_handler = handler
                break
                
        assert console_handler is not None
        assert console_handler.level == logging.INFO
    
    def test_setup_vocab_logging_writes_startup_messages(self):
        """Test that setup logging writes startup messages to file."""
        logger = setup_vocab_logging(self.log_dir, 'test_logger')
        
        log_file = self.log_dir / "vocab.log"
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "X-SPANFORMER VOCABULARY INDUCTION PIPELINE" in content
        assert "Pipeline started at:" in content
        assert "Following Algorithm 1" in content
    
    def test_setup_vocab_logging_clears_existing_handlers(self):
        """Test that setup clears existing handlers before adding new ones."""
        # Add some dummy handlers first
        dummy_handler = logging.StreamHandler()
        logging.root.addHandler(dummy_handler)
        initial_count = len(logging.root.handlers)
        
        logger = setup_vocab_logging(self.log_dir, 'test_logger')
        
        # Should have exactly 2 handlers (file + console), not initial + 2
        assert len(logging.root.handlers) == 2
    
    def test_get_vocab_logger_default_name(self):
        """Test get_vocab_logger with default name."""
        logger = get_vocab_logger()
        
        assert logger.name == 'jsonl2vocab'
        assert isinstance(logger, logging.Logger)
    
    def test_get_vocab_logger_custom_name(self):
        """Test get_vocab_logger with custom name."""
        logger = get_vocab_logger('custom_test')
        
        assert logger.name == 'custom_test'
        assert isinstance(logger, logging.Logger)
    
    def test_get_vocab_logger_creates_handler_when_none_exist(self):
        """Test that get_vocab_logger creates handler when none exist."""
        # Ensure no handlers exist
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logger = get_vocab_logger('isolated_test')
        
        # Should create a handler since none exist
        assert len(logger.handlers) > 0 or (logger.parent and len(logger.parent.handlers) > 0)
    
    def test_get_vocab_logger_reuses_existing_setup(self):
        """Test that get_vocab_logger reuses existing logging setup."""
        # First, set up logging properly
        setup_logger = setup_vocab_logging(self.log_dir, 'test_setup')
        
        # Then get logger - should reuse setup
        get_logger = get_vocab_logger('test_setup')
        
        assert setup_logger.name == get_logger.name
        # Should not create additional handlers
        assert len(get_logger.handlers) == 0  # Uses root handlers via propagation
    
    def test_logging_levels_work_correctly(self):
        """Test that different logging levels work as expected."""
        logger = setup_vocab_logging(self.log_dir, 'level_test')
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message") 
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check that messages were written to file
        log_file = self.log_dir / "vocab.log"
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # File handler should capture all levels
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content
    
    def test_rich_handler_configuration(self):
        """Test that RichHandler is configured correctly."""
        logger = setup_vocab_logging(self.log_dir, 'rich_test')
        
        rich_handler = None
        for handler in logging.root.handlers:
            if handler.__class__.__name__ == 'RichHandler':
                rich_handler = handler
                break
        
        assert rich_handler is not None
        # Rich handler should have console and proper settings
        # Check for actual RichHandler attributes
        assert hasattr(rich_handler, 'console') or hasattr(rich_handler, '_console')
        assert rich_handler.level == logging.INFO
    
    def test_logger_propagation_enabled(self):
        """Test that logger propagation is enabled for proper hierarchy."""
        logger = setup_vocab_logging(self.log_dir, 'propagation_test')
        
        # Logger should propagate to root
        assert logger.propagate is True
        
        # Logger should not have direct handlers (uses root)
        assert len(logger.handlers) == 0
    
    @patch('x_spanformer.vocab.vocab_logging.Console')
    def test_console_creation_in_handlers(self, mock_console):
        """Test that Console objects are created properly for handlers."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        logger = setup_vocab_logging(self.log_dir, 'console_test')
        
        # Console should be created for RichHandler
        assert mock_console.called
    
    def test_file_encoding_utf8(self):
        """Test that log files are created with UTF-8 encoding."""
        logger = setup_vocab_logging(self.log_dir, 'encoding_test')
        
        # Log some unicode content
        logger.info("Test unicode: 中文 测试 🚀 ✅")
        
        log_file = self.log_dir / "vocab.log"
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "中文 测试 🚀 ✅" in content
    
    def test_log_file_overwrite_mode(self):
        """Test that log files are opened in overwrite mode."""
        # Create initial log
        logger1 = setup_vocab_logging(self.log_dir, 'overwrite_test')
        logger1.info("First message")
        
        # Clear handlers and create new setup
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logger2 = setup_vocab_logging(self.log_dir, 'overwrite_test')
        logger2.info("Second message")
        
        log_file = self.log_dir / "vocab.log"
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should only contain second message (file was overwritten)
        assert "Second message" in content
        # First message should be gone
        lines_with_first = [line for line in content.split('\n') if "First message" in line]
        assert len(lines_with_first) == 0
