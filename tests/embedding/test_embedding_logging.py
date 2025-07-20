#!/usr/bin/env python3
"""
tests/embedding/test_embedding_logging_simple.py

Simplified tests for embedding_logging.py module that avoid Windows file locking issues.
"""
import pytest
import logging
from pathlib import Path
import sys

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.embedding.embedding_logging import (
    setup_embedding_logging,
    get_embedding_logger
)


class TestEmbeddingLoggingSimple:
    """Simplified test suite for embedding logging functionality."""
    
    def test_get_embedding_logger_nonexistent(self):
        """Test getting a logger that doesn't exist yet."""
        # This should create a basic logger for testing scenarios
        logger = get_embedding_logger('test_nonexistent_simple')
        
        assert logger is not None
        assert logger.name == 'test_nonexistent_simple'
        # Should have at least a basic handler
        assert len(logger.handlers) > 0 or (logger.parent and len(logger.parent.handlers) > 0)
    
    def test_logger_creation(self):
        """Test basic logger creation without file operations."""
        # Just test the logger object creation
        logger = get_embedding_logger('test_creation')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_creation'
    
    def test_setup_logging_creates_directory(self):
        """Test that setup creates directory structure."""
        # Use current directory for testing
        test_dir = Path('.') / 'temp_test_logs'
        try:
            # Clean up first if exists
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir)
            
            # Setup logging
            logger = setup_embedding_logging(test_dir, 'test_setup')
            
            # Verify directory was created
            assert test_dir.exists()
            assert (test_dir / "embedding.log").exists()
            
            # Verify logger
            assert logger.name == 'test_setup'
            assert logger.level == logging.DEBUG
            
        finally:
            # Clean up handlers
            for handler in logging.root.handlers[:]:
                try:
                    handler.close()
                except:
                    pass
                try:
                    logging.root.removeHandler(handler)
                except:
                    pass
            
            # Clean up directory
            if test_dir.exists():
                import shutil
                import time
                time.sleep(0.1)  # Brief pause for Windows
                try:
                    shutil.rmtree(test_dir)
                except:
                    pass  # Ignore cleanup errors
    
    def test_logging_produces_output(self):
        """Test that logging actually produces output."""
        # Use current directory
        test_dir = Path('.') / 'temp_test_output'
        try:
            # Clean up first
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir)
            
            # Setup and test logging
            logger = setup_embedding_logging(test_dir, 'test_output')
            logger.info("Test message for output verification")
            
            # Read log file
            log_file = test_dir / "embedding.log"
            assert log_file.exists()
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Test message for output verification" in content
                assert "X-SPANFORMER EMBEDDING GENERATION PIPELINE" in content
                
        finally:
            # Clean up
            for handler in logging.root.handlers[:]:
                try:
                    handler.close()
                except:
                    pass
                try:
                    logging.root.removeHandler(handler)
                except:
                    pass
            
            if test_dir.exists():
                import shutil
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(test_dir)
                except:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
