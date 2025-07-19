#!/usr/bin/env python3
"""
vocab_logging.py

Centralized logging configuration for vocabulary induction pipeline.
Provides consistent logging setup with Rich formatting and file output.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_vocab_logging(out_dir: Path, logger_name: str = 'jsonl2vocab') -> logging.Logger:
    """
    Setup comprehensive logging for the vocabulary induction pipeline.
    
    Args:
        out_dir: Output directory where log files will be saved
        logger_name: Name for the logger (default: 'jsonl2vocab')
        
    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create console for Rich output
    console = Console()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for comprehensive logging - catches everything
    log_file = out_dir / "vocab.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Rich console handler for clean terminal output
    console_handler = RichHandler(
        console=console, 
        show_path=False, 
        rich_tracebacks=True,
        markup=True
    )
    console_handler.setLevel(logging.INFO)
    # RichHandler handles its own formatting, so we don't set a formatter
    
    # Configure root logger to catch all messages
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Setup specific logger for this pipeline
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Don't add handlers directly to avoid duplication - let root handle it
    logger.propagate = True
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("X-SPANFORMER VOCABULARY INDUCTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("Following Algorithm 1 from Section 3.1 of the X-Spanformer paper")
    logger.info("-" * 80)
    
    return logger


def get_vocab_logger(logger_name: str = 'jsonl2vocab') -> logging.Logger:
    """
    Get the vocabulary pipeline logger, creating a basic one if none exists.
    
    Args:
        logger_name: Name of the logger to retrieve
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(logger_name)
    # Check if logger has handlers or if parent exists and has handlers
    has_handlers = bool(logger.handlers) or (logger.parent is not None and bool(logger.parent.handlers))
    
    if not has_handlers:
        # Create a basic handler for testing scenarios
        console = Console()
        handler = RichHandler(console=console, show_path=False)
        handler.setLevel(logging.WARNING)  # Only show warnings/errors in tests
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger
