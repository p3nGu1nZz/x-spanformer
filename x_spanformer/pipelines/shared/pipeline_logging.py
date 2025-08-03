"""
Shared logging utilities for X-Spanformer pipelines.

Provides centralized logging configuration, formatting, and utilities
for consistent logging across all pipeline types.
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class PipelineLogger:
    """
    Centralized logging configuration for X-Spanformer pipelines.
    """
    
    @staticmethod
    def setup_pipeline_logging(
        pipeline_name: str,
        log_level: str = "INFO",
        log_format: Optional[str] = None,
        log_to_file: bool = False,
        log_file_path: Optional[Path] = None,
        console_output: bool = True
    ) -> logging.Logger:
        """
        Set up logging for a pipeline with consistent formatting.
        
        Args:
            pipeline_name: Name of the pipeline for logger identification
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Custom log format string
            log_to_file: Whether to write logs to file
            log_file_path: Path to log file (if log_to_file is True)
            console_output: Whether to output to console
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(pipeline_name)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Set log level
        log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(log_level_obj)
        
        # Default format with timestamp, level, module and message
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        formatter = logging.Formatter(log_format)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level_obj)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            if log_file_path is None:
                # Default log file path
                log_file_path = Path(f"logs/{pipeline_name.lower().replace(' ', '_')}.log")
            
            # Ensure log directory exists
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(log_level_obj)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file_path}")
        
        return logger
    
    @staticmethod
    def setup_from_config(config, pipeline_name: str) -> logging.Logger:
        """
        Set up logging from a configuration object.
        
        Args:
            config: Configuration object with logging settings
            pipeline_name: Name of the pipeline
            
        Returns:
            Configured logger instance
        """
        # Extract logging configuration
        log_config = getattr(config, 'logging', None)
        if not log_config:
            # Fallback to default logging
            return PipelineLogger.setup_pipeline_logging(pipeline_name)
        
        log_level = getattr(log_config, 'level', 'INFO')
        log_format = getattr(log_config, 'format', None)
        log_to_file = getattr(log_config, 'log_to_file', False)
        
        # Determine log file path
        log_file_path = None
        if log_to_file:
            log_file_path = getattr(log_config, 'file_path', None)
            if log_file_path:
                log_file_path = Path(log_file_path)
        
        return PipelineLogger.setup_pipeline_logging(
            pipeline_name=pipeline_name,
            log_level=log_level,
            log_format=log_format,
            log_to_file=log_to_file,
            log_file_path=log_file_path
        )
    
    @staticmethod
    def log_pipeline_start(logger: logging.Logger, pipeline_name: str, **kwargs):
        """
        Log pipeline startup information.
        
        Args:
            logger: Logger instance
            pipeline_name: Name of the pipeline
            **kwargs: Additional parameters to log
        """
        logger.info("=" * 80)
        logger.info(f"Starting {pipeline_name}")
        logger.info("=" * 80)
        
        for key, value in kwargs.items():
            logger.info(f"{key}: {value}")
        
        logger.info(f"Started at: {datetime.now().isoformat()}")
        logger.info("-" * 40)
    
    @staticmethod
    def log_pipeline_completion(logger: logging.Logger, pipeline_name: str, stats: dict):
        """
        Log pipeline completion information.
        
        Args:
            logger: Logger instance
            pipeline_name: Name of the pipeline
            stats: Dictionary of completion statistics
        """
        logger.info("-" * 40)
        logger.info(f"{pipeline_name} Completed")
        logger.info(f"Completed at: {datetime.now().isoformat()}")
        
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=" * 80)
    
    @staticmethod
    def log_batch_progress(
        logger: logging.Logger, 
        batch_num: int, 
        total_batches: int, 
        batch_size: int,
        successful: int = 0,
        failed: int = 0
    ):
        """
        Log batch processing progress.
        
        Args:
            logger: Logger instance
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches
            batch_size: Size of current batch
            successful: Number of successful items in batch
            failed: Number of failed items in batch
        """
        progress_pct = (batch_num / total_batches) * 100
        logger.info(f"Batch {batch_num}/{total_batches} ({progress_pct:.1f}%): "
                   f"size={batch_size}, successful={successful}, failed={failed}")
    
    @staticmethod
    def log_error_with_context(logger: logging.Logger, error: Exception, context: str = ""):
        """
        Log an error with additional context information.
        
        Args:
            logger: Logger instance
            error: Exception that occurred
            context: Additional context about when/where the error occurred
        """
        error_msg = f"Error occurred"
        if context:
            error_msg += f" {context}"
        error_msg += f": {str(error)}"
        
        logger.error(error_msg)
        logger.debug(f"Error details: {type(error).__name__}: {error}", exc_info=True)


class SpanAnnotationLogger:
    """
    Specialized logging utilities for span annotation pipelines.
    """
    
    @staticmethod
    def log_resume_detection(logger: logging.Logger, working_files_count: int, max_processed_id: int):
        """Log resume mode detection information."""
        logger.info(f"Resume mode detected - found {working_files_count} existing working files")
        logger.info(f"Highest processed sequence ID: {max_processed_id}")
    
    @staticmethod
    def log_sequence_categorization(
        logger: logging.Logger,
        failed_retry_count: int,
        gap_count: int, 
        new_count: int,
        completed_count: int
    ):
        """Log how sequences were categorized for processing."""
        total_to_process = failed_retry_count + gap_count + new_count
        logger.info(f"Processing {total_to_process} sequences:")
        logger.info(f"  - Retrying {failed_retry_count} failed sequences")
        logger.info(f"  - Processing {gap_count} gap sequences (within processed range)")
        logger.info(f"  - Processing {new_count} new sequences (beyond processed range)")
        logger.info(f"  - Skipped {completed_count} completed sequences")
    
    @staticmethod
    def log_sequence_completion(logger: logging.Logger, sequence_id: int, span_count: int):
        """Log successful sequence completion."""
        logger.info(f"[COMPLETED] Sequence {sequence_id} - extracted {span_count} spans")
    
    @staticmethod
    def log_sequence_failure(logger: logging.Logger, sequence_id: int, reason: str, consecutive_failures: int):
        """Log sequence failure with context."""
        logger.warning(f"Sequence {sequence_id} failed: {reason} (consecutive failures: {consecutive_failures})")
    
    @staticmethod
    def log_metadata_correction(logger: logging.Logger, corrected_stats: dict):
        """Log metadata correction information."""
        logger.info("Metadata corrected from working files:")
        logger.info(f"  - Processed sequences: {corrected_stats.get('total_files', 0)}")
        logger.info(f"  - Successful annotations: {corrected_stats.get('successful_count', 0)}")
        logger.info(f"  - Failed annotations: {corrected_stats.get('failed_count', 0)}")
        logger.info(f"  - Total spans: {corrected_stats.get('total_spans', 0)}")
        success_rate = corrected_stats.get('successful_count', 0) / max(corrected_stats.get('total_files', 1), 1)
        logger.info(f"  - Success rate: {success_rate:.1%}")


# Convenience function for quick pipeline logging setup
def setup_logging(config, pipeline_name: str) -> logging.Logger:
    """
    Quick setup function for pipeline logging.
    
    Args:
        config: Configuration object
        pipeline_name: Name of the pipeline
        
    Returns:
        Configured logger instance
    """
    return PipelineLogger.setup_from_config(config, pipeline_name)
