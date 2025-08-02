"""
Configuration loader for X-Spanformer pipelines.

Provides utilities for loading and validating YAML configuration files
with support for default values and environment variable substitution.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    max_retries: int = 3
    conversation_timeout: float = 30.0
    batch_size: int = 64


@dataclass
class OutputConfig:
    """Output configuration."""
    save_working_files: bool = True
    save_failed_requests: bool = True
    consolidate_on_completion: bool = True
    include_metadata: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True


@dataclass
class SpanAnnotatorConfig:
    """Complete span annotator pipeline configuration."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def substitute_env_vars(value: Any) -> Any:
    """Substitute environment variables in configuration values."""
    if isinstance(value, str):
        # Replace ${VAR} with environment variable value
        for env_var in os.environ:
            value = value.replace(f"${{{env_var}}}", os.environ[env_var])
            value = value.replace(f"${env_var}", os.environ[env_var])
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    
    return value


def load_config(config_path: Optional[str] = None) -> SpanAnnotatorConfig:
    """
    Load configuration from YAML file with defaults.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
        
    Returns:
        SpanAnnotatorConfig object with loaded configuration
    """
    # Default configuration path
    if config_path is None:
        config_path = "config/pipelines/span_annotator.yaml"
    
    config_file = Path(config_path)
    
    # Start with default configuration
    config_dict = {}
    
    # Load from file if it exists
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                # Substitute environment variables
                config_dict = substitute_env_vars(file_config)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Config file {config_file} not found, using defaults")
    
    # Create configuration objects with proper nesting
    processing_config = ProcessingConfig(**config_dict.get("processing", {}))
    output_config = OutputConfig(**config_dict.get("output", {}))
    logging_config = LoggingConfig(**config_dict.get("logging", {}))
    
    return SpanAnnotatorConfig(
        processing=processing_config,
        output=output_config,
        logging=logging_config
    )


def save_config(config: SpanAnnotatorConfig, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: SpanAnnotatorConfig object to save
        config_path: Path to save configuration file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    config_dict = {
        "processing": {
            "max_retries": config.processing.max_retries,
            "conversation_timeout": config.processing.conversation_timeout,
            "batch_size": config.processing.batch_size
        },
        "output": {
            "save_working_files": config.output.save_working_files,
            "save_failed_requests": config.output.save_failed_requests,
            "consolidate_on_completion": config.output.consolidate_on_completion,
            "include_metadata": config.output.include_metadata
        },
        "logging": {
            "level": config.logging.level,
            "format": config.logging.format,
            "log_to_file": config.logging.log_to_file
        }
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved configuration to {config_file}")
