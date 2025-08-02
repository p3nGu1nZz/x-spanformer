"""
Tests for span annotator configuration loader with logging support.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from x_spanformer.config.span_annotator_config_loader import (
    load_config,
    save_config,
    SpanAnnotatorConfig,
    ProcessingConfig,
    OutputConfig,
    LoggingConfig,
    substitute_env_vars
)


class TestSpanAnnotatorConfigLoader:
    """Test configuration loading and saving functionality."""
    
    def test_load_default_config(self):
        """Test loading with default values when no file exists."""
        config = load_config("nonexistent/config.yaml")
        
        assert config.processing.max_retries == 3
        assert config.processing.conversation_timeout == 30.0
        assert config.processing.batch_size == 64
        
        assert config.output.save_working_files is True
        assert config.output.save_failed_requests is True
        assert config.output.consolidate_on_completion is True
        assert config.output.include_metadata is True
        
        assert config.logging.level == "INFO"
        assert config.logging.log_to_file is True
        assert "%(asctime)s" in config.logging.format
    
    def test_load_partial_config(self):
        """Test loading with partial configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "processing": {
                    "batch_size": 32
                },
                "logging": {
                    "level": "DEBUG"
                }
            }, f)
            temp_path = f.name
            
        config = load_config(temp_path)
        
        # Should use custom values where provided
        assert config.processing.batch_size == 32
        assert config.logging.level == "DEBUG"
        
        # Should use defaults for missing values
        assert config.processing.max_retries == 3
        assert config.output.save_working_files is True
        
        try:
            Path(temp_path).unlink()
        except PermissionError:
            pass  # Windows file cleanup issue
    
    def test_load_complete_config(self):
        """Test loading with complete configuration file."""
        config_data = {
            "processing": {
                "max_retries": 5,
                "conversation_timeout": 60.0,
                "batch_size": 16
            },
            "output": {
                "save_working_files": False,
                "save_failed_requests": False,
                "consolidate_on_completion": False,
                "include_metadata": False
            },
            "logging": {
                "level": "WARNING",
                "format": "%(levelname)s: %(message)s",
                "log_to_file": False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
            
        config = load_config(temp_path)
        
        assert config.processing.max_retries == 5
        assert config.processing.conversation_timeout == 60.0
        assert config.processing.batch_size == 16
        
        assert config.output.save_working_files is False
        assert config.output.save_failed_requests is False
        assert config.output.consolidate_on_completion is False
        assert config.output.include_metadata is False
        
        assert config.logging.level == "WARNING"
        assert config.logging.format == "%(levelname)s: %(message)s"
        assert config.logging.log_to_file is False
        
        try:
            Path(temp_path).unlink()
        except PermissionError:
            pass  # Windows file cleanup issue
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration roundtrip."""
        original_config = SpanAnnotatorConfig(
            processing=ProcessingConfig(
                max_retries=2,
                conversation_timeout=45.0,
                batch_size=8
            ),
            output=OutputConfig(
                save_working_files=False,
                consolidate_on_completion=True
            ),
            logging=LoggingConfig(
                level="ERROR",
                log_to_file=False
            )
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
            
        save_config(original_config, temp_path)
        loaded_config = load_config(temp_path)
        
        assert loaded_config.processing.max_retries == 2
        assert loaded_config.processing.conversation_timeout == 45.0
        assert loaded_config.processing.batch_size == 8
        
        assert loaded_config.output.save_working_files is False
        assert loaded_config.output.consolidate_on_completion is True
        
        assert loaded_config.logging.level == "ERROR"
        assert loaded_config.logging.log_to_file is False
        
        try:
            Path(temp_path).unlink()
        except PermissionError:
            pass  # Windows file cleanup issue
    
    def test_substitute_env_vars_string(self):
        """Test environment variable substitution in strings."""
        import os
        os.environ["TEST_VAR"] = "test_value"
        
        result = substitute_env_vars("Hello ${TEST_VAR} world")
        assert result == "Hello test_value world"
        
        result = substitute_env_vars("Path: $TEST_VAR/subdir")
        assert result == "Path: test_value/subdir"
        
        del os.environ["TEST_VAR"]
    
    def test_substitute_env_vars_dict(self):
        """Test environment variable substitution in dictionaries."""
        import os
        os.environ["TEST_HOST"] = "localhost"
        os.environ["TEST_PORT"] = "8080"
        
        data = {
            "host": "${TEST_HOST}",
            "port": "$TEST_PORT",
            "url": "http://${TEST_HOST}:$TEST_PORT"
        }
        
        result = substitute_env_vars(data)
        
        assert result["host"] == "localhost"
        assert result["port"] == "8080"
        assert result["url"] == "http://localhost:8080"
        
        del os.environ["TEST_HOST"]
        del os.environ["TEST_PORT"]
    
    def test_config_dataclass_fields(self):
        """Test that all configuration dataclasses have expected fields."""
        processing_config = ProcessingConfig()
        assert hasattr(processing_config, 'max_retries')
        assert hasattr(processing_config, 'conversation_timeout')
        assert hasattr(processing_config, 'batch_size')
        
        output_config = OutputConfig()
        assert hasattr(output_config, 'save_working_files')
        assert hasattr(output_config, 'save_failed_requests')
        assert hasattr(output_config, 'consolidate_on_completion')
        assert hasattr(output_config, 'include_metadata')
        
        logging_config = LoggingConfig()
        assert hasattr(logging_config, 'level')
        assert hasattr(logging_config, 'format')
        assert hasattr(logging_config, 'log_to_file')
        
        span_config = SpanAnnotatorConfig()
        assert hasattr(span_config, 'processing')
        assert hasattr(span_config, 'output')
        assert hasattr(span_config, 'logging')
    
    def test_invalid_yaml_fallback(self):
        """Test that invalid YAML falls back to defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:\n  - broken")
            temp_path = f.name
            
        config = load_config(temp_path)
        
        # Should use all defaults
        assert config.processing.max_retries == 3
        assert config.logging.level == "INFO"
        
        try:
            Path(temp_path).unlink()
        except PermissionError:
            pass  # Windows file cleanup issue
