"""
Comprehensive test suite for x_spanformer configuration system.

Tests all configuration files in /config directory:
- Agent configs: judge_agent.yaml, span_annotator_agent.yaml  
- Pipeline configs: span_annotator.yaml, vocab2embedding.yaml, etc.
- Config loaders: judge_config_loader, span_annotator_config_loader
- Validation, error handling, and defaults

Designed to work reliably in local development and GitHub Actions.
"""

import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from x_spanformer.config.judge_config_loader import load_judge_config
from x_spanformer.config.span_annotator_config_loader import load_config, SpanAnnotatorConfig


class TestActualConfigFiles:
    """Test that our actual config files exist and are valid."""
    
    @pytest.fixture
    def config_root(self):
        """Get the config directory path."""
        # From tests/core, go up to project root, then to config
        return Path(__file__).parent.parent.parent / "config"
    
    def test_config_directory_exists(self, config_root):
        """Test that config directory exists."""
        assert config_root.exists(), f"Config directory not found: {config_root}"
        assert config_root.is_dir(), "Config path is not a directory"
    
    def test_agent_configs_exist(self, config_root):
        """Test that agent config files exist."""
        agents_dir = config_root / "agents"
        assert agents_dir.exists(), "agents directory missing"
        
        judge_config = agents_dir / "judge_agent.yaml"
        span_config = agents_dir / "span_annotator_agent.yaml"
        
        assert judge_config.exists(), "judge_agent.yaml missing"
        assert span_config.exists(), "span_annotator_agent.yaml missing"
    
    def test_pipeline_configs_exist(self, config_root):
        """Test that pipeline config files exist."""
        pipelines_dir = config_root / "pipelines"
        assert pipelines_dir.exists(), "pipelines directory missing"
        
        expected_files = [
            "span_annotator.yaml",
            "vocab2embedding.yaml", 
            "jsonl2vocab.yaml",
            "repo2jsonl.yaml",
            "embedding2span.yaml"
        ]
        
        for filename in expected_files:
            config_file = pipelines_dir / filename
            assert config_file.exists(), f"Pipeline config missing: {filename}"
    
    def test_judge_agent_yaml_valid(self, config_root):
        """Test that judge_agent.yaml is valid YAML with expected structure."""
        judge_config_path = config_root / "agents" / "judge_agent.yaml"
        
        with open(judge_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Test required top-level keys
        assert "agent_type" in config
        assert "model" in config
        assert "judge" in config
        
        # Test agent_type is correct
        assert config["agent_type"] == "judge"
        
        # Test model section has required fields
        model = config["model"]
        assert "name" in model
        assert isinstance(model["name"], str)
        
        # Test judge section has required fields
        judge = config["judge"]
        assert "judges" in judge
        assert "threshold" in judge
        assert "max_retries" in judge
        assert isinstance(judge["judges"], int)
        assert isinstance(judge["threshold"], (int, float))
        assert isinstance(judge["max_retries"], int)
    
    def test_span_annotator_agent_yaml_valid(self, config_root):
        """Test that span_annotator_agent.yaml is valid YAML with expected structure."""
        span_config_path = config_root / "agents" / "span_annotator_agent.yaml"
        
        with open(span_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Test required top-level keys
        assert "agent_type" in config
        assert config["agent_type"] == "span_annotator"
        
        # Test model section exists
        assert "model" in config
        assert "name" in config["model"]
    
    def test_span_annotator_pipeline_yaml_valid(self, config_root):
        """Test that span_annotator.yaml pipeline config is valid."""
        pipeline_config_path = config_root / "pipelines" / "span_annotator.yaml"
        
        with open(pipeline_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Test required sections
        assert "processing" in config
        assert "output" in config
        
        # Test processing section
        processing = config["processing"]
        assert "max_retries" in processing
        assert "conversation_timeout" in processing
        assert "batch_size" in processing
        
        # Test output section
        output = config["output"]
        assert "save_working_files" in output
        assert "consolidate_on_completion" in output


class TestJudgeConfigLoader:
    """Test judge_config_loader functionality."""
    
    def test_load_judge_config_with_mock(self):
        """Test judge config loading with mocked file system."""
        mock_config = {
            "agent_type": "judge",
            "model": {"name": "phi4-mini", "temperature": 0.2},
            "judge": {"judges": 5, "threshold": 0.69, "max_retries": 3},
            "templates": {"system": "judge_system", "judge": "judge_template"}
        }
        
        yaml_content = yaml.dump(mock_config)
        
        # Mock the entire path construction and file operations
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_file = mock_open(read_data=yaml_content)
        mock_path.open = mock_file
        
        with patch("x_spanformer.config.judge_config_loader.Path") as mock_path_class:
            # Mock the path construction chain
            mock_path_class.return_value.parent.parent.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_path
            with patch("x_spanformer.config.judge_config_loader.c"):  # Suppress console
                result = load_judge_config("test.yaml", quiet=True)
                
                assert result["agent_type"] == "judge"
                assert result["model"]["name"] == "phi4-mini"
                assert result["judge"]["judges"] == 5
                assert result["judge"]["threshold"] == 0.69
    
    def test_load_judge_config_missing_file(self):
        """Test error handling when judge config file is missing."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("x_spanformer.config.judge_config_loader.c"):
                with pytest.raises(FileNotFoundError, match="Missing judge config"):
                    load_judge_config("missing.yaml")
    
    def test_load_judge_config_actual_file(self):
        """Test loading the actual judge_agent.yaml file."""
        # This tests the real config file exists and can be loaded
        config_path = Path(__file__).parent.parent.parent / "config" / "agents" / "judge_agent.yaml"
        
        with patch("x_spanformer.config.judge_config_loader.c"):  # Suppress console
            result = load_judge_config("judge_agent.yaml", quiet=True)
            
            # Test that we get a valid config structure
            assert isinstance(result, dict)
            assert "agent_type" in result
            assert result["agent_type"] == "judge"


class TestSpanAnnotatorConfigLoader:
    """Test span_annotator_config_loader functionality."""
    
    def test_load_config_defaults(self):
        """Test loading config with defaults when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("x_spanformer.config.span_annotator_config_loader.logger"):
                config = load_config()
                
                assert isinstance(config, SpanAnnotatorConfig)
                assert config.processing.max_retries == 3
                assert config.processing.conversation_timeout == 30.0
                assert config.processing.batch_size == 64
                assert config.output.save_working_files is True
                assert config.output.consolidate_on_completion is True
    
    def test_load_config_from_yaml(self):
        """Test loading config from custom YAML content."""
        config_data = {
            "processing": {
                "max_retries": 5,
                "conversation_timeout": 45.0,
                "batch_size": 128
            },
            "output": {
                "save_working_files": False,
                "save_failed_requests": True
            }
        }
        
        yaml_content = yaml.dump(config_data)
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch("x_spanformer.config.span_annotator_config_loader.logger"):
                    config = load_config("test.yaml")
                    
                    assert config.processing.max_retries == 5
                    assert config.processing.conversation_timeout == 45.0  
                    assert config.processing.batch_size == 128
                    assert config.output.save_working_files is False
    
    def test_load_config_actual_file(self):
        """Test loading the actual span_annotator.yaml pipeline config."""
        # This should work with the real config file
        with patch("x_spanformer.config.span_annotator_config_loader.logger"):
            config = load_config("span_annotator.yaml")
            
            assert isinstance(config, SpanAnnotatorConfig)
            # Test that we get valid default or loaded values
            assert config.processing.max_retries > 0
            assert config.processing.batch_size > 0
            assert isinstance(config.output.save_working_files, bool)


class TestConfigStructureValidation:
    """Test configuration structure and validation."""
    
    def test_span_annotator_config_dataclass(self):
        """Test SpanAnnotatorConfig dataclass structure."""
        config = SpanAnnotatorConfig()
        
        # Test that it has the expected attributes
        assert hasattr(config, 'processing')
        assert hasattr(config, 'output')
        
        # Test processing config
        assert hasattr(config.processing, 'max_retries')
        assert hasattr(config.processing, 'conversation_timeout')
        assert hasattr(config.processing, 'batch_size')
        
        # Test output config  
        assert hasattr(config.output, 'save_working_files')
        assert hasattr(config.output, 'save_failed_requests')
        assert hasattr(config.output, 'consolidate_on_completion')
        assert hasattr(config.output, 'include_metadata')
        
        # Test default values
        assert config.processing.max_retries == 3
        assert config.processing.conversation_timeout == 30.0
        assert config.processing.batch_size == 64
        assert config.output.save_working_files is True
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution functionality."""
        from x_spanformer.config.span_annotator_config_loader import substitute_env_vars
        
        # Set up test environment variable
        test_env_key = "X_SPANFORMER_TEST_VAR"
        test_env_value = "test_substitution_value"
        os.environ[test_env_key] = test_env_value
        
        try:
            test_data = {
                "simple": f"${{{test_env_key}}}",
                "alternative": f"${test_env_key}",
                "nested": {
                    "key": f"${{{test_env_key}}}"
                }
            }
            
            result = substitute_env_vars(test_data)
            
            assert result["simple"] == test_env_value
            assert result["alternative"] == test_env_value
            assert result["nested"]["key"] == test_env_value
            
        finally:
            # Clean up environment variable
            if test_env_key in os.environ:
                del os.environ[test_env_key]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_yaml_content(self):
        """Test handling of invalid YAML content."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=invalid_yaml)):
                with patch("x_spanformer.config.span_annotator_config_loader.logger"):
                    # Should fall back to defaults on YAML parse error
                    config = load_config("invalid.yaml")
                    
                    assert isinstance(config, SpanAnnotatorConfig)
                    # Should have default values when YAML parsing fails
                    assert config.processing.max_retries == 3
                    assert config.processing.batch_size == 64
    
    def test_partial_config_merging(self):
        """Test merging partial config with defaults."""
        partial_config = {
            "processing": {
                "max_retries": 10  # Override just this value
            }
        }
        
        yaml_content = yaml.dump(partial_config)
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch("x_spanformer.config.span_annotator_config_loader.logger"):
                    config = load_config("partial.yaml")
                    
                    # Should have overridden value
                    assert config.processing.max_retries == 10
                    # Should have defaults for missing values
                    assert config.processing.conversation_timeout == 30.0
                    assert config.processing.batch_size == 64
                    assert config.output.save_working_files is True
    
    def test_empty_config_file(self):
        """Test handling of empty config file."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="")):
                with patch("x_spanformer.config.span_annotator_config_loader.logger"):
                    config = load_config("empty.yaml")
                    
                    # Should fall back to defaults
                    assert isinstance(config, SpanAnnotatorConfig)
                    assert config.processing.max_retries == 3


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_all_config_files_loadable(self):
        """Test that all actual config files can be loaded without errors."""
        config_root = Path(__file__).parent.parent.parent / "config"
        
        # Test agent configs
        for agent_file in ["judge_agent.yaml", "span_annotator_agent.yaml"]:
            agent_path = config_root / "agents" / agent_file
            assert agent_path.exists(), f"Agent config missing: {agent_file}"
            
            # Load and verify it's valid YAML
            with open(agent_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                assert isinstance(config, dict), f"Invalid YAML in {agent_file}"
                assert "agent_type" in config, f"Missing agent_type in {agent_file}"
        
        # Test pipeline configs
        pipeline_files = ["span_annotator.yaml", "vocab2embedding.yaml", "jsonl2vocab.yaml", 
                         "repo2jsonl.yaml", "embedding2span.yaml"]
        
        for pipeline_file in pipeline_files:
            pipeline_path = config_root / "pipelines" / pipeline_file
            assert pipeline_path.exists(), f"Pipeline config missing: {pipeline_file}"
            
            # Load and verify it's valid YAML
            with open(pipeline_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                assert isinstance(config, dict), f"Invalid YAML in {pipeline_file}"
    
    def test_config_loaders_work_with_actual_files(self):
        """Test that config loaders work with actual config files."""
        # Test judge config loader
        with patch("x_spanformer.config.judge_config_loader.c"):
            judge_config = load_judge_config("judge_agent.yaml", quiet=True)
            assert isinstance(judge_config, dict)
            assert judge_config["agent_type"] == "judge"
        
        # Test span annotator config loader
        with patch("x_spanformer.config.span_annotator_config_loader.logger"):
            span_config = load_config("span_annotator.yaml")
            assert isinstance(span_config, SpanAnnotatorConfig)
            assert span_config.processing.max_retries > 0
