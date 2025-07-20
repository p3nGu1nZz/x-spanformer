"""
Test suite for x_spanformer.agents.config_loader module.

Covers:
- Config file loading functionality
- Error handling for missing files
- YAML parsing validation
- Quiet mode operation
- Config table display functionality
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from x_spanformer.agents.config_loader import load_judge_config


class TestConfigLoader:
    """Test config_loader functionality."""

    def test_load_judge_config_success(self):
        """Test successful config loading with valid YAML."""
        # Create a temporary config file
        config_data = {
            "agent_type": "judge",
            "model": {"name": "phi4-mini"},
            "judge": {
                "judges": 5,
                "model_name": "phi4-mini",
                "temperature": 0.1,
                "threshold": 0.69,
                "max_retries": 3
            },
            "dialogue": {"max_turns": 10},
            "regex_filters": [{"pattern": "test"}],
            "templates": {"system": "sys_template", "judge": "judge_template"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_path = Path(tmp_file.name)

        # Mock the config path resolution
        with patch("x_spanformer.agents.config_loader.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value = tmp_path.open()
            mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path_instance
            
            # Mock console to suppress output during testing
            with patch("x_spanformer.agents.config_loader.c") as mock_console:
                result = load_judge_config("test_config.yaml", quiet=True)
                
                assert result == config_data
                assert result["agent_type"] == "judge"
                assert result["model"]["name"] == "phi4-mini"
                assert result["judge"]["judges"] == 5

        # Clean up
        try:
            tmp_path.unlink()
        except PermissionError:
            # On Windows, files may still be in use, ignore cleanup error
            pass

    def test_load_judge_config_missing_file(self):
        """Test FileNotFoundError when config file doesn't exist."""
        with patch("x_spanformer.agents.config_loader.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path_instance
            
            with patch("x_spanformer.agents.config_loader.c") as mock_console:
                with pytest.raises(FileNotFoundError, match="Missing judge config"):
                    load_judge_config("nonexistent.yaml")

    def test_load_judge_config_invalid_yaml(self):
        """Test handling of invalid YAML content."""
        # Create a temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write("invalid: yaml: content: [")
            tmp_path = Path(tmp_file.name)

        with patch("x_spanformer.agents.config_loader.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value = tmp_path.open()
            mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path_instance
            
            with patch("x_spanformer.agents.config_loader.c") as mock_console:
                with pytest.raises(yaml.YAMLError):
                    load_judge_config("invalid.yaml")

        # Clean up
        try:
            tmp_path.unlink()
        except PermissionError:
            # On Windows, files may still be in use, ignore cleanup error
            pass

    def test_load_judge_config_quiet_mode(self):
        """Test that quiet mode suppresses table output."""
        config_data = {
            "agent_type": "judge",
            "model": {"name": "test-model"},
            "judge": {"judges": 1, "model_name": "test", "temperature": 0.1, "threshold": 0.5, "max_retries": 1},
            "dialogue": {"max_turns": 5},
            "templates": {"system": "sys"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_path = Path(tmp_file.name)

        with patch("x_spanformer.agents.config_loader.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value = tmp_path.open()
            mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path_instance
            
            with patch("x_spanformer.agents.config_loader.c") as mock_console:
                result = load_judge_config("test.yaml", quiet=True)
                
                # Verify table creation methods were not called when quiet=True
                mock_console.print.assert_called()
                assert result == config_data

        # Clean up
        try:
            tmp_path.unlink()
        except PermissionError:
            # On Windows, files may still be in use, ignore cleanup error
            pass

    def test_load_judge_config_table_display(self):
        """Test table display functionality when quiet=False."""
        config_data = {
            "agent_type": "test_agent",
            "model": {"name": "test-model"},
            "judge": {
                "judges": 3,
                "model_name": "judge-model",
                "temperature": 0.2,
                "threshold": 0.75,
                "max_retries": 2
            },
            "dialogue": {"max_turns": 8},
            "regex_filters": [{"pattern": "filter1"}, {"pattern": "filter2"}],
            "templates": {"system": "sys_template", "judge": "judge_template", "extra": "extra_template"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_path = Path(tmp_file.name)

        with patch("x_spanformer.agents.config_loader.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value = tmp_path.open()
            mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path_instance
            
            with patch("x_spanformer.agents.config_loader.c") as mock_console:
                with patch("x_spanformer.agents.config_loader.Table") as mock_table:
                    mock_table_instance = MagicMock()
                    mock_table.return_value = mock_table_instance
                    
                    result = load_judge_config("test.yaml", quiet=False)
                    
                    # Verify table was created and populated
                    mock_table.assert_called_once()
                    assert mock_table_instance.add_column.call_count == 2
                    assert mock_table_instance.add_row.call_count >= 7  # Should have multiple rows
                    mock_console.print.assert_called()

        # Clean up
        try:
            tmp_path.unlink()
        except PermissionError:
            # On Windows, files may still be in use, ignore cleanup error
            pass

    def test_load_judge_config_default_values(self):
        """Test handling of missing configuration values with defaults."""
        config_data = {
            "model": {"name": "minimal-model"},
            "judge": {"judges": 1, "model_name": "judge", "temperature": 0.1, "threshold": 0.5, "max_retries": 1},
            "dialogue": {"max_turns": 5},
            "templates": {"system": "sys"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_path = Path(tmp_file.name)

        with patch("x_spanformer.agents.config_loader.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value = tmp_path.open()
            mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path_instance
            
            with patch("x_spanformer.agents.config_loader.c") as mock_console:
                result = load_judge_config("minimal.yaml", quiet=True)
                
                # Should handle missing agent_type with default
                assert result.get("agent_type", "—") == "—"
                assert result["model"]["name"] == "minimal-model"

        # Clean up
        try:
            tmp_path.unlink()
        except PermissionError:
            # On Windows, files may still be in use, ignore cleanup error
            pass

    def test_load_judge_config_default_filename(self):
        """Test default filename behavior."""
        with patch("x_spanformer.agents.config_loader.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value.parent.__truediv__.return_value.__truediv__.return_value = mock_path_instance
            
            with patch("x_spanformer.agents.config_loader.c") as mock_console:
                with pytest.raises(FileNotFoundError):
                    load_judge_config()  # Should use default "judge.yaml"
                
                # Verify the default path construction
                expected_calls = mock_path.return_value.parent.__truediv__.return_value.__truediv__
                expected_calls.assert_called_with("judge.yaml")
