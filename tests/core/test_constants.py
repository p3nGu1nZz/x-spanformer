"""
Test suite for x_spanformer.agents.constants module.

Tests constants and default values used throughout the system.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.agents import constants


class TestConstants:
    """Test constants module."""
    
    def test_default_system_exists(self):
        """Test that DEFAULT_SYSTEM constant is defined."""
        assert hasattr(constants, 'DEFAULT_SYSTEM')
        assert isinstance(constants.DEFAULT_SYSTEM, str)
        assert len(constants.DEFAULT_SYSTEM) > 0
    
    def test_judge_model_exists(self):
        """Test that JUDGE_MODEL constant is defined."""
        assert hasattr(constants, 'JUDGE_MODEL')
        assert isinstance(constants.JUDGE_MODEL, str)
        assert len(constants.JUDGE_MODEL) > 0
    
    def test_temperature_exists(self):
        """Test that TEMPERATURE constant is defined."""
        assert hasattr(constants, 'TEMPERATURE')
        assert isinstance(constants.TEMPERATURE, (int, float))
        assert 0.0 <= constants.TEMPERATURE <= 2.0  # Reasonable temperature range
    
    def test_default_system_content(self):
        """Test DEFAULT_SYSTEM has reasonable content."""
        assert constants.DEFAULT_SYSTEM == "You are a helpful assistant."
    
    def test_judge_model_format(self):
        """Test JUDGE_MODEL follows expected format."""
        # Should be in format "model:tag" or just "model"
        assert "phi4" in constants.JUDGE_MODEL.lower() or "ollama" in constants.JUDGE_MODEL.lower()
    
    def test_temperature_value(self):
        """Test TEMPERATURE has reasonable value."""
        assert constants.TEMPERATURE == 0.2
    
    def test_constants_are_immutable_types(self):
        """Test that constants use immutable types."""
        # String is immutable
        assert isinstance(constants.DEFAULT_SYSTEM, str)
        assert isinstance(constants.JUDGE_MODEL, str)
        # Numbers are immutable  
        assert isinstance(constants.TEMPERATURE, (int, float))
    
    def test_constants_not_empty(self):
        """Test that string constants are not empty."""
        assert constants.DEFAULT_SYSTEM.strip() != ""
        assert constants.JUDGE_MODEL.strip() != ""


class TestConstantsUsage:
    """Test how constants might be used in practice."""
    
    def test_constants_can_be_imported(self):
        """Test that constants can be imported individually."""
        from x_spanformer.agents.constants import DEFAULT_SYSTEM, JUDGE_MODEL, TEMPERATURE
        
        assert DEFAULT_SYSTEM is not None
        assert JUDGE_MODEL is not None
        assert TEMPERATURE is not None
    
    def test_temperature_suitable_for_llm(self):
        """Test that TEMPERATURE is suitable for LLM usage."""
        # Temperature should be between 0 and 1 for most LLMs
        assert 0.0 <= constants.TEMPERATURE <= 1.0
        # Should not be exactly 0 (too deterministic) or exactly 1 (too random)
        assert 0.0 < constants.TEMPERATURE < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
