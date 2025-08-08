"""
Test suite for x_spanformer.agents.prompts module.

Tests basic Jinja2 template rendering functionality used by the judge agent.
The span annotation functionality has been moved to XBarAnnotator.
"""

import pytest
from pathlib import Path

from x_spanformer.agents.prompts import (
    render_prompt,
    get_system_prompt
)


class TestTemplateRendering:
    """Test basic template rendering functionality."""
    
    def test_render_prompt_basic(self):
        """Test basic template rendering."""
        # This should use the fallback string template if file not found
        result = render_prompt("Hello {{ name }}", name="World")
        assert "Hello World" in result
    
    def test_render_prompt_with_variables(self):
        """Test template rendering with multiple variables."""
        template = "{{ greeting }} {{ name }}! Today is {{ day }}."
        result = render_prompt(template, greeting="Hi", name="Alice", day="Monday")
        assert "Hi Alice! Today is Monday." in result
    
    def test_render_prompt_empty_template(self):
        """Test rendering empty template."""
        result = render_prompt("", name="test")
        assert result == ""
    
    def test_get_system_prompt_default(self):
        """Test getting default system prompt."""
        result = get_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_get_system_prompt_custom(self):
        """Test getting custom system prompt."""
        result = get_system_prompt("judge_system", context="test")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_get_system_prompt_with_kwargs(self):
        """Test system prompt with additional kwargs."""
        result = get_system_prompt("test_template", 
                                   variable1="value1", 
                                   variable2="value2")
        assert isinstance(result, str)


class TestTemplateFileHandling:
    """Test template file handling behavior."""
    
    def test_template_not_found_fallback(self):
        """Test that missing template files fall back to string templates."""
        template_name = "nonexistent_template_{{ variable }}"
        result = render_prompt(template_name, variable="test")
        assert "nonexistent_template_test" in result
    
    def test_template_with_jinja_syntax(self):
        """Test template with various Jinja2 syntax."""
        template = "{% if condition %}Yes{% else %}No{% endif %}: {{ value }}"
        result = render_prompt(template, condition=True, value="success")
        assert "Yes: success" in result
        
        result = render_prompt(template, condition=False, value="failure")
        assert "No: failure" in result
    
    def test_template_with_loops(self):
        """Test template with loop syntax."""
        template = "Items: {% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}"
        result = render_prompt(template, items=["apple", "banana", "cherry"])
        assert "Items: apple, banana, cherry" in result


class TestErrorHandling:
    """Test error handling in template rendering."""
    
    def test_missing_variable_in_template(self):
        """Test behavior when template variable is missing."""
        template = "Hello {{ missing_variable }}"
        # Should not raise error, might render empty or default value
        result = render_prompt(template)
        assert isinstance(result, str)
    
    def test_invalid_jinja_syntax(self):
        """Test behavior with invalid Jinja syntax."""
        template = "Hello {{ unclosed_variable"
        # Should handle gracefully
        try:
            result = render_prompt(template)
            assert isinstance(result, str)
        except Exception:
            # Acceptable to raise exception for invalid syntax
            pass
