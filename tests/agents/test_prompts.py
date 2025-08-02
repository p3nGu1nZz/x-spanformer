"""
Test suite for x_spanformer.agents.prompts module.

Tests Jinja2 template rendering for span annotator system prompts and 
annotation requests with domain-specific classifier integration.
"""

import pytest
from pathlib import Path

from x_spanformer.agents.prompts import (
    render_prompt,
    get_system_prompt,
    render_span_annotator_system_prompt,
    render_span_annotation_request
)
from x_spanformer.xbar.xbar_map import DomainType


class TestTemplateRendering:
    """Test basic template rendering functionality."""
    
    def test_render_prompt_basic(self):
        """Test basic template rendering."""
        # This should use the fallback string template if file not found
        result = render_prompt("Hello {{ name }}", name="World")
        assert "Hello World" in result
    
    def test_render_prompt_with_file_template(self):
        """Test rendering with actual template file."""
        # This should load an actual template file
        result = render_prompt("judge_system", domain="test")
        assert isinstance(result, str)
        assert len(result) > 0
    
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


class TestSpanAnnotatorSystemPrompt:
    """Test span annotator system prompt rendering."""
    
    def test_render_natural_domain_prompt(self):
        """Test rendering system prompt for natural language domain."""
        result = render_span_annotator_system_prompt(domain_type="natural")
        
        assert isinstance(result, str)
        assert len(result) > 100  # Should be substantial
        assert "natural" in result.lower()
        assert "noun" in result.lower()  # Should contain natural language classifiers
        assert "verb" in result.lower()
        assert "X-bar" in result or "xbar" in result.lower()
    
    def test_render_code_domain_prompt(self):
        """Test rendering system prompt for code domain."""
        result = render_span_annotator_system_prompt(domain_type="code")
        
        assert isinstance(result, str)
        assert len(result) > 100  # Should be substantial
        assert "code" in result.lower()
        assert "keyword" in result.lower()  # Should contain code classifiers
        assert "identifier" in result.lower()
        assert "X-bar" in result or "xbar" in result.lower()
    
    def test_render_mixed_domain_prompt(self):
        """Test rendering system prompt for mixed domain."""
        result = render_span_annotator_system_prompt(domain_type="mixed")
        
        assert isinstance(result, str)
        assert len(result) > 100  # Should be substantial
        assert "mixed" in result.lower()
        # Should contain both natural and code elements
        assert "noun" in result.lower()
        assert "keyword" in result.lower()
        assert "inline_code" in result.lower()  # Mixed-specific classifier
    
    def test_render_invalid_domain_fallback(self):
        """Test rendering with invalid domain falls back to natural."""
        result = render_span_annotator_system_prompt(domain_type="invalid")
        
        assert isinstance(result, str)
        assert len(result) > 100
        # Should fall back to natural domain
        assert "noun" in result.lower()
        assert "verb" in result.lower()
    
    def test_render_with_additional_kwargs(self):
        """Test rendering with additional template variables."""
        result = render_span_annotator_system_prompt(
            domain_type="natural",
            custom_var="test_value"
        )
        
        assert isinstance(result, str)
        assert len(result) > 100
        # Should still contain natural domain content
        assert "noun" in result.lower()


class TestSpanAnnotationRequest:
    """Test span annotation request rendering."""
    
    def test_render_basic_request(self):
        """Test rendering basic annotation request."""
        text = "The quick brown fox jumps over the lazy dog."
        result = render_span_annotation_request(text=text)
        
        assert isinstance(result, str)
        assert text in result
        assert "analyze" in result.lower()
        assert "span" in result.lower()
        assert "json" in result.lower()
    
    def test_render_natural_domain_request(self):
        """Test rendering request for natural language domain."""
        text = "The cat sat on the mat."
        result = render_span_annotation_request(
            text=text,
            domain_type="natural"
        )
        
        assert isinstance(result, str)
        assert text in result
        assert "natural" in result.lower()
        assert "word" in result.lower()
        assert "phrase" in result.lower()
        assert "clause" in result.lower()
        assert "sentence" in result.lower()
    
    def test_render_code_domain_request(self):
        """Test rendering request for code domain."""
        text = "def function_name(param):"
        result = render_span_annotation_request(
            text=text,
            domain_type="code"
        )
        
        assert isinstance(result, str)
        assert text in result
        assert "code" in result.lower()
        assert "statement" in result.lower()
        # Should contain code-specific classifiers
        assert "keyword" in result.lower()
        assert "identifier" in result.lower()
    
    def test_render_mixed_domain_request(self):
        """Test rendering request for mixed domain."""
        text = "Use the `print()` function to display output."
        result = render_span_annotation_request(
            text=text,
            domain_type="mixed"
        )
        
        assert isinstance(result, str)
        assert text in result
        assert "mixed" in result.lower()
        # Should reference mixed content types
        assert "inline_code" in result.lower() or "mixed content" in result.lower()
    
    def test_render_with_turn_information(self):
        """Test rendering with turn number and focus area."""
        text = "Hello world"
        result = render_span_annotation_request(
            text=text,
            turn_number=2,
            max_turns=5,
            focus_area="detailed phrase analysis"
        )
        
        assert isinstance(result, str)
        assert text in result
        assert "2" in result  # Should include turn number
        assert "5" in result  # Should include max turns
        assert "detailed phrase analysis" in result
    
    def test_render_invalid_domain_fallback(self):
        """Test rendering with invalid domain falls back gracefully."""
        text = "Test text"
        result = render_span_annotation_request(
            text=text,
            domain_type="invalid_domain"
        )
        
        assert isinstance(result, str)
        assert text in result
        # Should fall back to natural domain behavior
        assert "word" in result.lower()
        assert "phrase" in result.lower()
    
    def test_render_with_additional_kwargs(self):
        """Test rendering with additional template variables."""
        text = "Test text"
        result = render_span_annotation_request(
            text=text,
            custom_instruction="Focus on specific patterns",
            analysis_depth="comprehensive"
        )
        
        assert isinstance(result, str)
        assert text in result


class TestTemplateIntegration:
    """Test integration between different template functions."""
    
    def test_system_and_request_consistency(self):
        """Test that system prompt and request are consistent."""
        domain_type = "natural"
        
        system_prompt = render_span_annotator_system_prompt(domain_type=domain_type)
        request = render_span_annotation_request(
            text="Test sentence.",
            domain_type=domain_type
        )
        
        # Both should reference the same domain
        assert domain_type in system_prompt.lower()
        assert domain_type in request.lower()
        
        # Both should mention similar classifier concepts
        assert "noun" in system_prompt.lower()
        assert "noun" in request.lower()
    
    def test_all_domains_work(self):
        """Test that all domain types work for both functions."""
        domains = ["natural", "code", "mixed"]
        text = "Sample text for analysis."
        
        for domain in domains:
            system_prompt = render_span_annotator_system_prompt(domain_type=domain)
            request = render_span_annotation_request(
                text=text,
                domain_type=domain
            )
            
            assert isinstance(system_prompt, str)
            assert isinstance(request, str)
            assert len(system_prompt) > 50
            assert len(request) > 50
            assert text in request
    
    def test_template_variables_propagation(self):
        """Test that template variables are properly propagated."""
        result = render_span_annotator_system_prompt(
            domain_type="natural",
            special_instruction="Use careful analysis"
        )
        
        assert isinstance(result, str)
        # Should still have core natural language content
        assert "noun" in result.lower()


class TestTemplateErrorHandling:
    """Test error handling in template rendering."""
    
    def test_missing_template_fallback(self):
        """Test fallback when template file is missing."""
        # This should use the fallback string template mechanism
        result = render_prompt("nonexistent_template", test_var="value")
        
        # Should not raise an exception, should return processed string
        assert isinstance(result, str)
    
    def test_empty_text_handling(self):
        """Test handling of empty text in annotation request."""
        result = render_span_annotation_request(text="")
        
        assert isinstance(result, str)
        assert len(result) > 0  # Should still have template content
    
    def test_none_values_handling(self):
        """Test handling of None values in template variables."""
        result = render_span_annotation_request(
            text="Test",
            focus_area=None
        )
        
        assert isinstance(result, str)
        assert "Test" in result


class TestTemplateContentValidation:
    """Test validation of template content."""
    
    def test_system_prompt_contains_required_elements(self):
        """Test that system prompt contains all required elements."""
        result = render_span_annotator_system_prompt(domain_type="natural")
        
        # Should contain key instruction elements
        required_elements = [
            "X-bar",
            "span",
            "json",
            "confidence",
            "character",
            "position"
        ]
        
        result_lower = result.lower()
        for element in required_elements:
            assert element.lower() in result_lower, f"Missing required element: {element}"
    
    def test_request_contains_required_elements(self):
        """Test that annotation request contains all required elements."""
        text = "Analyze this text."
        result = render_span_annotation_request(text=text)
        
        # Should contain key elements (case insensitive check for some)
        result_lower = result.lower()
        required_elements = [
            (text, result),  # Exact match
            ("analyze", result_lower),
            ("span", result_lower), 
            ("json", result_lower),
            ("start_char", result),  # Exact match
            ("end_char", result),    # Exact match  
            ("confidence", result_lower)
        ]
        
        for element, search_text in required_elements:
            assert element in search_text, f"Missing required element: {element}"
    
    def test_classifier_names_in_output(self):
        """Test that appropriate classifier names appear in output."""
        # Natural domain should have linguistic classifiers
        natural_result = render_span_annotator_system_prompt(domain_type="natural")
        assert "noun" in natural_result.lower()
        assert "verb" in natural_result.lower()
        
        # Code domain should have programming classifiers
        code_result = render_span_annotator_system_prompt(domain_type="code")
        assert "keyword" in code_result.lower()
        assert "identifier" in code_result.lower()
        
        # Mixed domain should have both
        mixed_result = render_span_annotator_system_prompt(domain_type="mixed")
        assert "noun" in mixed_result.lower()
        assert "keyword" in mixed_result.lower()
        assert "inline_code" in mixed_result.lower()
