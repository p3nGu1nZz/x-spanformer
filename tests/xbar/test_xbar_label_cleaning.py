#!/usr/bin/env python3
"""Test cases for XBar label cleaning functionality."""

import pytest
from x_spanformer.xbar.xbar_map import XBarLabelMap


class TestXBarLabelCleaning:
    """Test XBar label cleaning and mapping functionality."""
    
    def test_get_label_mapping_suggestions_valid_mappings(self):
        """Test that invalid labels are mapped to valid ones correctly."""
        test_cases = [
            ("proper noun", "noun"),
            ("proper_noun", "noun"),
            ("code_statement", "expression"),
            ("code statement", "expression"),
            ("code_expression", "expression"),
            ("parenthesis", "punctuation"),
            ("colon", "punctuation"),
            ("numeral", "literal"),
            ("prefix", "identifier"),
            ("noun, punctuation", "noun"),  # Multi-label case
        ]
        
        for invalid_label, expected_mapping in test_cases:
            result = XBarLabelMap.get_label_mapping_suggestions(invalid_label)
            assert result == expected_mapping, f"Expected '{invalid_label}' to map to '{expected_mapping}', got '{result}'"
    
    def test_get_label_mapping_suggestions_no_mapping(self):
        """Test that truly invalid labels return None."""
        invalid_cases = [
            "completely_invalid_label",
            "xyz123",
            "",
            None
        ]
        
        for invalid_label in invalid_cases:
            result = XBarLabelMap.get_label_mapping_suggestions(invalid_label)
            assert result is None, f"Expected None for '{invalid_label}', got '{result}'"
    
    def test_clean_and_validate_labels_basic(self):
        """Test basic label cleaning functionality."""
        annotations = [
            {"xbar_label": "noun", "text": "cat"},
            {"xbar_label": "proper noun", "text": "John"},
            {"xbar_label": "code_statement", "text": "x = 1"},
            {"xbar_label": "invalid_label", "text": "something"},
        ]
        
        cleaned, stats = XBarLabelMap.clean_and_validate_labels(annotations)
        
        # Should have 3 annotations (1 valid, 2 mapped, 1 removed)
        assert len(cleaned) == 3
        assert stats["valid"] == 1
        assert stats["mapped"] == 2
        assert stats["removed"] == 1
        
        # Check specific mappings
        labels = [ann["xbar_label"] for ann in cleaned]
        assert "noun" in labels  # Original valid label
        assert "expression" in labels  # Mapped from code_statement
        
        # Check that original labels are preserved
        mapped_annotations = [ann for ann in cleaned if "original_label" in ann]
        assert len(mapped_annotations) == 2
    
    def test_clean_and_validate_labels_multi_label(self):
        """Test handling of multi-label cases."""
        annotations = [
            {"xbar_label": "noun, punctuation", "text": "word"},  # Just the word without punctuation
            {"xbar_label": "adjective, noun", "text": "quick-brown"},  # Valid identifier-style word
        ]
        
        cleaned, stats = XBarLabelMap.clean_and_validate_labels(annotations)
        
        # Both should be mapped since multi-labels are treated as invalid
        assert len(cleaned) == 2
        assert stats["mapped"] == 2  # Both multi-labels get mapped
        assert stats["valid"] == 0   # No multi-labels are considered valid
        assert stats["removed"] == 0
        
        # Check that they map to first valid component
        labels = [ann["xbar_label"] for ann in cleaned]
        assert "noun" in labels  # From "noun, punctuation" 
        assert "adjective" in labels  # From "adjective, noun"
    
    def test_clean_and_validate_labels_preserves_structure(self):
        """Test that cleaning preserves annotation structure."""
        original_annotation = {
            "id": 1,
            "sequence_number": 42,
            "xbar_label": "proper noun",
            "text": "Alice",
            "start_pos": 0,
            "end_pos": 5,
            "domain_type": "natural"
        }
        
        cleaned, stats = XBarLabelMap.clean_and_validate_labels([original_annotation])
        
        assert len(cleaned) == 1
        cleaned_ann = cleaned[0]
        
        # Check that all original fields are preserved
        assert cleaned_ann["id"] == 1
        assert cleaned_ann["sequence_number"] == 42
        assert cleaned_ann["text"] == "Alice"
        assert cleaned_ann["start_pos"] == 0
        assert cleaned_ann["end_pos"] == 5
        assert cleaned_ann["domain_type"] == "natural"
        
        # Check that label was mapped and original preserved
        assert cleaned_ann["xbar_label"] == "noun"
        assert cleaned_ann["original_label"] == "proper noun"
    
    def test_pattern_based_mappings(self):
        """Test pattern-based label mappings."""
        test_cases = [
            ("custom_noun_type", "noun"),
            ("special_verb_form", "verb"),
            ("my_code_statement", "expression"),
            ("bracket_punctuation", "punctuation"),
            ("user_identifier", "identifier"),
            ("number_literal", "literal"),
        ]
        
        for pattern_label, expected in test_cases:
            result = XBarLabelMap.get_label_mapping_suggestions(pattern_label)
            assert result == expected, f"Pattern '{pattern_label}' should map to '{expected}', got '{result}'"


if __name__ == "__main__":
    pytest.main([__file__])
