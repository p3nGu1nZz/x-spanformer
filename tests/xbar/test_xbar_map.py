"""
Test suite for x_spanformer.xbar package.

Tests XBar label mappings and domain-specific label definitions
used in span annotation pipeline.
"""

import pytest
from typing import Dict, List

from x_spanformer.xbar.xbar_map import XBarLabelMap, DomainType


class TestDomainType:
    """Test DomainType enum."""
    
    def test_domain_type_values(self):
        """Test that DomainType has expected values."""
        assert DomainType.NATURAL.value == "natural"
        assert DomainType.CODE.value == "code"
        assert DomainType.MIXED.value == "mixed"
    
    def test_domain_type_members(self):
        """Test that DomainType has exactly the expected members."""
        expected_domains = {"NATURAL", "CODE", "MIXED"}
        actual_domains = {member.name for member in DomainType}
        assert actual_domains == expected_domains


class TestXBarLabelMap:
    """Test XBarLabelMap functionality."""
    
    def test_initialization(self):
        """Test that XBarLabelMap has the expected label dictionaries."""
        assert hasattr(XBarLabelMap, 'NATURAL_LABELS')
        assert hasattr(XBarLabelMap, 'CODE_LABELS')
        assert hasattr(XBarLabelMap, 'MIXED_LABELS')
        assert isinstance(XBarLabelMap.NATURAL_LABELS, dict)
        assert isinstance(XBarLabelMap.CODE_LABELS, dict)
        assert isinstance(XBarLabelMap.MIXED_LABELS, dict)
    
    def test_get_labels_for_natural_domain(self):
        """Test getting labels for natural language domain."""
        natural_labels = XBarLabelMap.get_labels_for_domain(DomainType.NATURAL)
        
        # Should contain only natural language labels
        assert isinstance(natural_labels, dict)
        assert len(natural_labels) > 0
        
        # Check for expected natural language labels
        expected_labels = ["noun", "verb", "adjective", "noun_phrase", "verb_phrase", "main_clause"]
        for label in expected_labels:
            assert label in natural_labels
            assert isinstance(natural_labels[label], str)
            assert len(natural_labels[label]) > 0
    
    def test_get_labels_for_code_domain(self):
        """Test getting labels for code domain."""
        code_labels = XBarLabelMap.get_labels_for_domain(DomainType.CODE)
        
        # Should contain only code labels
        assert isinstance(code_labels, dict)
        assert len(code_labels) > 0
        
        # Check for expected code labels
        expected_labels = ["keyword", "identifier", "operator", "function_call", "class_definition"]
        for label in expected_labels:
            assert label in code_labels
            assert isinstance(code_labels[label], str)
            assert len(code_labels[label]) > 0
    
    def test_get_labels_for_mixed_domain(self):
        """Test getting labels for mixed domain."""
        mixed_labels = XBarLabelMap.get_labels_for_domain(DomainType.MIXED)
        
        # Should contain natural + code + mixed labels
        assert isinstance(mixed_labels, dict)
        assert len(mixed_labels) > 0
        
        # Should include natural labels
        assert "noun" in mixed_labels
        assert "verb" in mixed_labels
        
        # Should include code labels  
        assert "keyword" in mixed_labels
        assert "identifier" in mixed_labels
        
        # Should include mixed-specific labels
        assert "inline_code" in mixed_labels
        assert "code_block" in mixed_labels
    
    def test_get_label_names(self):
        """Test getting label names for each domain."""
        natural_names = XBarLabelMap.get_label_names(DomainType.NATURAL)
        code_names = XBarLabelMap.get_label_names(DomainType.CODE)
        mixed_names = XBarLabelMap.get_label_names(DomainType.MIXED)
        
        assert isinstance(natural_names, list)
        assert isinstance(code_names, list)
        assert isinstance(mixed_names, list)
        
        assert len(natural_names) > 0
        assert len(code_names) > 0
        assert len(mixed_names) > 0
        
        # Mixed should be largest (contains all labels)
        assert len(mixed_names) >= len(natural_names)
        assert len(mixed_names) >= len(code_names)
    
    def test_validate_label(self):
        """Test label validation for each domain."""
        # Valid labels for each domain
        assert XBarLabelMap.validate_label("noun", DomainType.NATURAL)
        assert XBarLabelMap.validate_label("keyword", DomainType.CODE)
        assert XBarLabelMap.validate_label("inline_code", DomainType.MIXED)
        
        # Invalid labels
        assert not XBarLabelMap.validate_label("invalid_label", DomainType.NATURAL)
        assert not XBarLabelMap.validate_label("keyword", DomainType.NATURAL)  # keyword not in natural
        
        # Mixed domain should accept natural and code labels
        assert XBarLabelMap.validate_label("noun", DomainType.MIXED)
        assert XBarLabelMap.validate_label("keyword", DomainType.MIXED)
    
    def test_label_descriptions_are_meaningful(self):
        """Test that all labels have meaningful descriptions."""
        for domain in [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]:
            labels = XBarLabelMap.get_labels_for_domain(domain)
            for label_name, description in labels.items():
                assert isinstance(label_name, str)
                assert isinstance(description, str)
                assert len(label_name) > 0
                assert len(description) > 10  # Descriptions should be meaningful
                assert description != label_name  # Description should be different from name    
    def test_get_hierarchical_level_word_level(self):
        """Test hierarchical level classification for word-level labels."""
        word_level_labels = [
            "noun", "verb", "adjective", "adverb", "determiner", "preposition",
            "pronoun", "conjunction", "punctuation", "keyword", "identifier", 
            "operator", "literal", "delimiter", "type_name", "comment",
            "proper_noun", "proper noun", "parenthesis", "colon", "prefix", "numeral"
        ]
        
        for label in word_level_labels:
            level = XBarLabelMap.get_hierarchical_level(label)
            assert level == "word_level", f"Expected 'word_level' for '{label}', got '{level}'"
    
    def test_get_hierarchical_level_phrase_level(self):
        """Test hierarchical level classification for phrase-level labels."""
        phrase_level_labels = [
            "noun_phrase", "verb_phrase", "adjective_phrase", "adverb_phrase", 
            "prepositional_phrase", "expression", "function_call", "assignment",
            "parameter_list", "argument_list", "inline_code", "code_block", "code_expression"
        ]
        
        for label in phrase_level_labels:
            level = XBarLabelMap.get_hierarchical_level(label)
            assert level == "phrase_level", f"Expected 'phrase_level' for '{label}', got '{level}'"
    
    def test_get_hierarchical_level_clause_level(self):
        """Test hierarchical level classification for clause-level labels."""
        clause_level_labels = [
            "main_clause", "subordinate_clause", "relative_clause", "if_statement",
            "loop_statement", "function_definition", "class_definition", 
            "import_statement", "return_statement", "documentation_comment",
            "code_statement", "code statement"
        ]
        
        for label in clause_level_labels:
            level = XBarLabelMap.get_hierarchical_level(label)
            assert level == "clause_level", f"Expected 'clause_level' for '{label}', got '{level}'"
    
    def test_get_hierarchical_level_multi_label(self):
        """Test hierarchical level classification for multi-label cases."""
        multi_label_cases = [
            ("noun, punctuation", "word_level"),  # Should take first valid label
            ("verb, identifier", "word_level"),
            ("expression, keyword", "phrase_level"),  # Expression wins over keyword
        ]
        
        for label, expected_level in multi_label_cases:
            level = XBarLabelMap.get_hierarchical_level(label)
            assert level == expected_level, f"Expected '{expected_level}' for '{label}', got '{level}'"
    
    def test_get_hierarchical_level_pattern_matching(self):
        """Test hierarchical level classification using pattern matching."""
        pattern_cases = [
            ("some_noun_variant", "word_level"),  # Contains 'noun'
            ("complex_expression", "phrase_level"),  # Contains 'expression'
            ("custom_statement", "clause_level"),  # Contains 'statement'
            ("code_something", "phrase_level"),  # Contains 'code' (default to phrase)
        ]
        
        for label, expected_level in pattern_cases:
            level = XBarLabelMap.get_hierarchical_level(label)
            assert level == expected_level, f"Expected '{expected_level}' for '{label}', got '{level}'"
    
    def test_get_hierarchical_level_edge_cases(self):
        """Test hierarchical level classification for edge cases."""
        edge_cases = [
            ("", None),  # Empty string
            ("   ", None),  # Whitespace only
            (None, None),  # None input
            ("unknown_label_type", None),  # Completely unknown
        ]
        
        for label, expected_level in edge_cases:
            level = XBarLabelMap.get_hierarchical_level(label)
            assert level == expected_level, f"Expected '{expected_level}' for '{label}', got '{level}'"
    
    def test_get_hierarchical_level_case_insensitive(self):
        """Test that hierarchical level classification is case insensitive."""
        test_cases = [
            ("NOUN", "word_level"),
            ("Verb_Phrase", "phrase_level"),
            ("CODE_STATEMENT", "clause_level"),
            ("Proper Noun", "word_level"),
        ]
        
        for label, expected_level in test_cases:
            level = XBarLabelMap.get_hierarchical_level(label)
            assert level == expected_level, f"Expected '{expected_level}' for '{label}', got '{level}'"
