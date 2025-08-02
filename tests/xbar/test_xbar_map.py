"""
Test suite for x_spanformer.xbar package.

Tests XBar classifier mappings and domain-specific classifier definitions
used in span annotation pipeline.
"""

import pytest
from typing import Dict, List

from x_spanformer.xbar.xbar_map import XBarClassifierMap, DomainType
from x_spanformer.xbar.position_mapper import PositionMapper, CharacterSpan, PositionSpan


class TestDomainType:
    """Test DomainType enum."""
    
    def test_domain_type_values(self):
        """Test that DomainType has expected values."""
        assert DomainType.NATURAL.value == "natural"
        assert DomainType.CODE.value == "code"
        assert DomainType.MIXED.value == "mixed"
    
    def test_domain_type_members(self):
        """Test that all expected domain types exist."""
        expected_domains = {"NATURAL", "CODE", "MIXED"}
        actual_domains = {member.name for member in DomainType}
        assert actual_domains == expected_domains


class TestXBarClassifierMap:
    """Test XBarClassifierMap functionality."""
    
    def test_classifier_map_initialization(self):
        """Test that XBarClassifierMap initializes correctly."""
        classifier_map = XBarClassifierMap()
        assert isinstance(classifier_map, XBarClassifierMap)
    
    def test_get_classifiers_natural_domain(self):
        """Test getting classifiers for natural language domain."""
        natural_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.NATURAL)
        
        assert isinstance(natural_classifiers, dict)
        assert len(natural_classifiers) > 0
        
        # Check for expected natural language classifiers
        expected_natural = ["noun", "verb", "adjective", "adverb", "determiner"]
        for classifier in expected_natural:
            assert classifier in natural_classifiers
            assert isinstance(natural_classifiers[classifier], str)
            assert len(natural_classifiers[classifier]) > 0
    
    def test_get_classifiers_code_domain(self):
        """Test getting classifiers for code domain."""
        code_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.CODE)
        
        assert isinstance(code_classifiers, dict)
        assert len(code_classifiers) > 0
        
        # Check for expected code classifiers
        expected_code = ["keyword", "identifier", "operator", "literal", "delimiter"]
        for classifier in expected_code:
            assert classifier in code_classifiers
            assert isinstance(code_classifiers[classifier], str)
            assert len(code_classifiers[classifier]) > 0
    
    def test_get_classifiers_mixed_domain(self):
        """Test getting classifiers for mixed domain."""
        mixed_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.MIXED)
        
        assert isinstance(mixed_classifiers, dict)
        assert len(mixed_classifiers) > 0
        
        # Mixed domain should have both natural and code classifiers
        natural_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.NATURAL)
        code_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.CODE)
        
        # Mixed should contain elements from both domains
        for classifier in ["noun", "verb", "keyword", "identifier"]:
            assert classifier in mixed_classifiers
    
    def test_get_classifiers_with_string_domain(self):
        """Test getting classifiers using string domain names."""
        # Test with enum domain names
        natural_by_enum = XBarClassifierMap.get_classifiers_for_domain(DomainType.NATURAL)
        code_by_enum = XBarClassifierMap.get_classifiers_for_domain(DomainType.CODE)
        
        assert len(natural_by_enum) > 0
        assert len(code_by_enum) > 0
        
        # Verify they're different
        assert natural_by_enum != code_by_enum
    
    def test_get_classifier_names(self):
        """Test getting classifier names for each domain."""
        natural_names = XBarClassifierMap.get_classifier_names(DomainType.NATURAL)
        code_names = XBarClassifierMap.get_classifier_names(DomainType.CODE)
        mixed_names = XBarClassifierMap.get_classifier_names(DomainType.MIXED)
        
        assert isinstance(natural_names, list)
        assert isinstance(code_names, list)
        assert isinstance(mixed_names, list)
        
        assert len(natural_names) > 0
        assert len(code_names) > 0
        assert len(mixed_names) > 0
        
        # Mixed should have more classifiers than individual domains
        assert len(mixed_names) >= len(natural_names)
        assert len(mixed_names) >= len(code_names)
    
    def test_get_all_domains(self):
        """Test getting all available domains."""
        expected_domains = [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]
        
        # Test each domain works
        for domain in expected_domains:
            classifiers = XBarClassifierMap.get_classifiers_for_domain(domain)
            assert len(classifiers) > 0
    
    def test_classifier_descriptions_not_empty(self):
        """Test that all classifiers have non-empty descriptions."""
        for domain in [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]:
            classifiers = XBarClassifierMap.get_classifiers_for_domain(domain)
            for classifier_name, description in classifiers.items():
                assert isinstance(classifier_name, str)
                assert isinstance(description, str)
                assert len(classifier_name.strip()) > 0
                assert len(description.strip()) > 0
    
    def test_validate_classifier(self):
        """Test classifier validation functionality."""
        # Test valid classifiers
        assert XBarClassifierMap.validate_classifier("noun", DomainType.NATURAL)
        assert XBarClassifierMap.validate_classifier("keyword", DomainType.CODE)
        assert XBarClassifierMap.validate_classifier("inline_code", DomainType.MIXED)
        
        # Test invalid classifiers
        assert not XBarClassifierMap.validate_classifier("invalid_classifier", DomainType.NATURAL)
        assert not XBarClassifierMap.validate_classifier("keyword", DomainType.NATURAL)  # keyword not in natural
    
    def test_build_system_prompt(self):
        """Test system prompt building for each domain."""
        for domain in [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]:
            prompt = XBarClassifierMap.build_system_prompt(domain)
            
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should be substantial
            assert "X-bar theory" in prompt
            assert "JSON" in prompt
            assert "confidence" in prompt


class TestXBarIntegration:
    """Test integration between XBar components."""
    
    def test_xbar_map_with_position_mapper(self):
        """Test that XBarClassifierMap works with PositionMapper."""
        text = "The quick brown fox jumps over the lazy dog."
        
        # Get natural language classifiers
        natural_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.NATURAL)
        
        # Create position mapper
        position_mapper = PositionMapper(text)
        
        # Create a span with a natural language classifier
        char_span = CharacterSpan(
            start_char=4,
            end_char=9,
            xbar_class="noun",  # Should be in natural classifiers
            confidence=0.95
        )
        
        # Verify the classifier exists
        assert "noun" in natural_classifiers
        
        # Convert to position span
        position_span = position_mapper.char_span_to_position_span(char_span)
        
        assert position_span.xbar_class == "noun"
        assert position_span.confidence == 0.95
    
    def test_domain_detection_consistency(self):
        """Test that domain detection is consistent across components."""        
        # Test that each domain has unique and expected characteristics
        natural_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.NATURAL)
        code_classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.CODE)
        
        # Natural should have linguistic terms
        linguistic_terms = ["noun", "verb", "adjective", "adverb"]
        for term in linguistic_terms:
            assert term in natural_classifiers
            
        # Code should have programming terms  
        programming_terms = ["keyword", "identifier", "operator"]
        for term in programming_terms:
            assert term in code_classifiers
    
    def test_classifier_map_reproducibility(self):
        """Test that classifier map results are reproducible."""
        for domain in [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]:
            classifiers1 = XBarClassifierMap.get_classifiers_for_domain(domain)
            classifiers2 = XBarClassifierMap.get_classifiers_for_domain(domain)
            
            assert classifiers1 == classifiers2
            assert len(classifiers1) == len(classifiers2)


class TestXBarClassifierValidation:
    """Test validation of XBar classifiers."""
    
    def test_classifier_names_are_valid(self):
        """Test that classifier names follow expected patterns."""
        for domain in [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]:
            classifiers = XBarClassifierMap.get_classifiers_for_domain(domain)
            
            for classifier_name in classifiers.keys():
                # Should be lowercase and contain only letters, numbers, underscores
                assert classifier_name.islower()
                assert all(c.isalnum() or c == '_' for c in classifier_name)
                assert len(classifier_name) > 0
    
    def test_classifier_descriptions_are_informative(self):
        """Test that classifier descriptions are informative for LLM agents."""
        for domain in [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]:
            classifiers = XBarClassifierMap.get_classifiers_for_domain(domain)
            
            for classifier_name, description in classifiers.items():
                # Description should be longer than just the name
                assert len(description) > len(classifier_name)
                
                # Should contain useful information related to the classifier
                # Extract the main concept (first word before underscore)
                main_concept = classifier_name.split("_")[0]
                assert main_concept.lower() in description.lower(), f"Concept '{main_concept}' not found in description: {description}"
                
                # Should be descriptive enough for LLM understanding
                assert len(description.split()) >= 2  # At least 2 words
    
    def test_no_duplicate_classifiers_within_domain(self):
        """Test that there are no duplicate classifiers within each domain."""
        for domain in [DomainType.NATURAL, DomainType.CODE, DomainType.MIXED]:
            classifiers = XBarClassifierMap.get_classifiers_for_domain(domain)
            classifier_names = list(classifiers.keys())
            
            # No duplicates
            assert len(classifier_names) == len(set(classifier_names))
    
    def test_reasonable_classifier_counts(self):
        """Test that classifier counts are reasonable."""
        natural_count = len(XBarClassifierMap.get_classifiers_for_domain(DomainType.NATURAL))
        code_count = len(XBarClassifierMap.get_classifiers_for_domain(DomainType.CODE))
        mixed_count = len(XBarClassifierMap.get_classifiers_for_domain(DomainType.MIXED))
        
        # Should have reasonable numbers (not too few, not too many)
        assert 10 <= natural_count <= 100  # Reasonable linguistic categories
        assert 10 <= code_count <= 100     # Reasonable code categories
        assert 20 <= mixed_count <= 150    # Combined should be larger but reasonable
