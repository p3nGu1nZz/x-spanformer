"""
Test suite for position_mapper module.

Tests position mapping functionality for converting between character-based
and position-based spans in X-Spanformer pipeline.
"""

import pytest
from typing import List

from x_spanformer.xbar.position_mapper import (
    PositionMapper, 
    CharacterSpan, 
    PositionSpan
)


class TestCharacterSpan:
    """Test CharacterSpan dataclass."""
    
    def test_character_span_creation(self):
        """Test creating a CharacterSpan."""
        span = CharacterSpan(
            start_char=10,
            end_char=20,
            xbar_class="noun",
            confidence=0.85
        )
        
        assert span.start_char == 10
        assert span.end_char == 20
        assert span.xbar_class == "noun"
        assert span.confidence == 0.85
    
    def test_character_span_validation(self):
        """Test CharacterSpan validation."""
        # Valid span
        span = CharacterSpan(0, 5, "verb", 0.9)
        assert span.start_char < span.end_char
        
        # Test that confidence can be in valid range
        span_high = CharacterSpan(0, 5, "noun", 1.0)
        span_low = CharacterSpan(0, 5, "noun", 0.0)
        assert span_high.confidence == 1.0
        assert span_low.confidence == 0.0
    
    def test_character_span_length(self):
        """Test CharacterSpan length calculation."""
        span = CharacterSpan(5, 15, "adjective", 0.8)
        assert span.end_char - span.start_char == 10


class TestPositionSpan:
    """Test PositionSpan dataclass."""
    
    def test_position_span_creation(self):
        """Test creating a PositionSpan."""
        span = PositionSpan(
            start_pos=1,
            end_pos=5,
            xbar_class="keyword",
            confidence=0.92
        )
        
        assert span.start_pos == 1
        assert span.end_pos == 5
        assert span.xbar_class == "keyword"
        assert span.confidence == 0.92
    
    def test_position_span_validation(self):
        """Test PositionSpan validation."""
        # Valid span
        span = PositionSpan(2, 8, "identifier", 0.75)
        assert span.start_pos < span.end_pos
        
        # Test edge cases
        span_zero = PositionSpan(0, 5, "noun", 0.8)
        assert span_zero.start_pos == 0
    
    def test_position_span_length(self):
        """Test PositionSpan length calculation."""
        span = PositionSpan(3, 12, "operator", 0.88)
        assert span.end_pos - span.start_pos == 9


class TestPositionMapper:
    """Test PositionMapper functionality."""
    
    def test_position_mapper_initialization(self):
        """Test PositionMapper initialization."""
        text = "Hello world! This is a test."
        mapper = PositionMapper(text)
        
        assert mapper.text == text
        assert hasattr(mapper, 'char_to_pos')
        assert hasattr(mapper, 'pos_to_char')
    
    def test_position_mapper_with_empty_text(self):
        """Test PositionMapper with empty text."""
        mapper = PositionMapper("")
        assert mapper.text == ""
        assert len(mapper.char_to_pos) == 0
    
    def test_position_mapper_with_whitespace_text(self):
        """Test PositionMapper with whitespace-only text."""
        text = "   \n\t  "
        mapper = PositionMapper(text)
        assert mapper.text == text
        # Should handle whitespace properly
        assert len(mapper.char_to_pos) == len(text)
    
    def test_char_span_to_position_span_basic(self):
        """Test basic character to position span conversion."""
        text = "The quick brown fox"
        mapper = PositionMapper(text)
        
        # "quick" spans characters 4-9
        char_span = CharacterSpan(4, 9, "adjective", 0.9)
        position_span = mapper.char_span_to_position_span(char_span)
        
        assert isinstance(position_span, PositionSpan)
        assert position_span.xbar_class == "adjective"
        assert position_span.confidence == 0.9
        assert position_span.start_pos >= 0
        assert position_span.end_pos > position_span.start_pos
    
    def test_position_span_to_char_span_basic(self):
        """Test basic position to character span conversion."""
        text = "The quick brown fox"
        mapper = PositionMapper(text)
        
        # Create a position span and convert back
        position_span = PositionSpan(1, 2, "determiner", 0.85)
        char_span = mapper.position_span_to_char_span(position_span)
        
        assert isinstance(char_span, CharacterSpan)
        assert char_span.xbar_class == "determiner"
        assert char_span.confidence == 0.85
        assert char_span.start_char >= 0
        assert char_span.end_char > char_span.start_char
    
    def test_bidirectional_conversion(self):
        """Test that conversion is bidirectional (round-trip)."""
        text = "function calculateSum(a, b) { return a + b; }"
        mapper = PositionMapper(text)
        
        # Start with character span
        original_char_span = CharacterSpan(9, 21, "identifier", 0.95)
        
        # Convert to position span and back
        position_span = mapper.char_span_to_position_span(original_char_span)
        recovered_char_span = mapper.position_span_to_char_span(position_span)
        
        # Should preserve xbar_class and confidence
        assert recovered_char_span.xbar_class == original_char_span.xbar_class
        assert recovered_char_span.confidence == original_char_span.confidence
        
        # Character positions should be close or exact (allowing for tokenization differences)
        assert abs(recovered_char_span.start_char - original_char_span.start_char) <= 2
        assert abs(recovered_char_span.end_char - original_char_span.end_char) <= 2
    
    def test_multiple_spans_conversion(self):
        """Test converting multiple spans."""
        text = "def process_data(input_file): return parse(input_file)"
        mapper = PositionMapper(text)
        
        char_spans = [
            CharacterSpan(0, 3, "keyword", 0.98),      # "def"
            CharacterSpan(4, 16, "identifier", 0.92),   # "process_data"
            CharacterSpan(17, 27, "identifier", 0.88),  # "input_file"
            CharacterSpan(30, 36, "keyword", 0.95),     # "return"
        ]
        
        position_spans = [mapper.char_span_to_position_span(cs) for cs in char_spans]
        
        assert len(position_spans) == len(char_spans)
        for i, ps in enumerate(position_spans):
            assert ps.xbar_class == char_spans[i].xbar_class
            assert ps.confidence == char_spans[i].confidence
    
    def test_edge_case_start_of_text(self):
        """Test span at start of text."""
        text = "Hello world"
        mapper = PositionMapper(text)
        
        # Span at very beginning
        char_span = CharacterSpan(0, 5, "interjection", 0.8)
        position_span = mapper.char_span_to_position_span(char_span)
        
        assert position_span.start_pos == 0
        assert position_span.xbar_class == "interjection"
    
    def test_edge_case_end_of_text(self):
        """Test span at end of text."""
        text = "Hello world"
        mapper = PositionMapper(text)
        
        # Span at very end
        char_span = CharacterSpan(6, 11, "noun", 0.7)
        position_span = mapper.char_span_to_position_span(char_span)
        
        assert position_span.xbar_class == "noun"
        assert position_span.end_pos <= len(text)
    
    def test_invalid_character_span_bounds(self):
        """Test handling of invalid character span bounds."""
        text = "Short text"
        mapper = PositionMapper(text)
        
        # Span beyond text length - should be handled gracefully
        char_span = CharacterSpan(0, 100, "noun", 0.5)
        position_span = mapper.char_span_to_position_span(char_span)
        assert position_span.end_pos <= len(text)
        
        # Negative start - should be handled gracefully  
        char_span = CharacterSpan(-5, 5, "verb", 0.5)
        position_span = mapper.char_span_to_position_span(char_span)
        assert position_span.start_pos >= 0
    
    def test_position_mapping_with_punctuation(self):
        """Test position mapping with punctuation and special characters."""
        text = "Hello, world! How are you? I'm fine."
        mapper = PositionMapper(text)
        
        # Span including punctuation
        char_span = CharacterSpan(5, 13, "punctuation", 0.6)  # ", world!"
        position_span = mapper.char_span_to_position_span(char_span)
        
        assert position_span.xbar_class == "punctuation"
        assert position_span.confidence == 0.6
    
    def test_position_mapping_with_numbers(self):
        """Test position mapping with numeric content."""
        text = "The answer is 42 and π ≈ 3.14159"
        mapper = PositionMapper(text)
        
        # Span over number
        char_span = CharacterSpan(14, 16, "literal", 0.99)  # "42"
        position_span = mapper.char_span_to_position_span(char_span)
        
        assert position_span.xbar_class == "literal"
        assert position_span.confidence == 0.99
    
    def test_position_mapping_reproducibility(self):
        """Test that position mapping is reproducible."""
        text = "Reproducible test case"
        mapper1 = PositionMapper(text)
        mapper2 = PositionMapper(text)
        
        char_span = CharacterSpan(12, 16, "noun", 0.8)
        
        position_span1 = mapper1.char_span_to_position_span(char_span)
        position_span2 = mapper2.char_span_to_position_span(char_span)
        
        assert position_span1.start_pos == position_span2.start_pos
        assert position_span1.end_pos == position_span2.end_pos
        assert position_span1.xbar_class == position_span2.xbar_class
        assert position_span1.confidence == position_span2.confidence
    
    def test_get_text_for_character_span(self):
        """Test extracting text for character span."""
        text = "The quick brown fox jumps"
        mapper = PositionMapper(text)
        
        char_span = CharacterSpan(4, 9, "adjective", 0.8)  # "quick"
        extracted_text = text[char_span.start_char:char_span.end_char]
        
        assert extracted_text == "quick"


class TestPositionMapperIntegration:
    """Test PositionMapper integration with other components."""
    
    def test_mapper_with_code_text(self):
        """Test position mapper with code text."""
        code_text = """def calculate_average(numbers):
    total = sum(numbers)  
    return total / len(numbers)"""
        
        mapper = PositionMapper(code_text)
        
        # Test multiple code spans
        spans = [
            CharacterSpan(0, 3, "keyword", 0.98),      # "def"
            CharacterSpan(4, 21, "identifier", 0.95),   # "calculate_average"
            CharacterSpan(22, 29, "identifier", 0.90),  # "numbers"
        ]
        
        for char_span in spans:
            position_span = mapper.char_span_to_position_span(char_span)
            recovered_span = mapper.position_span_to_char_span(position_span)
            
            assert recovered_span.xbar_class == char_span.xbar_class
            assert abs(recovered_span.confidence - char_span.confidence) < 0.01
    
    def test_mapper_with_natural_text(self):
        """Test position mapper with natural language text."""
        natural_text = "The beautiful sunset painted the sky in vibrant colors."
        mapper = PositionMapper(natural_text)
        
        # Test natural language spans
        spans = [
            CharacterSpan(0, 3, "determiner", 0.95),    # "The"
            CharacterSpan(4, 13, "adjective", 0.90),    # "beautiful"
            CharacterSpan(14, 20, "noun", 0.88),        # "sunset"
            CharacterSpan(21, 28, "verb", 0.85),        # "painted"
        ]
        
        for char_span in spans:
            position_span = mapper.char_span_to_position_span(char_span)
            assert position_span.xbar_class == char_span.xbar_class
            assert position_span.confidence == char_span.confidence
    
    def test_mapper_performance_with_long_text(self):
        """Test mapper performance with longer text."""
        # Create a reasonably long text
        long_text = "This is a test. " * 100  # 1600 characters
        mapper = PositionMapper(long_text)
        
        # Create multiple spans
        spans = [
            CharacterSpan(i, i+4, "test", 0.5)
            for i in range(0, min(100, len(long_text)-4), 20)
        ]
        
        # Should handle conversion efficiently
        for char_span in spans:
            position_span = mapper.char_span_to_position_span(char_span)
            assert position_span.xbar_class == "test"
            assert position_span.confidence == 0.5
