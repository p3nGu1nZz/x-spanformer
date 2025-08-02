"""
Test suite for position_mapper.py - Character to position mapping utilities.

Tests the mapping between LLM agent character-level span annotations
and the position-wise contextual embeddings used by X-Spanformer.
"""

import unittest
from typing import List

from x_spanformer.xbar.position_mapper import (
    PositionMapper,
    CharacterSpan,
    PositionSpan,
    parse_character_spans_from_agent_response
)


class TestCharacterSpan(unittest.TestCase):
    """Test CharacterSpan dataclass."""
    
    def test_character_span_creation(self):
        """Test character span creation."""
        span = CharacterSpan(
            start_char=4,
            end_char=19,
            xbar_class="NP",
            confidence=0.95,
            text="quick brown fox"
        )
        
        self.assertEqual(span.start_char, 4)
        self.assertEqual(span.end_char, 19)
        self.assertEqual(span.xbar_class, "NP")
        self.assertEqual(span.confidence, 0.95)
        self.assertEqual(span.text, "quick brown fox")
    
    def test_character_span_defaults(self):
        """Test character span default values."""
        span = CharacterSpan(
            start_char=0,
            end_char=5,
            xbar_class="Det"
        )
        
        self.assertEqual(span.confidence, 1.0)  # Default confidence
        self.assertIsNone(span.text)  # Default text


class TestPositionSpan(unittest.TestCase):
    """Test PositionSpan dataclass."""
    
    def test_position_span_creation(self):
        """Test position span creation."""
        span = PositionSpan(
            start_pos=4,
            end_pos=20,
            xbar_class="NP",
            confidence=0.88,
            positions=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        
        self.assertEqual(span.start_pos, 4)
        self.assertEqual(span.end_pos, 20)
        self.assertEqual(span.xbar_class, "NP")
        self.assertEqual(span.confidence, 0.88)
        self.assertIsNotNone(span.positions)
        self.assertEqual(len(span.positions or []), 16)
    
    def test_position_span_defaults(self):
        """Test position span default values."""
        span = PositionSpan(
            start_pos=0,
            end_pos=10,
            xbar_class="VP"
        )
        
        self.assertEqual(span.confidence, 1.0)
        self.assertIsNone(span.positions)


class TestPositionMapper(unittest.TestCase):
    """Test PositionMapper class for character-to-position conversion."""
    
    def setUp(self):
        """Set up test data."""
        self.text = "The quick brown fox jumps over the lazy dog."
        self.mapper = PositionMapper(text=self.text)
    
    def test_position_mapper_init(self):
        """Test position mapper initialization."""
        self.assertEqual(self.mapper.text, self.text)
        self.assertEqual(len(self.text), 44)
        
        # Check mapping creation
        self.assertIsNotNone(self.mapper.char_to_pos)
        self.assertIsNotNone(self.mapper.pos_to_char)
        self.assertEqual(len(self.mapper.char_to_pos), len(self.text))
        self.assertEqual(len(self.mapper.pos_to_char), len(self.text))
    
    def test_char_to_pos_mapping(self):
        """Test character to position mapping."""
        # In tokenizer-free architecture, character indices map directly to positions
        for i in range(len(self.text)):
            self.assertEqual(self.mapper.char_to_pos[i], i)
            self.assertEqual(self.mapper.pos_to_char[i], i)
    
    def test_char_span_to_position_span(self):
        """Test character span to position span conversion."""
        char_span = CharacterSpan(
            start_char=4,
            end_char=9,  # "quick" 
            xbar_class="Adj",
            confidence=0.92
        )
        
        pos_span = self.mapper.char_span_to_position_span(char_span)
        
        self.assertEqual(pos_span.start_pos, 4)
        self.assertEqual(pos_span.end_pos, 9)  # Exclusive end
        self.assertEqual(pos_span.xbar_class, "Adj")
        self.assertEqual(pos_span.confidence, 0.92)
        self.assertEqual(pos_span.positions, [4, 5, 6, 7, 8])
    
    def test_position_span_to_char_span(self):
        """Test position span to character span conversion."""
        pos_span = PositionSpan(
            start_pos=10,
            end_pos=15,  # "brown"
            xbar_class="Adj",
            confidence=0.85
        )
        
        char_span = self.mapper.position_span_to_char_span(pos_span)
        
        self.assertEqual(char_span.start_char, 10)
        self.assertEqual(char_span.end_char, 15)
        self.assertEqual(char_span.xbar_class, "Adj")
        self.assertEqual(char_span.confidence, 0.85)
        self.assertEqual(char_span.text, "brown")
    
    def test_batch_char_to_position(self):
        """Test batch character to position conversion."""
        char_spans = [
            CharacterSpan(start_char=0, end_char=3, xbar_class="Det", text="The"),
            CharacterSpan(start_char=4, end_char=9, xbar_class="Adj", text="quick"),
            CharacterSpan(start_char=16, end_char=19, xbar_class="N", text="fox")
        ]
        
        pos_spans = self.mapper.batch_char_to_position(char_spans)
        
        self.assertEqual(len(pos_spans), 3)
        
        # Check first span
        self.assertEqual(pos_spans[0].start_pos, 0)
        self.assertEqual(pos_spans[0].end_pos, 3)
        self.assertEqual(pos_spans[0].xbar_class, "Det")
        
        # Check second span
        self.assertEqual(pos_spans[1].start_pos, 4)
        self.assertEqual(pos_spans[1].end_pos, 9)
        self.assertEqual(pos_spans[1].xbar_class, "Adj")
        
        # Check third span
        self.assertEqual(pos_spans[2].start_pos, 16)
        self.assertEqual(pos_spans[2].end_pos, 19)
        self.assertEqual(pos_spans[2].xbar_class, "N")
    
    def test_batch_position_to_char(self):
        """Test batch position to character conversion."""
        pos_spans = [
            PositionSpan(start_pos=0, end_pos=3, xbar_class="Det"),
            PositionSpan(start_pos=20, end_pos=25, xbar_class="V"),
            PositionSpan(start_pos=35, end_pos=39, xbar_class="Adj")
        ]
        
        char_spans = self.mapper.batch_position_to_char(pos_spans)
        
        self.assertEqual(len(char_spans), 3)
        
        # Check conversions
        self.assertEqual(char_spans[0].text, "The")
        self.assertEqual(char_spans[1].text, "jumps")
        self.assertEqual(char_spans[2].text, "lazy")
    
    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        # Test span at beginning of text
        start_span = CharacterSpan(start_char=0, end_char=3, xbar_class="Det")
        start_pos = self.mapper.char_span_to_position_span(start_span)
        self.assertEqual(start_pos.start_pos, 0)
        self.assertEqual(start_pos.end_pos, 3)
        
        # Test span at end of text
        end_span = CharacterSpan(start_char=40, end_char=43, xbar_class="N")
        end_pos = self.mapper.char_span_to_position_span(end_span)
        self.assertEqual(end_pos.start_pos, 40)
        self.assertEqual(end_pos.end_pos, 43)
        
        # Test full text span
        full_span = CharacterSpan(start_char=0, end_char=44, xbar_class="S")
        full_pos = self.mapper.char_span_to_position_span(full_span)
        self.assertEqual(full_pos.start_pos, 0)
        self.assertEqual(full_pos.end_pos, 44)
    
    def test_out_of_bounds_handling(self):
        """Test handling of out-of-bounds character indices."""
        # Test span with end beyond text length
        out_of_bounds_span = CharacterSpan(
            start_char=40,
            end_char=50,  # Beyond text length (44)
            xbar_class="Test"
        )
        
        pos_span = self.mapper.char_span_to_position_span(out_of_bounds_span)
        
        # Should be clamped to text boundaries
        self.assertEqual(pos_span.start_pos, 40)
        self.assertEqual(pos_span.end_pos, 44)  # Clamped to text length
    
    def test_empty_span_handling(self):
        """Test handling of empty spans."""
        empty_span = CharacterSpan(
            start_char=10,
            end_char=10,  # Empty span
            xbar_class="Empty"
        )
        
        pos_span = self.mapper.char_span_to_position_span(empty_span)
        
        # Should handle gracefully
        self.assertEqual(pos_span.start_pos, 10)
        self.assertEqual(pos_span.end_pos, 10)
        self.assertEqual(len(pos_span.positions or []), 0)


class TestAgentResponseParsing(unittest.TestCase):
    """Test parsing of LLM agent responses."""
    
    def test_parse_character_spans_basic(self):
        """Test basic parsing of character spans from agent response."""
        response = '''Based on X-bar theory analysis:

"The quick brown fox" (0-18) -> NP [confidence: 0.88]
"jumps" (20-24) -> V [confidence: 0.95]
"over the lazy dog" (26-42) -> PP [confidence: 0.87]

The sentence shows a simple clause structure.'''
        
        text = "The quick brown fox jumps over the lazy dog."
        spans = parse_character_spans_from_agent_response(response, text)
        
        self.assertEqual(len(spans), 3)
        
        # Check first span
        self.assertEqual(spans[0].start_char, 0)
        self.assertEqual(spans[0].end_char, 19)  # Now exclusive (was 18 inclusive)
        self.assertEqual(spans[0].xbar_class, "NP")
        self.assertAlmostEqual(spans[0].confidence, 0.88, places=2)
        self.assertEqual(spans[0].text, "The quick brown fox")
        
        # Check second span
        self.assertEqual(spans[1].start_char, 20)
        self.assertEqual(spans[1].end_char, 25)  # Now exclusive (was 24 inclusive)
        self.assertEqual(spans[1].xbar_class, "V")
        self.assertAlmostEqual(spans[1].confidence, 0.95, places=2)
        
        # Check third span
        self.assertEqual(spans[2].start_char, 26)
        self.assertEqual(spans[2].end_char, 43)  # Now exclusive (was 42 inclusive)
        self.assertEqual(spans[2].xbar_class, "PP")
        self.assertAlmostEqual(spans[2].confidence, 0.87, places=2)
    
    def test_parse_character_spans_malformed(self):
        """Test parsing of malformed agent responses."""
        malformed_response = "This response doesn't contain valid span information."
        text = "Some test text."
        
        spans = parse_character_spans_from_agent_response(malformed_response, text)
        
        # Should return empty list for malformed response
        self.assertEqual(len(spans), 0)
    
    def test_parse_character_spans_partial(self):
        """Test parsing with some valid and some invalid spans."""
        response = '''Analysis:

"Valid span" (0-9) -> NP [confidence: 0.90]
"Invalid format" -> Bad
"Another valid" (11-23) -> VP [confidence: 0.85]'''
        
        text = "Valid span Another valid span text."
        spans = parse_character_spans_from_agent_response(response, text)
        
        # Should parse only the valid spans
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].xbar_class, "NP")
        self.assertEqual(spans[1].xbar_class, "VP")
    
    def test_parse_character_spans_confidence_variations(self):
        """Test parsing with different confidence score formats."""
        response = '''Analysis:

"Span one" (0-7) -> NP [confidence: 0.95]
"Span two" (9-16) -> VP [confidence: 0.8]
"Span three" (18-27) -> PP [confidence: 1.0]
"Span four" (29-37) -> Det [confidence: 0.75]'''
        
        text = "Span one Span two Span three Span four extra"
        spans = parse_character_spans_from_agent_response(response, text)
        
        self.assertEqual(len(spans), 4)
        
        confidences = [span.confidence for span in spans]
        expected_confidences = [0.95, 0.8, 1.0, 0.75]
        
        for actual, expected in zip(confidences, expected_confidences):
            self.assertAlmostEqual(actual, expected, places=2)


class TestPositionMapperIntegration(unittest.TestCase):
    """Integration tests for position mapper with real-world scenarios."""
    
    def setUp(self):
        """Set up integration test data."""
        self.code_text = "def hello_world():\n    print('Hello, World!')"
        self.code_mapper = PositionMapper(text=self.code_text)
        
        self.natural_text = "The quick brown fox jumps over the lazy dog."
        self.natural_mapper = PositionMapper(text=self.natural_text)
    
    def test_code_span_mapping(self):
        """Test span mapping for code text."""
        # Function definition span - need to match actual text length
        func_text = "def hello_world()"  # Without colon for end_char=17
        func_span = CharacterSpan(
            start_char=0,
            end_char=17,  # "def hello_world()"
            xbar_class="FunctionDef",
            text=func_text
        )
        
        pos_span = self.code_mapper.char_span_to_position_span(func_span)
        
        self.assertEqual(pos_span.start_pos, 0)
        self.assertEqual(pos_span.end_pos, 17)
        self.assertEqual(pos_span.xbar_class, "FunctionDef")
        
        # Verify character mapping
        char_span_back = self.code_mapper.position_span_to_char_span(pos_span)
        self.assertEqual(char_span_back.text, func_text)
    
    def test_natural_language_hierarchical_spans(self):
        """Test hierarchical span mapping for natural language."""
        spans = [
            CharacterSpan(start_char=0, end_char=3, xbar_class="Det", text="The"),
            CharacterSpan(start_char=4, end_char=19, xbar_class="AdjP", text="quick brown fox"),
            CharacterSpan(start_char=0, end_char=19, xbar_class="NP", text="The quick brown fox"),
            CharacterSpan(start_char=20, end_char=43, xbar_class="VP", text="jumps over the lazy dog"),
            CharacterSpan(start_char=0, end_char=43, xbar_class="S", text="The quick brown fox jumps over the lazy dog")
        ]
        
        pos_spans = self.natural_mapper.batch_char_to_position(spans)
        
        self.assertEqual(len(pos_spans), 5)
        
        # Check hierarchical nesting
        det_span = pos_spans[0]
        np_span = pos_spans[2]
        s_span = pos_spans[4]
        
        # Det should be contained within NP
        self.assertGreaterEqual(det_span.start_pos, np_span.start_pos)
        self.assertLessEqual(det_span.end_pos, np_span.end_pos)
        
        # NP should be contained within S
        self.assertGreaterEqual(np_span.start_pos, s_span.start_pos)
        self.assertLessEqual(np_span.end_pos, s_span.end_pos)
    
    def test_boundary_training_target_generation(self):
        """Test conversion to boundary training targets."""
        spans = [
            CharacterSpan(start_char=0, end_char=3, xbar_class="Det"),
            CharacterSpan(start_char=4, end_char=19, xbar_class="NP"),
        ]
        
        pos_spans = self.natural_mapper.batch_char_to_position(spans)
        
        # Simulate boundary target generation
        sequence_length = len(self.natural_text)
        start_targets = [0] * sequence_length
        end_targets = [0] * sequence_length
        
        for pos_span in pos_spans:
            start_targets[pos_span.start_pos] = 1
            end_targets[pos_span.end_pos - 1] = 1  # Inclusive end for targets
        
        # Check targets
        self.assertEqual(start_targets[0], 1)  # Det start
        self.assertEqual(start_targets[4], 1)  # NP start
        self.assertEqual(end_targets[2], 1)    # Det end (inclusive)
        self.assertEqual(end_targets[18], 1)   # NP end (inclusive)
        
        # Count total targets
        self.assertEqual(sum(start_targets), 2)
        self.assertEqual(sum(end_targets), 2)


if __name__ == "__main__":
    unittest.main()
