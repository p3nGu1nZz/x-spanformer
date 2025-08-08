#!/usr/bin/env python3
"""
Unit tests for X-bar JSON parser and repair functionality.

Tests all the edge cases and repair patterns that have been identified
during processing of LLM-generated annotation responses.
"""
import unittest
import json
from typing import List, Dict, Any

from x_spanformer.xbar.xbar_json import XBarJsonParser


class TestXBarJsonParser(unittest.TestCase):
    """Test cases for XBarJsonParser."""
    
    def setUp(self):
        """Set up test instance."""
        self.parser = XBarJsonParser()
    
    def test_valid_json_parsing(self):
        """Test parsing of valid JSON responses."""
        valid_json = '''[
            {"text": "hello", "xbar_label": "noun"},
            {"text": "world", "xbar_label": "noun"}
        ]'''
        
        result = self.parser.parse_json_response(valid_json)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'hello')
        self.assertEqual(result[0]['xbar_label'], 'noun')
    
    def test_json_code_block_extraction(self):
        """Test extraction from markdown code blocks."""
        response = '''Here are the annotations:
        ```json
        [{"text": "function", "xbar_label": "keyword"}]
        ```
        '''
        
        result = self.parser.parse_json_response(response)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'function')
        self.assertEqual(result[0]['xbar_label'], 'keyword')
    
    def test_generic_code_block_extraction(self):
        """Test extraction from generic code blocks."""
        response = '''```
        [{"text": "variable", "xbar_label": "identifier"}]
        ```'''
        
        result = self.parser.parse_json_response(response)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'variable')
    
    def test_comma_instead_of_colon_repair(self):
        """Test repair of comma instead of colon in text field."""
        malformed = '[{"text","value","xbar_label":"noun"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        # Should repair to valid JSON
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['text'], 'value')
        self.assertEqual(parsed[0]['xbar_label'], 'noun')
    
    def test_missing_quotes_repair(self):
        """Test repair of missing quotes around property names."""
        malformed = '[{"text":"value",xbar_label:"operator"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['xbar_label'], 'operator')
    
    def test_empty_text_removal(self):
        """Test removal of entries with empty text values."""
        malformed = '[{"text":"","xbar_label":"noun"},{"text":"valid","xbar_label":"verb"}]'
        
        result = self.parser.parse_json_response(malformed)
        # Should filter out the empty text entry
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'valid')
    
    def test_repetitive_garbage_removal(self):
        """Test removal of repetitive garbage blocks."""
        malformed = '''[
            {"text":"valid","xbar_label":"noun"},
            {"text","xbar_label":"",},
            {"text","xbar_label":"",},
            {"text","xbar_label":"",},
            {"text","xbar_label":"",},
            {"text","xbar_label":"",},
            {"text":"another","xbar_label":"verb"}
        ]'''
        
        result = self.parser.parse_json_response(malformed)
        # Should remove the repetitive garbage and keep valid entries
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'valid')
        self.assertEqual(result[1]['text'], 'another')
    
    def test_truncated_json_repair(self):
        """Test repair of truncated JSON responses."""
        truncated = '[{"text":"hello","xbar_label":"noun"},{"text":"wo'
        repaired = self.parser.attempt_json_repair(truncated)
        
        # Should be valid JSON after repair
        parsed = json.loads(repaired)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]['text'], 'hello')
    
    def test_unquoted_property_names_repair(self):
        """Test repair of unquoted property names."""
        malformed = '[{text:"value",xbar_label:"noun"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['text'], 'value')
    
    def test_null_value_repair(self):
        """Test repair of null values in xbar_label."""
        malformed = '[{"text":"value","xbar_label":null}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['xbar_label'], 'unknown')
    
    def test_missing_xbar_label_key_repair(self):
        """Test repair when xbar_label key is missing entirely."""
        malformed = '[{"text":"value","noun"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['xbar_label'], 'noun')
    
    def test_unquoted_text_with_special_chars(self):
        """Test repair of unquoted text containing special characters."""
        malformed = '[{"text"(,"xbar_label":"operator"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['text'], '(')
    
    def test_trailing_comma_repair(self):
        """Test removal of trailing commas."""
        malformed = '[{"text":"value","xbar_label":"noun",}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['text'], 'value')
    
    def test_double_comma_repair(self):
        """Test repair of double commas."""
        malformed = '[{"text":"value",,"xbar_label":"noun"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['text'], 'value')
    
    def test_missing_comma_between_objects(self):
        """Test repair of missing commas between objects."""
        malformed = '[{"text":"one","xbar_label":"noun"}{"text":"two","xbar_label":"verb"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(len(parsed), 2)
    
    def test_multi_word_label_normalization(self):
        """Test normalization of multi-word labels."""
        malformed = '[{"text":"value","xbar_label":"proper noun"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        # Should normalize "proper noun" to just "noun"
        self.assertEqual(parsed[0]['xbar_label'], 'noun')
    
    def test_filter_garbage_annotations(self):
        """Test filtering of garbage/placeholder annotations."""
        annotations = [
            {'text': 'valid', 'xbar_label': 'noun'},
            {'text': '', 'xbar_label': 'verb'},  # Empty text
            {'text': 'text', 'xbar_label': 'label'},  # Placeholder values
            {'text': 'unknown', 'xbar_label': 'noun'},  # Placeholder text
            {'text': 'good', 'xbar_label': ''},  # Empty label
            {'text': ',', 'xbar_label': 'punctuation'},  # Single char punctuation
            {'text': ',,,,', 'xbar_label': 'artifact'},  # Repeated chars
            {'text': 'another', 'xbar_label': 'verb'}  # Valid
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        
        # Should keep only the valid annotations
        self.assertEqual(len(filtered), 2)
        valid_texts = [ann['text'] for ann in filtered]
        self.assertIn('valid', valid_texts)
        self.assertIn('another', valid_texts)
    
    def test_alternative_label_keys(self):
        """Test handling of alternative label key names."""
        annotations = [
            {'text': 'test1', 'label': 'noun'},  # Alternative key
            {'text': 'test2', 'xbar_class': 'verb'},  # Alternative key
            {'text': 'test3', 'class': 'adjective'},  # Alternative key
            {'text': 'test4', 'xbar_label': 'adverb'}  # Standard key
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        self.assertEqual(len(filtered), 4)
        
        # All should be converted to xbar_label
        for ann in filtered:
            self.assertIn('xbar_label', ann)
            self.assertNotIn('label', ann)
            self.assertNotIn('xbar_class', ann)
            self.assertNotIn('class', ann)
    
    def test_preserve_overlapping_spans(self):
        """Test that legitimate overlapping spans are preserved."""
        annotations = [
            {'text': 'function', 'xbar_label': 'noun'},  # Word level
            {'text': 'function call', 'xbar_label': 'noun_phrase'},  # Phrase level
            {'text': 'function', 'xbar_label': 'keyword'},  # Same text, different label
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        
        # Should preserve all three as they have different labels or represent different levels
        self.assertEqual(len(filtered), 3)
    
    def test_exact_duplicate_removal(self):
        """Test removal of exact duplicates."""
        annotations = [
            {'text': 'hello', 'xbar_label': 'noun'},
            {'text': 'hello', 'xbar_label': 'noun'},  # Exact duplicate
            {'text': 'Hello', 'xbar_label': 'noun'},  # Case difference - should be treated as duplicate
            {'text': 'hello', 'xbar_label': 'verb'},  # Different label - should be kept
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        
        # Should keep original + different label version, remove exact duplicate
        self.assertEqual(len(filtered), 2)
    
    def test_complex_malformed_patterns(self):
        """Test repair of complex malformed patterns from real LLM output."""
        # Based on actual sequence 7 and 10 failures
        test_cases = [
            # Pattern that can be repaired: unquoted property name
            ('{"text":"value",xbar_label:"literal"}', True),
            # Valid pattern that should pass  
            ('{"text":"hello","xbar_label":"operator"}', True),
            # Patterns that should be filtered out due to being too malformed
            ('{"text":"form","xbar_label":"noun"', False),    # Missing closing quote and brace
            ('{"text",","xbar_label":"operator"}', False),    # Too broken to repair
            ('{"text","xbar_label":"",}', False),            # Empty values, should be filtered
        ]

        for malformed, should_succeed in test_cases:
            with self.subTest(malformed=malformed, should_succeed=should_succeed):
                if should_succeed:
                    # Should repair to valid JSON
                    result = self.parser.parse_json_response(f'[{malformed}]')
                    self.assertGreater(len(result), 0, f"Should repair and parse: {malformed}")
                    
                    # Check that results are valid
                    for entry in result:
                        self.assertIn('text', entry)
                        self.assertIn('xbar_label', entry)
                        self.assertTrue(entry['text'])  # Non-empty text
                        self.assertTrue(entry['xbar_label'])  # Non-empty label
                else:
                    # Should either fail to repair or filter out invalid entries
                    try:
                        result = self.parser.parse_json_response(f'[{malformed}]')
                        # If it parses, it should filter out invalid entries
                        self.assertEqual(len(result), 0, f"Should filter out invalid entry: {malformed}")
                    except (json.JSONDecodeError, ValueError):
                        # Expected to fail for badly malformed JSON
                        pass
    
    def test_add_custom_repair_pattern(self):
        """Test adding custom repair patterns."""
        # Add a custom pattern
        self.parser.add_repair_pattern(
            r'"custom_error"', 
            r'"fixed_value"', 
            'fix_custom_error'
        )
        
        malformed = '[{"text":"custom_error","xbar_label":"noun"}]'
        repaired = self.parser.attempt_json_repair(malformed)
        
        parsed = json.loads(repaired)
        self.assertEqual(parsed[0]['text'], 'fixed_value')
    
    def test_remove_repair_pattern(self):
        """Test removing repair patterns."""
        original_count = len(self.parser.repair_patterns)
        
        # Add a pattern
        self.parser.add_repair_pattern(r'test', r'replacement', 'test_pattern')
        self.assertEqual(len(self.parser.repair_patterns), original_count + 1)
        
        # Remove it
        self.parser.remove_repair_pattern('test_pattern')
        self.assertEqual(len(self.parser.repair_patterns), original_count)
    
    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        empty_responses = ['', '   ', '\n\n', '[]', '[  ]']
        
        for response in empty_responses:
            with self.subTest(response=repr(response)):
                result = self.parser.parse_json_response(response)
                self.assertEqual(len(result), 0)
    
    def test_malformed_array_structure(self):
        """Test repair of malformed array structures."""
        malformed_arrays = [
            '[{"text":"value","xbar_label":"noun"},]',  # Trailing comma
            '[,{"text":"value","xbar_label":"noun"}]',  # Leading comma
            '{"text":"value","xbar_label":"noun"}]',    # Missing opening bracket
            '[{"text":"value","xbar_label":"noun"',     # Missing closing bracket
        ]
        
        for malformed in malformed_arrays:
            with self.subTest(malformed=malformed):
                try:
                    result = self.parser.parse_json_response(malformed)
                    # Should either parse successfully or return empty result
                    self.assertIsInstance(result, list)
                except Exception as e:
                    self.fail(f"Failed to handle malformed array: {malformed}, error: {e}")


if __name__ == '__main__':
    unittest.main()
