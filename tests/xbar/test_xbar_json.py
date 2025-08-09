#!/usr/bin/env python3
"""
Unit tests for simplified X-bar JSON parser.

Tests basic JSON parsing and filtering functionality.
No repair tests since repair functionality has been removed.
"""
import unittest
import json
from typing import List, Dict, Any

from x_spanformer.xbar.xbar_json import XBarJsonParser


class TestXBarJsonParser(unittest.TestCase):
    """Test cases for simplified XBarJsonParser."""
    
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
    
    def test_malformed_json_skipped(self):
        """Test that malformed JSON is handled appropriately - some can be auto-fixed."""
        malformed_cases = [
            ('[{"text","value","xbar_label":"noun"}]', 0),     # Missing colon - too malformed
            ('[{"text":"value",,"xbar_label":"noun"}]', 0),    # Double comma - too malformed
            ('[{text:"value",xbar_label:"noun"}]', 1),         # Unquoted property - can be fixed
            ('[{"text":"value","xbar_label":"noun"]', 0),      # Missing closing brace - too malformed
            ('{"text":"incomplete"', 0),                       # Truly truncated JSON - too malformed
        ]
        
        for malformed, expected_count in malformed_cases:
            with self.subTest(malformed=malformed):
                result = self.parser.parse_json_response(malformed)
                self.assertEqual(len(result), expected_count, 
                               f"Expected {expected_count} results for: {malformed}")
                
        # Test partially valid case - should extract valid parts
        partially_valid = '[{"text":"hello","xbar_label":"noun"},{"text":"wo'
        result = self.parser.parse_json_response(partially_valid)
        # Should extract the valid first object
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'hello')
    
    def test_empty_text_removal(self):
        """Test removal of entries with empty text values."""
        valid_json = '[{"text":"","xbar_label":"noun"},{"text":"valid","xbar_label":"verb"}]'
        
        result = self.parser.parse_json_response(valid_json)
        # Should filter out the empty text entry
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'valid')
    
    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        empty_responses = ['', '   ', '\n\n', '[]', '[  ]']
        
        for response in empty_responses:
            with self.subTest(response=repr(response)):
                result = self.parser.parse_json_response(response)
                self.assertEqual(len(result), 0)
    
    def test_null_value_handling(self):
        """Test handling of null values in valid JSON."""
        valid_json_with_null = '[{"text":"value","xbar_label":null}]'
        
        result = self.parser.parse_json_response(valid_json_with_null)
        # Should filter out null labels completely
        self.assertEqual(len(result), 0)
    
    def test_alternative_label_keys(self):
        """Test handling of alternative label key names in valid JSON."""
        valid_json = '''[
            {"text": "test1", "label": "noun"},
            {"text": "test2", "xbar_class": "verb"},
            {"text": "test3", "class": "adjective"},
            {"text": "test4", "xbar_label": "adverb"}
        ]'''
        
        result = self.parser.parse_json_response(valid_json)
        self.assertEqual(len(result), 4)
        
        # All should be converted to xbar_label
        for ann in result:
            self.assertIn('xbar_label', ann)
    
    def test_filter_garbage_annotations(self):
        """Test filtering of garbage/placeholder annotations."""
        annotations = [
            {'text': 'valid', 'xbar_label': 'noun'},
            {'text': '', 'xbar_label': 'verb'},  # Empty text - filtered
            {'text': 'text', 'xbar_label': 'label'},  # Placeholder values - filtered
            {'text': 'good', 'xbar_label': ''},  # Empty label - filtered
            {'text': ',', 'xbar_label': 'punctuation'},  # Meaningful punctuation - kept
            {'text': ',,,,', 'xbar_label': 'artifact'},  # Repeated chars - filtered
            {'text': 'another', 'xbar_label': 'verb'}  # Valid - kept
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        
        # Should keep: valid, punctuation (with proper label), another = 3 total
        self.assertEqual(len(filtered), 3)
        valid_texts = [ann['text'] for ann in filtered]
        self.assertIn('valid', valid_texts)
        self.assertIn(',', valid_texts)  # Should keep punctuation with proper label
        self.assertIn('another', valid_texts)
    
    def test_filter_repetitive_patterns(self):
        """Test filtering of repetitive/hallucinated patterns."""
        annotations = [
            {'text': 'valid', 'xbar_label': 'noun'},
            {'text': '....', 'xbar_label': 'punctuation'},  # Repetitive dots - filtered
            {'text': '---', 'xbar_label': 'delimiter'},  # Repetitive dashes - filtered
            {'text': '((((', 'xbar_label': 'bracket'},  # Repetitive brackets - filtered
            {'text': ',', 'xbar_label': 'punctuation'},  # Single punctuation - kept
            {'text': 'word', 'xbar_label': 'noun'},  # Valid word - kept
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        
        # Should keep: valid, single comma, word = 3 total
        self.assertEqual(len(filtered), 3)
        valid_texts = [ann['text'] for ann in filtered]
        self.assertIn('valid', valid_texts)
        self.assertIn(',', valid_texts)
        self.assertIn('word', valid_texts)
        # Should not include repetitive patterns
        self.assertNotIn('....', valid_texts)
        self.assertNotIn('---', valid_texts)
        self.assertNotIn('((((', valid_texts)
    
    def test_preserve_overlapping_spans(self):
        """Test that legitimate overlapping spans are preserved."""
        annotations = [
            {'text': 'function', 'xbar_label': 'noun'},  # Word level
            {'text': 'function call', 'xbar_label': 'noun_phrase'},  # Phrase level
            {'text': 'function', 'xbar_label': 'keyword'},  # Same text, different label
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        
        # Should preserve all three as they have different labels
        self.assertEqual(len(filtered), 3)
    
    def test_exact_duplicate_removal(self):
        """Test that parser preserves duplicates - deduplication happens at sequence level."""
        annotations = [
            {'text': 'hello', 'xbar_label': 'noun'},
            {'text': 'hello', 'xbar_label': 'noun'},  # Exact duplicate - preserved at parse level
            {'text': 'Hello', 'xbar_label': 'noun'},  # Case difference - preserved
            {'text': 'hello', 'xbar_label': 'verb'},  # Different label - kept
        ]
        
        filtered = self.parser.filter_valid_annotations(annotations)
        
        # Parser doesn't deduplicate - all valid annotations preserved
        self.assertEqual(len(filtered), 4)
    
    def test_single_object_json(self):
        """Test parsing single object JSON (not array)."""
        single_object = '{"text": "single", "xbar_label": "noun"}'
        
        result = self.parser.parse_json_response(single_object)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'single')
    
    def test_non_json_response_skipped(self):
        """Test that non-JSON responses are skipped."""
        non_json_responses = [
            "This is just plain text",
            "Here are some words but no JSON",
            "```\nSome code but not JSON\n```",
            "The annotations are: word1, word2, word3"
        ]
        
        for response in non_json_responses:
            with self.subTest(response=response):
                result = self.parser.parse_json_response(response)
                self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
