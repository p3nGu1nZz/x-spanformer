"""
Test suite for span_validator module.

Tests the SpanValidator and SpanCleaner classes for X-bar theory validation
and linguistic span filtering functionality.
"""

import pytest
from collections import Counter, defaultdict
from typing import Dict, List

from x_spanformer.xbar.span_validator import SpanValidator, SpanCleaner


class TestSpanValidator:
    """Test SpanValidator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = SpanValidator()
    
    def test_span_validator_initialization(self):
        """Test SpanValidator initialization with proper mappings."""
        assert isinstance(self.validator.word_class_mappings, dict)
        assert isinstance(self.validator.valid_single_char_classes, dict)
        assert isinstance(self.validator.max_repetitions_per_sequence, dict)
        
        # Check key mappings exist
        assert 'in' in self.validator.word_class_mappings
        assert 'the' in self.validator.word_class_mappings
        assert 'punctuation' in self.validator.valid_single_char_classes
        assert 'preposition' in self.validator.max_repetitions_per_sequence
    
    def test_is_word_boundary(self):
        """Test word boundary detection."""
        # Valid word boundaries
        assert self.validator.is_word_boundary(' ')
        assert self.validator.is_word_boundary(',')
        assert self.validator.is_word_boundary('.')
        assert self.validator.is_word_boundary('(')
        assert self.validator.is_word_boundary(')')
        
        # Non-word boundaries
        assert not self.validator.is_word_boundary('a')
        assert not self.validator.is_word_boundary('Z')
        assert not self.validator.is_word_boundary('0')
    
    def test_validate_word_class_match_valid(self):
        """Test word-class validation for valid matches."""
        # Valid prepositions
        assert self.validator.validate_word_class_match('in', 'preposition')
        assert self.validator.validate_word_class_match('on', 'preposition')
        assert self.validator.validate_word_class_match('at', 'preposition')
        
        # Valid determiners
        assert self.validator.validate_word_class_match('the', 'determiner')
        assert self.validator.validate_word_class_match('a', 'determiner')
        assert self.validator.validate_word_class_match('an', 'determiner')
        
        # Valid pronouns
        assert self.validator.validate_word_class_match('I', 'pronoun')
        assert self.validator.validate_word_class_match('you', 'pronoun')
        assert self.validator.validate_word_class_match('he', 'pronoun')
        
        # Words not in mappings should pass (default accept)
        assert self.validator.validate_word_class_match('unknown', 'noun')
        assert self.validator.validate_word_class_match('example', 'verb')
    
    def test_validate_word_class_match_invalid(self):
        """Test word-class validation for invalid matches."""
        # "in" incorrectly classified as determiner (main issue we're fixing)
        assert not self.validator.validate_word_class_match('in', 'determiner')
        
        # Other invalid classifications
        assert not self.validator.validate_word_class_match('the', 'preposition')
        assert not self.validator.validate_word_class_match('and', 'preposition')
        
        # Note: 'I' is not in word_class_mappings, so it passes by default
        # This is correct behavior - only mapped words are strictly validated
    
    def test_validate_single_char_span_valid_punctuation(self):
        """Test validation of valid single character punctuation spans."""
        raw_text = "Hello, world."
        
        # Valid comma as punctuation
        span_info = {
            'text': ',',
            'start_pos': 5,
            'end_pos': 6,
            'xbar_class': 'punctuation',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert is_valid
        assert "Valid punctuation" in reason
        
        # Valid period as punctuation
        span_info = {
            'text': '.',
            'start_pos': 12,
            'end_pos': 13,
            'xbar_class': 'punctuation',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert is_valid
        assert "Valid punctuation" in reason
    
    def test_validate_single_char_span_valid_standalone(self):
        """Test validation of valid standalone single character spans."""
        raw_text = "I went to a store."
        
        # Valid 'I' as pronoun
        span_info = {
            'text': 'I',
            'start_pos': 0,
            'end_pos': 1,
            'xbar_class': 'pronoun',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert is_valid
        assert "Valid standalone pronoun 'I'" in reason
        
        # Valid 'a' as determiner
        span_info = {
            'text': 'a',
            'start_pos': 10,
            'end_pos': 11,
            'xbar_class': 'determiner',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert is_valid
        assert "Valid standalone determiner 'a'" in reason
    
    def test_validate_single_char_span_embedded_invalid(self):
        """Test validation of invalid embedded single character spans."""
        raw_text = "The cat sat."
        
        # Invalid 'a' embedded in "cat"
        span_info = {
            'text': 'a',
            'start_pos': 5,
            'end_pos': 6,
            'xbar_class': 'determiner',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert not is_valid
        assert "embedded in word" in reason
        
        # Invalid 'c' embedded in "cat"
        span_info = {
            'text': 'c',
            'start_pos': 4,
            'end_pos': 5,
            'xbar_class': 'noun',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert not is_valid
        assert "embedded in word" in reason
    
    def test_validate_single_char_span_invalid_standalone(self):
        """Test validation of invalid standalone single character spans."""
        raw_text = "The number 3 is prime."
        
        # Invalid standalone '3' as literal
        span_info = {
            'text': '3',
            'start_pos': 11,
            'end_pos': 12,
            'xbar_class': 'literal',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert not is_valid
        assert "not permitted" in reason
        
        # Test with a truly standalone 's' (need different text)
        raw_text2 = "The letter s stands for something."
        span_info = {
            'text': 's',
            'start_pos': 11,
            'end_pos': 12,
            'xbar_class': 'noun',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text2)
        assert not is_valid
        assert "not permitted" in reason
    
    def test_validate_single_char_span_invalid_class_mismatch(self):
        """Test validation of single chars with wrong class assignments."""
        raw_text = "Use operator = here."
        
        # Invalid: ';' not valid for delimiter class
        span_info = {
            'text': ';',
            'start_pos': 12,
            'end_pos': 13,
            'xbar_class': 'delimiter',
            'length': 1
        }
        is_valid, reason = self.validator.validate_single_char_span(span_info, raw_text)
        assert not is_valid
        assert "not valid for class" in reason
    
    def test_validate_multi_char_span_valid(self):
        """Test validation of valid multi-character spans."""
        # Valid determiner
        span_info = {
            'text': 'the',
            'xbar_class': 'determiner',
            'length': 3
        }
        is_valid, reason = self.validator.validate_multi_char_span(span_info)
        assert is_valid
        assert "appears valid" in reason
        
        # Valid preposition
        span_info = {
            'text': 'through',
            'xbar_class': 'preposition',
            'length': 7
        }
        is_valid, reason = self.validator.validate_multi_char_span(span_info)
        assert is_valid
        assert "appears valid" in reason
    
    def test_validate_multi_char_span_word_class_mismatch(self):
        """Test validation of multi-char spans with word-class mismatches."""
        # Invalid: "in" classified as determiner (main bug we're fixing)
        span_info = {
            'text': 'in',
            'xbar_class': 'determiner',
            'length': 2
        }
        is_valid, reason = self.validator.validate_multi_char_span(span_info)
        assert not is_valid
        assert "incorrectly classified" in reason
        assert "determiner" in reason
        
        # Invalid: "the" classified as preposition
        span_info = {
            'text': 'the',
            'xbar_class': 'preposition',
            'length': 3
        }
        is_valid, reason = self.validator.validate_multi_char_span(span_info)
        assert not is_valid
        assert "incorrectly classified" in reason
    
    def test_validate_multi_char_span_length_issues(self):
        """Test validation of spans with length issues."""
        # Empty span
        span_info = {
            'text': '',
            'xbar_class': 'noun',
            'length': 0
        }
        is_valid, reason = self.validator.validate_multi_char_span(span_info)
        assert not is_valid
        assert "Empty or whitespace-only" in reason
        
        # Length mismatch
        span_info = {
            'text': 'hello',
            'xbar_class': 'noun',
            'length': 10  # Wrong length
        }
        is_valid, reason = self.validator.validate_multi_char_span(span_info)
        assert not is_valid
        assert "Length mismatch" in reason
        
        # Suspiciously long span
        long_text = 'a' * 150
        span_info = {
            'text': long_text,
            'xbar_class': 'noun',
            'length': 150
        }
        is_valid, reason = self.validator.validate_multi_char_span(span_info)
        assert not is_valid
        assert "Suspiciously long span" in reason
    
    def test_validate_span_integration(self):
        """Test the main validate_span method with full annotations."""
        # Valid multi-char span
        annotation = {
            'raw': 'The cat sat on the mat.',
            'span_annotation': {
                'text': 'The',
                'xbar_class': 'determiner',
                'start_pos': 0,
                'end_pos': 3,
                'length': 3
            }
        }
        is_valid, reason = self.validator.validate_span(annotation)
        assert is_valid
        
        # Invalid single-char span (embedded)
        annotation = {
            'raw': 'The cat sat.',
            'span_annotation': {
                'text': 'a',
                'xbar_class': 'determiner',
                'start_pos': 5,
                'end_pos': 6,
                'length': 1
            }
        }
        is_valid, reason = self.validator.validate_span(annotation)
        assert not is_valid
        assert "embedded in word" in reason
    
    def test_check_repetition_limits_valid(self):
        """Test repetition limit checking for valid cases."""
        sequence_word_counts = defaultdict(Counter)
        sequence_word_counts['preposition']['in'] = 3  # Under limit of 5
        
        is_valid, reason = self.validator.check_repetition_limits(
            'in', 'preposition', sequence_word_counts
        )
        assert is_valid
        assert "Within repetition limits" in reason
    
    def test_check_repetition_limits_invalid(self):
        """Test repetition limit checking for invalid cases."""
        sequence_word_counts = defaultdict(Counter)
        sequence_word_counts['preposition']['in'] = 8  # Over limit of 5
        
        is_valid, reason = self.validator.check_repetition_limits(
            'in', 'preposition', sequence_word_counts
        )
        assert not is_valid
        assert "Excessive repetition" in reason
        assert "appears 8 times" in reason
        assert "max 5" in reason
    
    def test_check_repetition_limits_non_tracked_class(self):
        """Test repetition limits for classes not being tracked."""
        sequence_word_counts = defaultdict(Counter)
        sequence_word_counts['noun']['example'] = 100  # Lots, but not tracked
        
        is_valid, reason = self.validator.check_repetition_limits(
            'example', 'noun', sequence_word_counts
        )
        assert is_valid  # Should pass since 'noun' is not in repetition limits


class TestSpanCleaner:
    """Test SpanCleaner class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cleaner = SpanCleaner()
    
    def test_span_cleaner_initialization(self):
        """Test SpanCleaner initialization."""
        assert isinstance(self.cleaner.validator, SpanValidator)
        assert isinstance(self.cleaner.stats, dict)
        assert self.cleaner.stats['total_processed'] == 0
        assert self.cleaner.stats['valid_spans'] == 0
        assert self.cleaner.stats['invalid_spans'] == 0
    
    def test_clean_span_list_all_valid(self):
        """Test cleaning a list of all valid spans."""
        annotations = [
            {
                'raw': 'The cat sat.',
                'sequence_number': 1,
                'span_annotation': {
                    'text': 'The',
                    'xbar_class': 'determiner',
                    'start_pos': 0,
                    'end_pos': 3,
                    'length': 3
                }
            },
            {
                'raw': 'The cat sat.',
                'sequence_number': 1,
                'span_annotation': {
                    'text': 'cat',
                    'xbar_class': 'noun',
                    'start_pos': 4,
                    'end_pos': 7,
                    'length': 3
                }
            }
        ]
        
        cleaned = self.cleaner.clean_span_list(annotations)
        stats = self.cleaner.get_cleaning_stats()
        
        assert len(cleaned) == 2
        assert stats['total_processed'] == 2
        assert stats['valid_spans'] == 2
        assert stats['invalid_spans'] == 0
        assert stats['success_rate'] == 1.0
    
    def test_clean_span_list_mixed_validity(self):
        """Test cleaning a list with both valid and invalid spans."""
        annotations = [
            # Valid span
            {
                'raw': 'The cat sat.',
                'sequence_number': 1,
                'span_annotation': {
                    'text': 'The',
                    'xbar_class': 'determiner',
                    'start_pos': 0,
                    'end_pos': 3,
                    'length': 3
                }
            },
            # Invalid: "in" as determiner
            {
                'raw': 'He walked in the room.',
                'sequence_number': 2,
                'span_annotation': {
                    'text': 'in',
                    'xbar_class': 'determiner',
                    'start_pos': 10,
                    'end_pos': 12,
                    'length': 2
                }
            },
            # Invalid: embedded single char
            {
                'raw': 'The cat sat.',
                'sequence_number': 3,
                'span_annotation': {
                    'text': 'a',
                    'xbar_class': 'determiner',
                    'start_pos': 5,
                    'end_pos': 6,
                    'length': 1
                }
            }
        ]
        
        cleaned = self.cleaner.clean_span_list(annotations)
        stats = self.cleaner.get_cleaning_stats()
        
        assert len(cleaned) == 1  # Only valid span remains
        assert stats['total_processed'] == 3
        assert stats['valid_spans'] == 1
        assert stats['invalid_spans'] == 2
        assert stats['success_rate'] == 1/3
        
        # Check removal reasons
        assert 'word_class_mismatch' in stats['removed_by_rule']
        assert 'embedded_char' in stats['removed_by_rule']
        assert stats['removed_by_rule']['word_class_mismatch'] == 1
        assert stats['removed_by_rule']['embedded_char'] == 1
    
    def test_clean_span_list_repetition_limits(self):
        """Test cleaning with repetition limit enforcement."""
        # Create 7 "in" prepositions in one sequence (exceeds limit of 5)
        annotations = []
        for i in range(7):
            annotations.append({
                'raw': f'Text with in position {i}',
                'sequence_number': 1,
                'span_annotation': {
                    'text': 'in',
                    'xbar_class': 'preposition',
                    'start_pos': 10,
                    'end_pos': 12,
                    'length': 2
                }
            })
        
        cleaned = self.cleaner.clean_span_list(annotations)
        stats = self.cleaner.get_cleaning_stats()
        
        assert len(cleaned) == 5  # Only first 5 should remain
        assert stats['total_processed'] == 7
        assert stats['valid_spans'] == 5
        assert stats['invalid_spans'] == 2
        assert 'repetition_limit' in stats['removed_by_rule']
        assert stats['removed_by_rule']['repetition_limit'] == 2
    
    def test_get_cleaning_stats(self):
        """Test getting cleaning statistics."""
        # Process some spans to generate stats
        annotations = [
            {
                'raw': 'Valid span.',
                'sequence_number': 1,
                'span_annotation': {
                    'text': 'Valid',
                    'xbar_class': 'adjective',
                    'start_pos': 0,
                    'end_pos': 5,
                    'length': 5
                }
            }
        ]
        
        self.cleaner.clean_span_list(annotations)
        stats = self.cleaner.get_cleaning_stats()
        
        required_keys = ['total_processed', 'valid_spans', 'invalid_spans', 'success_rate', 'removed_by_rule']
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats['success_rate'], float)
        assert 0 <= stats['success_rate'] <= 1
        assert isinstance(stats['removed_by_rule'], dict)


class TestSpanValidatorIntegration:
    """Integration tests for span validation components."""
    
    def test_real_world_validation_scenarios(self):
        """Test validation with real-world problematic spans."""
        validator = SpanValidator()
        
        # Test cases based on actual issues found in validation report
        test_cases = [
            # The main "in" as determiner issue
            {
                'annotation': {
                    'raw': 'We walked in the park.',
                    'span_annotation': {
                        'text': 'in',
                        'xbar_class': 'determiner',
                        'start_pos': 10,
                        'end_pos': 12,
                        'length': 2
                    }
                },
                'expected_valid': False,
                'expected_reason_contains': 'incorrectly classified'
            },
            # Embedded 's' in word
            {
                'annotation': {
                    'raw': 'This works well.',
                    'span_annotation': {
                        'text': 's',
                        'xbar_class': 'noun',
                        'start_pos': 3,
                        'end_pos': 4,
                        'length': 1
                    }
                },
                'expected_valid': False,
                'expected_reason_contains': 'embedded in word'
            },
            # Valid punctuation
            {
                'annotation': {
                    'raw': 'Hello, world!',
                    'span_annotation': {
                        'text': ',',
                        'xbar_class': 'punctuation',
                        'start_pos': 5,
                        'end_pos': 6,
                        'length': 1
                    }
                },
                'expected_valid': True,
                'expected_reason_contains': 'Valid'
            },
            # Suspiciously long span
            {
                'annotation': {
                    'raw': 'A' * 200,
                    'span_annotation': {
                        'text': 'A' * 150,
                        'xbar_class': 'noun',
                        'start_pos': 0,
                        'end_pos': 150,
                        'length': 150
                    }
                },
                'expected_valid': False,
                'expected_reason_contains': 'Suspiciously long'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            is_valid, reason = validator.validate_span(test_case['annotation'])
            
            assert is_valid == test_case['expected_valid'], \
                f"Test case {i+1}: Expected valid={test_case['expected_valid']}, got {is_valid}. Reason: {reason}"
            
            assert test_case['expected_reason_contains'] in reason, \
                f"Test case {i+1}: Expected reason to contain '{test_case['expected_reason_contains']}', got: {reason}"
    
    def test_full_pipeline_cleaning_scenario(self):
        """Test a complete cleaning scenario similar to actual pipeline usage."""
        cleaner = SpanCleaner()
        
        # Create annotations similar to real pipeline output
        annotations = [
            # Valid spans
            {
                'raw': 'The quick brown fox jumps.',
                'sequence_number': 1,
                'span_annotation': {
                    'text': 'The',
                    'xbar_class': 'determiner',
                    'start_pos': 0,
                    'end_pos': 3,
                    'length': 3
                }
            },
            {
                'raw': 'The quick brown fox jumps.',
                'sequence_number': 1,
                'span_annotation': {
                    'text': 'fox',
                    'xbar_class': 'noun',
                    'start_pos': 16,
                    'end_pos': 19,
                    'length': 3
                }
            },
            # Invalid: "in" as determiner (our main target)
            {
                'raw': 'They walked in the park.',
                'sequence_number': 2,
                'span_annotation': {
                    'text': 'in',
                    'xbar_class': 'determiner',
                    'start_pos': 12,
                    'end_pos': 14,
                    'length': 2
                }
            },
            # Invalid: embedded character
            {
                'raw': 'The cats run.',
                'sequence_number': 3,
                'span_annotation': {
                    'text': 'a',
                    'xbar_class': 'determiner',
                    'start_pos': 5,
                    'end_pos': 6,
                    'length': 1
                }
            },
            # Invalid: standalone number
            {
                'raw': 'Version 3 is new.',
                'sequence_number': 4,
                'span_annotation': {
                    'text': '3',
                    'xbar_class': 'number',
                    'start_pos': 8,
                    'end_pos': 9,
                    'length': 1
                }
            }
        ]
        
        # Clean the annotations
        cleaned = cleaner.clean_span_list(annotations)
        stats = cleaner.get_cleaning_stats()
        
        # Verify results
        assert len(cleaned) == 2  # Only 2 valid spans should remain
        assert stats['total_processed'] == 5
        assert stats['valid_spans'] == 2
        assert stats['invalid_spans'] == 3
        assert stats['success_rate'] == 0.4
        
        # Verify specific removal reasons
        expected_rules = ['word_class_mismatch', 'embedded_char', 'invalid_standalone_char']
        for rule in expected_rules:
            assert rule in stats['removed_by_rule']
        
        # Verify cleaned spans are the expected ones
        cleaned_texts = [span['span_annotation']['text'] for span in cleaned]
        assert 'The' in cleaned_texts
        assert 'fox' in cleaned_texts
        assert 'in' not in cleaned_texts  # Should be removed
        assert 'a' not in cleaned_texts   # Should be removed
        assert '3' not in cleaned_texts   # Should be removed


if __name__ == "__main__":
    pytest.main([__file__])
