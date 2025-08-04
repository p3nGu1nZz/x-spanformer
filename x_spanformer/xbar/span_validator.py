"""
X-bar Span Validator

Core validation logic for spans based on X-bar theory and linguistic principles.
Provides reusable validation functionality for both pipeline integration and 
standalone validation scripts.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter


class SpanValidator:
    """Core span validator implementing X-bar theory and linguistic validation rules."""
    
    def __init__(self):
        """Initialize span validator with linguistic rules and thresholds."""
        
        # Enhanced word-class mappings to catch common misclassifications
        self.word_class_mappings = {
            # Common prepositions that are often mislabeled as determiners
            'in': ['preposition', 'adverb'],
            'on': ['preposition', 'adverb'],
            'at': ['preposition'],
            'by': ['preposition', 'adverb'],
            'for': ['preposition', 'conjunction'],
            'with': ['preposition'],
            'from': ['preposition'],
            'to': ['preposition', 'infinitive_marker'],
            'of': ['preposition'],
            'into': ['preposition'],
            'onto': ['preposition'],
            'through': ['preposition', 'adverb'],
            'over': ['preposition', 'adverb'],
            'under': ['preposition', 'adverb'],
            'about': ['preposition', 'adverb'],
            'after': ['preposition', 'adverb'],
            'before': ['preposition', 'adverb'],
            'during': ['preposition'],
            'within': ['preposition'],
            'without': ['preposition'],
            
            # Common determiners
            'the': ['determiner'],
            'a': ['determiner'],
            'an': ['determiner'],
            'this': ['determiner', 'pronoun'],
            'that': ['determiner', 'pronoun', 'conjunction'],
            'these': ['determiner', 'pronoun'],
            'those': ['determiner', 'pronoun'],
            'my': ['determiner'],
            'your': ['determiner'],
            'his': ['determiner'],
            'her': ['determiner'],
            'its': ['determiner'],
            'our': ['determiner'],
            'their': ['determiner'],
            
            # Common conjunctions
            'and': ['conjunction'],
            'or': ['conjunction'],
            'but': ['conjunction'],
            'yet': ['conjunction'],
            'so': ['conjunction', 'adverb'],
            'nor': ['conjunction'],
            
            # Common articles/pronouns
            'I': ['pronoun'],
            'you': ['pronoun'],
            'he': ['pronoun'],
            'she': ['pronoun'],
            'it': ['pronoun'],
            'we': ['pronoun'],
            'they': ['pronoun'],
        }
        
        # Valid single char classes (more restrictive)
        self.valid_single_char_classes = {
            'punctuation': {'.', ',', ';', ':', '!', '?', '"', "'", '(', ')', '[', ']', '{', '}'},
            'delimiter': {'(', ')', '[', ']', '{', '}', '|', '/', '\\', '<', '>', '-', '_'},
            'operator': {'+', '-', '*', '/', '=', '<', '>', '&', '|', '^', '~', '%'},
            'symbol': {'$', '#', '@', '%', '&', '*', '+', '=', '~', '^'},
        }
        
        # Repetition thresholds per sequence - prevent excessive repetition
        self.max_repetitions_per_sequence = {
            'preposition': 5,  # Max 5 "in" per sequence, not 13!
            'determiner': 8,
            'conjunction': 6,
            'pronoun': 10,
            'punctuation': 20,  # Punctuation can repeat more
        }
    
    def is_word_boundary(self, char: str) -> bool:
        """Check if character represents a word boundary."""
        return bool(re.match(r'[\s\.,;:!?\(\)\[\]{}"\'`\-_/\\|<>=+*&%$#@~^]', char))
    
    def validate_word_class_match(self, text: str, xbar_class: str) -> bool:
        """
        Check if word matches expected grammatical class.
        
        Args:
            text: The span text
            xbar_class: The assigned X-bar class
            
        Returns:
            True if the word-class assignment is valid
        """
        text_lower = text.lower()
        
        # Check our word-class mappings
        if text_lower in self.word_class_mappings:
            valid_classes = self.word_class_mappings[text_lower]
            if xbar_class not in valid_classes:
                return False
        
        return True
    
    def validate_single_char_span(self, span_info: Dict, raw_text: str) -> Tuple[bool, str]:
        """
        Enhanced validation for single-character spans.
        
        Args:
            span_info: Span annotation information
            raw_text: The original raw text
            
        Returns:
            Tuple of (is_valid, reason)
        """
        char = span_info['text']
        start_pos = span_info['start_pos']
        end_pos = span_info['end_pos']
        xbar_class = span_info['xbar_class']
        
        # Get context
        char_before = raw_text[start_pos - 1] if start_pos > 0 else '<START>'
        char_after = raw_text[end_pos] if end_pos < len(raw_text) else '<END>'
        
        # Check if it's surrounded by word boundaries
        before_is_boundary = (start_pos == 0) or self.is_word_boundary(char_before)
        after_is_boundary = (end_pos >= len(raw_text)) or self.is_word_boundary(char_after)
        is_standalone = before_is_boundary and after_is_boundary
        
        # Rule 1: Valid punctuation/symbols/operators
        if xbar_class in self.valid_single_char_classes:
            if char in self.valid_single_char_classes[xbar_class]:
                return True, f"Valid {xbar_class}"
            else:
                return False, f"'{char}' not valid for class {xbar_class}"
        
        # Rule 2: Characters embedded in words are almost always invalid
        if not is_standalone:
            return False, f"'{char}' embedded in word, not a valid span"
        
        # Rule 3: Standalone single characters - very restrictive
        if is_standalone:
            # Only allow very specific standalone single chars
            if char == 'a' and xbar_class == 'determiner':
                return True, f"Valid standalone determiner 'a'"
            elif char == 'I' and xbar_class == 'pronoun':
                return True, f"Valid standalone pronoun 'I'"
            elif char == '&' and xbar_class == 'conjunction':
                return True, f"Valid standalone conjunction '&'"
            else:
                return False, f"Standalone '{char}' as {xbar_class} not permitted"
        
        return False, f"'{char}' as {xbar_class} doesn't meet validation criteria"
    
    def validate_multi_char_span(self, span_info: Dict) -> Tuple[bool, str]:
        """
        Enhanced validation for multi-character spans.
        
        Args:
            span_info: Span annotation information
            
        Returns:
            Tuple of (is_valid, reason)
        """
        text = span_info['text']
        xbar_class = span_info['xbar_class']
        
        # Basic checks
        if text.strip() == '':
            return False, "Empty or whitespace-only span"
        
        if len(text) != span_info['length']:
            return False, f"Length mismatch: text='{text}' length={span_info['length']}"
        
        if len(text) > 100:
            return False, f"Suspiciously long span ({len(text)} chars)"
        
        # Check word-class match - this will catch "in" labeled as "determiner"
        if not self.validate_word_class_match(text, xbar_class):
            return False, f"Word '{text}' incorrectly classified as {xbar_class}"
        
        return True, "Multi-character span appears valid"
    
    def validate_span(self, annotation: Dict) -> Tuple[bool, str]:
        """
        Validate a single span annotation.
        
        Args:
            annotation: Full annotation record with 'span_annotation' and 'raw' fields
            
        Returns:
            Tuple of (is_valid, reason)
        """
        span_info = annotation['span_annotation']
        raw_text = annotation['raw']
        
        if span_info['length'] == 1:
            return self.validate_single_char_span(span_info, raw_text)
        else:
            return self.validate_multi_char_span(span_info)
    
    def check_repetition_limits(
        self, 
        text: str, 
        xbar_class: str, 
        sequence_word_counts: Dict[str, Counter]
    ) -> Tuple[bool, str]:
        """
        Check if span exceeds repetition limits for its class within a sequence.
        
        Args:
            text: The span text (lowercased)
            xbar_class: The X-bar class
            sequence_word_counts: Counter of word occurrences by class for current sequence
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if xbar_class in self.max_repetitions_per_sequence:
            max_allowed = self.max_repetitions_per_sequence[xbar_class]
            current_count = sequence_word_counts[xbar_class][text]
            
            if current_count > max_allowed:
                return False, f"Excessive repetition: '{text}' as {xbar_class} appears {current_count} times (max {max_allowed})"
        
        return True, "Within repetition limits"


class SpanCleaner:
    """
    Span cleaner that removes invalid spans from annotation datasets.
    Integrates with SpanValidator for comprehensive cleaning.
    """
    
    def __init__(self):
        """Initialize span cleaner with validator."""
        self.validator = SpanValidator()
        self.stats = {
            'total_processed': 0,
            'valid_spans': 0,
            'invalid_spans': 0,
            'removed_by_rule': defaultdict(int)
        }
    
    def clean_span_list(self, annotations: List[Dict]) -> List[Dict]:
        """
        Clean a list of annotations, removing invalid spans.
        
        Args:
            annotations: List of annotation records
            
        Returns:
            List of valid annotation records
        """
        cleaned_annotations = []
        sequence_word_counts = defaultdict(lambda: defaultdict(Counter))
        
        # Group by sequence for repetition checking
        sequence_groups = defaultdict(list)
        for annotation in annotations:
            sequence_id = annotation.get('sequence_id', 0)
            sequence_groups[sequence_id].append(annotation)
        
        # Process each sequence
        for sequence_id, seq_annotations in sequence_groups.items():
            seq_word_counts = defaultdict(Counter)
            
            for annotation in seq_annotations:
                self.stats['total_processed'] += 1
                
                # Basic span validation
                is_valid, reason = self.validator.validate_span(annotation)
                
                if is_valid:
                    # Check repetition limits
                    span_info = annotation['span_annotation']
                    text = span_info['text'].lower()
                    xbar_class = span_info['xbar_class']
                    
                    # Increment count
                    seq_word_counts[xbar_class][text] += 1
                    
                    # Check if this exceeds limits
                    rep_valid, rep_reason = self.validator.check_repetition_limits(
                        text, xbar_class, seq_word_counts
                    )
                    
                    if rep_valid:
                        cleaned_annotations.append(annotation)
                        self.stats['valid_spans'] += 1
                    else:
                        self.stats['invalid_spans'] += 1
                        self.stats['removed_by_rule']['repetition_limit'] += 1
                else:
                    self.stats['invalid_spans'] += 1
                    # Categorize removal reason
                    if 'embedded in word' in reason:
                        rule_type = 'embedded_char'
                    elif 'incorrectly classified' in reason:
                        rule_type = 'word_class_mismatch'
                    elif 'not valid for class' in reason:
                        rule_type = 'invalid_single_char'
                    elif 'not permitted' in reason:
                        rule_type = 'invalid_standalone_char'
                    else:
                        rule_type = 'other_validation'
                    
                    self.stats['removed_by_rule'][rule_type] += 1
        
        return cleaned_annotations
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get statistics from the cleaning process."""
        return {
            'total_processed': self.stats['total_processed'],
            'valid_spans': self.stats['valid_spans'],
            'invalid_spans': self.stats['invalid_spans'],
            'success_rate': self.stats['valid_spans'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
            'removed_by_rule': dict(self.stats['removed_by_rule'])
        }
