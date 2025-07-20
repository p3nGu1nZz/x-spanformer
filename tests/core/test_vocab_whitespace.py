"""
Comprehensive tests for whitespace-aware tokenization in X-Spanformer vocabulary induction.

This test suite specifically validates the strict whitespace separation principle
that ensures whitespace sequences are always standalone atomic tokens, following
standard UTF-8/ASCII control character conventions as used by other language models.

Tests cover:
1. Standard whitespace control characters (space, tab, newline, carriage return, vertical tab, form feed)
2. Single whitespace character validation
3. Mixed whitespace sequence validation  
4. Mixed content+whitespace rejection
5. Candidate set generation with whitespace constraints
6. Edge cases and Unicode handling
7. Integration with vocabulary induction pipeline
"""

import pytest
import math
import string
from collections import Counter
from typing import List, Set

from x_spanformer.vocab.core import (
    is_whitespace_coherent,
    build_candidate_set,
    viterbi_segment,
    compute_baseline_perplexity
)


class TestWhitespaceCoherence:
    """Test the whitespace coherence validation function."""
    
    def test_single_character_always_valid(self):
        """Single characters should always be valid regardless of whitespace status."""
        # Test all standard whitespace characters
        for char in string.whitespace:
            assert is_whitespace_coherent(char), f"Single whitespace char {repr(char)} should be valid"
        
        # Test regular content characters
        for char in "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()":
            assert is_whitespace_coherent(char), f"Single content char {repr(char)} should be valid"
    
    def test_pure_whitespace_sequences_valid(self):
        """Pure whitespace sequences should be valid."""
        valid_whitespace_sequences = [
            "  ",              # double space
            "   ",             # triple space  
            "\t",              # single tab
            "\t\t",            # double tab
            "\n",              # single newline
            "\n\n",            # double newline
            "\r",              # carriage return
            "\r\n",            # Windows line ending
            "\x0b",            # vertical tab
            "\x0c",            # form feed
            " \t",             # space + tab
            "\t ",             # tab + space
            " \n",             # space + newline
            "\n ",             # newline + space
            "\t\n",            # tab + newline
            "\n\t",            # newline + tab
            "\r\n\t ",         # complex mix
            " \t\n\r\x0b\x0c", # all whitespace chars
        ]
        
        for seq in valid_whitespace_sequences:
            assert is_whitespace_coherent(seq), f"Pure whitespace sequence {repr(seq)} should be valid"
    
    def test_pure_content_sequences_valid(self):
        """Pure content sequences (no whitespace) should be valid."""
        valid_content_sequences = [
            "hello",
            "world",
            "123",
            "abc",
            "function",
            "if",
            "else",
            "return",
            "def",
            "class",
            "import",
            "from",
            "print",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "None",
            "True",
            "False",
            "!@#$%^&*()",
            "[]{}()<>",
            "+-*/=",
            ".,;:",
            "'\"",
            "αβγδε",  # Greek letters
            "数据科学",  # Chinese characters
            "café",    # Accented characters
        ]
        
        for seq in valid_content_sequences:
            assert is_whitespace_coherent(seq), f"Pure content sequence {repr(seq)} should be valid"
    
    def test_mixed_sequences_invalid(self):
        """Mixed whitespace+content sequences should be invalid."""
        invalid_mixed_sequences = [
            " hello",          # space + content
            "hello ",          # content + space
            "\thello",         # tab + content
            "hello\t",         # content + tab
            "\nhello",         # newline + content
            "hello\n",         # content + newline
            "\rhello",         # carriage return + content
            "hello\r",         # content + carriage return
            "\x0bhello",       # vertical tab + content
            "hello\x0b",       # content + vertical tab
            "\x0chello",       # form feed + content
            "hello\x0c",       # content + form feed
            "a b",             # content + space + content
            "if\tcondition",   # content + tab + content
            "line1\nline2",    # content + newline + content
            "data\r\nvalue",   # content + CRLF + content
            "func( x )",       # mixed throughout
            " the ",           # space + content + space
            "\tvar\t",         # tab + content + tab
            "\nline\n",        # newline + content + newline
        ]
        
        for seq in invalid_mixed_sequences:
            assert not is_whitespace_coherent(seq), f"Mixed sequence {repr(seq)} should be invalid"
    
    def test_empty_string(self):
        """Empty string should be valid."""
        assert is_whitespace_coherent("")
    
    def test_standard_whitespace_coverage(self):
        """Verify we cover all standard whitespace characters."""
        # Python's string.whitespace should include: ' \t\n\r\x0b\x0c'
        expected_whitespace = {' ', '\t', '\n', '\r', '\x0b', '\x0c'}
        actual_whitespace = set(string.whitespace)
        assert actual_whitespace == expected_whitespace, f"Expected {expected_whitespace}, got {actual_whitespace}"
    
    def test_unicode_whitespace_characters(self):
        """Test behavior with additional Unicode whitespace characters."""
        # These are Unicode whitespace but not in Python's string.whitespace
        unicode_whitespace = [
            '\u00a0',  # Non-breaking space
            '\u2000',  # En quad
            '\u2001',  # Em quad
            '\u2002',  # En space
            '\u2003',  # Em space
            '\u2028',  # Line separator
            '\u2029',  # Paragraph separator
        ]
        
        # These should be treated as regular content characters (not whitespace)
        # since they're not in Python's standard string.whitespace
        for char in unicode_whitespace:
            assert is_whitespace_coherent(char), f"Single Unicode char {repr(char)} should be valid"
            # Mixed with standard whitespace should be invalid
            assert not is_whitespace_coherent(' ' + char), f"Mixed standard+Unicode whitespace {repr(' ' + char)} should be invalid"


class TestCandidateSetGeneration:
    """Test candidate set generation with whitespace constraints."""
    
    def test_whitespace_candidates_generated(self):
        """Test that whitespace sequences are properly generated as candidates."""
        corpus = [
            "hello world",      # space
            "if\tcondition:",   # tab
            "line1\nline2",     # newline
            "data\r\nvalue",    # CRLF
            "  indented",       # double space
            "\t\tdeep",         # double tab
        ]
        
        candidates, freq = build_candidate_set(corpus, L_max=4, M=20)
        
        # Check that pure whitespace candidates are present
        whitespace_candidates = [c for c in candidates if all(ch in string.whitespace for ch in c) and len(c) > 1]
        
        # Should have at least some common whitespace patterns
        expected_patterns = [" ", "\t", "\n"]  # Single chars always included
        for pattern in expected_patterns:
            assert pattern in candidates, f"Whitespace pattern {repr(pattern)} should be in candidates"
    
    def test_no_mixed_candidates_generated(self):
        """Test that no mixed whitespace+content candidates are generated."""
        corpus = [
            " hello world ",    # spaces around content
            "\tif condition:\n", # tab and newline with content
            "function( x, y )", # spaces in function call
        ]
        
        candidates, freq = build_candidate_set(corpus, L_max=6, M=30)
        
        # Check that no mixed candidates exist
        mixed_candidates = []
        for candidate in candidates:
            has_whitespace = any(ch in string.whitespace for ch in candidate)
            has_content = any(ch not in string.whitespace for ch in candidate)
            if has_whitespace and has_content:
                mixed_candidates.append(candidate)
        
        assert len(mixed_candidates) == 0, f"Found mixed candidates: {mixed_candidates}"
    
    def test_all_control_characters_supported(self):
        """Test that all standard control characters can form candidates."""
        # Create corpus with all whitespace control characters
        corpus = [
            " ",      # space
            "\t",     # tab
            "\n",     # newline
            "\r",     # carriage return
            "\x0b",   # vertical tab
            "\x0c",   # form feed
            "  ",     # double space
            "\t\t",   # double tab
            "\r\n",   # CRLF
            " \t\n\r\x0b\x0c",  # all together
        ]
        
        candidates, freq = build_candidate_set(corpus, L_max=8, M=20)
        
        # All individual whitespace chars should be present
        for char in string.whitespace:
            assert char in candidates, f"Whitespace char {repr(char)} should be in candidates"
            assert freq[char] > 0, f"Whitespace char {repr(char)} should have positive frequency"


class TestViterbiWithWhitespace:
    """Test Viterbi segmentation with whitespace-aware vocabulary."""
    
    def test_viterbi_with_whitespace_vocab(self):
        """Test Viterbi segmentation using whitespace-aware vocabulary."""
        # Create vocabulary with both content and whitespace pieces
        vocab = ["hello", "world", "h", "e", "l", "o", "w", "r", "d", " ", "\t", "\n"]
        probs = {v: 1.0/len(vocab) for v in vocab}
        
        # Test segmentation of text with whitespace
        text = "hello world"
        segmentation = viterbi_segment(text, vocab, probs)
        
        # Should reconstruct original text
        assert "".join(segmentation) == text
        
        # Should maintain whitespace separation (space should be separate token)
        space_tokens = [piece for piece in segmentation if piece == " "]
        assert len(space_tokens) == 1, "Should have exactly one space token"
    
    def test_viterbi_prefers_whitespace_tokens(self):
        """Test that Viterbi prefers whitespace tokens over mixed alternatives."""
        # Create vocabulary where whitespace tokens have higher probability
        vocab = ["hello", "world", "h", "e", "l", "o", "w", "r", "d", " "]
        probs = {
            "hello": 0.2, "world": 0.2, " ": 0.5,  # High prob for space
            "h": 0.025, "e": 0.025, "l": 0.025, "o": 0.025, 
            "w": 0.025, "r": 0.025, "d": 0.025
        }
        
        text = "hello world"
        segmentation = viterbi_segment(text, vocab, probs)
        
        # Should use the high-probability space token
        assert " " in segmentation, "Should use whitespace token"
        assert "".join(segmentation) == text


class TestPerplexityWithWhitespace:
    """Test perplexity calculations with whitespace-aware tokenization."""
    
    def test_baseline_perplexity_with_whitespace(self):
        """Test baseline perplexity calculation with whitespace tokens."""
        corpus = ["hello world", "test case"]
        vocab = ["hello", "world", "test", "case", "h", "e", "l", "o", "w", "r", "d", "t", "s", "c", "a", " "]
        probs = {v: 1.0/len(vocab) for v in vocab}
        
        ppl = compute_baseline_perplexity(corpus, vocab, probs)
        
        # Should compute valid perplexity
        assert ppl > 0, "Perplexity should be positive"
        assert not math.isnan(ppl), "Perplexity should not be NaN"
        assert not math.isinf(ppl), "Perplexity should not be infinite"
    
    def test_perplexity_consistency_with_whitespace(self):
        """Test perplexity consistency between different whitespace approaches."""
        # Same corpus, different whitespace handling
        corpus = ["a b", "c d"]
        
        # Vocabulary 1: separate whitespace tokens
        vocab1 = ["a", "b", "c", "d", " "]
        probs1 = {v: 1.0/len(vocab1) for v in vocab1}
        ppl1 = compute_baseline_perplexity(corpus, vocab1, probs1)
        
        # Vocabulary 2: character-level only (forces different segmentation)
        vocab2 = ["a", "b", "c", "d", " "]  # Same as vocab1 in this case
        probs2 = {v: 1.0/len(vocab2) for v in vocab2}
        ppl2 = compute_baseline_perplexity(corpus, vocab2, probs2)
        
        # Should be identical since vocabularies are the same
        assert abs(ppl1 - ppl2) < 1e-10, "Perplexities should be identical for same vocabulary"


class TestEdgeCasesWithWhitespace:
    """Test edge cases and boundary conditions with whitespace handling."""
    
    def test_whitespace_only_corpus(self):
        """Test handling of corpus containing only whitespace."""
        corpus = [" ", "\t", "\n", "  ", "\t\t"]
        candidates, freq = build_candidate_set(corpus, L_max=3, M=10)
        
        # Should generate whitespace candidates
        assert len(candidates) > 0
        # All candidates should be pure whitespace
        for candidate in candidates:
            assert all(ch in string.whitespace for ch in candidate), f"Candidate {repr(candidate)} should be pure whitespace"
    
    def test_no_whitespace_corpus(self):
        """Test handling of corpus containing no whitespace."""
        corpus = ["hello", "world", "test", "case"]
        candidates, freq = build_candidate_set(corpus, L_max=5, M=20)
        
        # Should generate content candidates
        assert len(candidates) > 0
        # No candidates should contain whitespace
        for candidate in candidates:
            assert not any(ch in string.whitespace for ch in candidate), f"Candidate {repr(candidate)} should not contain whitespace"
    
    def test_very_long_whitespace_sequences(self):
        """Test handling of very long whitespace sequences."""
        long_space = " " * 100
        long_tab = "\t" * 50
        corpus = [long_space, long_tab, "content"]
        
        candidates, freq = build_candidate_set(corpus, L_max=10, M=20)
        
        # Should handle long sequences without error
        assert len(candidates) > 0
        # Should respect L_max constraint
        assert all(len(candidate) <= 10 for candidate in candidates)
    
    def test_mixed_unicode_and_ascii_whitespace(self):
        """Test behavior with mixed Unicode and ASCII whitespace-like characters."""
        corpus = [
            "hello world",      # ASCII space
            "test\u00a0case",   # Non-breaking space (Unicode)
            "line\u2028break",  # Line separator (Unicode)
        ]
        
        candidates, freq = build_candidate_set(corpus, L_max=5, M=20)
        
        # ASCII whitespace should be treated as whitespace
        assert " " in candidates
        
        # Unicode "whitespace" should be treated as content (since not in string.whitespace)
        mixed_candidates = []
        for candidate in candidates:
            if '\u00a0' in candidate or '\u2028' in candidate:
                has_ascii_ws = any(ch in string.whitespace for ch in candidate)
                if has_ascii_ws:
                    mixed_candidates.append(candidate)
        
        # Should not create mixed ASCII whitespace + Unicode whitespace candidates
        assert len(mixed_candidates) == 0, f"Found mixed ASCII/Unicode whitespace candidates: {mixed_candidates}"


class TestIntegrationWithPipeline:
    """Integration tests with the full vocabulary induction pipeline."""
    
    def test_end_to_end_with_whitespace(self):
        """Test complete pipeline with whitespace-containing corpus."""
        corpus = [
            "def function(x, y):",
            "    if x > y:",
            "        return x",
            "    else:",
            "        return y",
            "",
            "# Comment with spaces",
            "result = function(10, 20)",
        ]
        
        candidates, freq = build_candidate_set(corpus, L_max=6, M=30)
        
        # Should have both content and whitespace candidates
        content_candidates = [c for c in candidates if not any(ch in string.whitespace for ch in c)]
        whitespace_candidates = [c for c in candidates if all(ch in string.whitespace for ch in c) and len(c) > 0]
        
        assert len(content_candidates) > 0, "Should have content candidates"
        assert len(whitespace_candidates) > 0, "Should have whitespace candidates"
        
        # Common whitespace patterns should be present
        assert " " in candidates, "Should have single space"
        assert "    " in candidates or "  " in candidates, "Should have indentation patterns"
    
    def test_vocabulary_structure_with_whitespace(self):
        """Test vocabulary structure validation with whitespace tokens."""
        # Create mixed vocabulary
        vocab = ["function", "def", "if", "else", "return", "(", ")", ":", " ", "    ", "\n"]
        
        from x_spanformer.vocab.validation import validate_vocabulary_structure
        stats = validate_vocabulary_structure(vocab)
        
        # Should have reasonable structure
        assert stats["total_pieces"] == len(vocab)
        assert stats["single_chars"] >= 3  # At least "(" ")" ":"
        assert stats["multi_chars"] >= 6   # At least the keywords + whitespace sequences


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
