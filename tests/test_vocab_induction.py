"""
Comprehensive unit tests for the X-Spanformer vocabulary induction pipeline.

This test suite covers all aspects of the vocabulary induction process:
1. Core mathematical functions (Viterbi, perplexity calculations)
2. EM algorithm implementation
3. Validation functions
4. Pipeline integration
5. Edge cases and error conditions

The tests follow the mathematical formulations from Section 3.1 of the
X-Spanformer paper and ensure strict adherence to the paper's requirements.
"""

import pytest
import math
import tempfile
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict

from x_spanformer.vocab import (
    # Core functions
    viterbi_segment,
    compute_coverage,
    compute_baseline_perplexity,
    compute_pruning_perplexity_and_oov,
    compute_corpus_coverage,
    build_candidate_set,
    
    # EM Algorithm
    initialize_probabilities,
    em_iteration,
    adaptive_pruning,
    induce_vocabulary,
    
    # Validation
    validate_vocabulary_completeness,
    validate_probabilities,
    validate_segmentation_consistency,
    validate_vocabulary_structure
)


class TestCoreFunctions:
    """Test core mathematical functions."""
    
    def test_viterbi_segment_basic(self):
        """Test basic Viterbi segmentation."""
        corpus = ["hello", "world"]
        vocab = ["h", "e", "l", "o", "w", "r", "d", "hello", "world"]
        probs = {v: 0.1 for v in vocab}
        
        # Should prefer longer sequences if probabilities are equal
        seg = viterbi_segment("hello", vocab, probs)
        assert "hello" in seg or seg == ["h", "e", "l", "l", "o"]
        assert "".join(seg) == "hello"
    
    def test_viterbi_segment_probability_preference(self):
        """Test that Viterbi prefers higher probability pieces."""
        vocab = ["h", "e", "l", "o", "hello"]
        probs = {
            "h": 0.01, "e": 0.01, "l": 0.01, "o": 0.01,
            "hello": 0.96  # Much higher probability
        }
        
        seg = viterbi_segment("hello", vocab, probs)
        assert seg == ["hello"]
    
    def test_viterbi_segment_missing_piece(self):
        """Test that Viterbi raises error for missing pieces."""
        vocab = ["h", "e", "l", "o"]
        probs = {"h": 0.25, "e": 0.25, "l": 0.25, "o": 0.25}
        
        with pytest.raises(ValueError, match="Segmentation failed"):
            viterbi_segment("hello world", vocab, probs)
    
    def test_viterbi_segment_invalid_probability(self):
        """Test that Viterbi handles zero probabilities gracefully."""
        vocab = ["h", "e", "l", "o"]
        probs = {"h": 0.25, "e": 0.25, "l": 0.25, "o": 0.0}  # Zero prob for 'o'
        
        # Should fail because 'o' has zero probability and can't be used for segmentation
        with pytest.raises(ValueError, match="Segmentation failed"):
            viterbi_segment("hello", vocab, probs)
    
    def test_viterbi_segment_incomplete_vocabulary(self):
        """Test that Viterbi raises error for incomplete vocabulary."""
        vocab = ["h", "e", "l"]  # Missing 'o'
        probs = {"h": 0.33, "e": 0.33, "l": 0.34}
        
        with pytest.raises(ValueError, match="Segmentation failed"):
            viterbi_segment("hello", vocab, probs)
    
    def test_compute_coverage_complete(self):
        """Test coverage computation with complete coverage."""
        x = "hello"
        segmentation = ["h", "e", "l", "l", "o"]
        coverage = compute_coverage(x, segmentation)
        assert coverage == {0, 1, 2, 3, 4}  # All positions covered
    
    def test_compute_coverage_partial(self):
        """Test coverage computation with partial coverage."""
        x = "hello"
        segmentation = ["h", "ell"]  # Missing 'o'
        coverage = compute_coverage(x, segmentation)
        assert coverage == {0, 1, 2, 3}  # Position 4 not covered
    
    def test_compute_coverage_empty(self):
        """Test coverage computation with empty segmentation."""
        x = "hello"
        segmentation = []
        coverage = compute_coverage(x, segmentation)
        assert coverage == set()
    
    def test_compute_baseline_perplexity_uniform(self):
        """Test baseline perplexity calculation with uniform probabilities."""
        corpus = ["aa", "bb"]
        vocab = ["a", "b"]
        probs = {"a": 0.5, "b": 0.5}
        
        ppl = compute_baseline_perplexity(corpus, vocab, probs)
        expected_ppl = math.exp(-math.log(0.5))  # Each piece has prob 0.5
        assert abs(ppl - expected_ppl) < 1e-10
    
    def test_compute_baseline_perplexity_nonuniform(self):
        """Test baseline perplexity with non-uniform probabilities."""
        corpus = ["ab"]
        vocab = ["a", "b"]
        probs = {"a": 0.8, "b": 0.2}
        
        ppl = compute_baseline_perplexity(corpus, vocab, probs)
        expected_log_prob = math.log(0.8) + math.log(0.2)
        expected_ppl = math.exp(-expected_log_prob / 2)  # 2 pieces
        assert abs(ppl - expected_ppl) < 1e-10
    
    def test_compute_pruning_perplexity_and_oov(self):
        """Test pruning perplexity and OOV calculation."""
        corpus = ["hello"]
        vocab = ["h", "e", "l", "o"]  # All chars present, no OOV expected
        probs = {v: 0.25 for v in vocab}
        
        ppl, oov = compute_pruning_perplexity_and_oov(corpus, vocab, probs)
        
        # Should have zero OOV with complete vocabulary 
        assert oov == 0.0
        assert ppl > 0
    
    def test_compute_corpus_coverage(self):
        """Test corpus-level coverage calculation."""
        corpus = ["hello", "world"]
        vocab = ["h", "e", "l", "o", "w", "r", "d"]
        probs = {v: 1.0/len(vocab) for v in vocab}
        
        coverage = compute_corpus_coverage(corpus, vocab, probs)
        assert 0 <= coverage <= 1
    
    def test_build_candidate_set_basic(self):
        """Test basic candidate set building."""
        corpus = ["hello", "world"]
        L_max = 3
        M = 5
        
        U_0, freq = build_candidate_set(corpus, L_max, M)
        
        # Should include all single characters
        single_chars = {c for text in corpus for c in text}
        vocab_single_chars = {u for u in U_0 if len(u) == 1}
        assert single_chars.issubset(vocab_single_chars)
        
        # Should have reasonable frequency counts
        assert all(freq[u] > 0 for u in U_0)
    
    def test_build_candidate_set_length_limit(self):
        """Test that candidate set respects length limits."""
        corpus = ["hello"]
        L_max = 2
        M = 10
        
        U_0, freq = build_candidate_set(corpus, L_max, M)
        
        # No piece should be longer than L_max
        assert all(len(u) <= L_max for u in U_0)


class TestEMAlgorithm:
    """Test EM algorithm implementation."""
    
    def test_initialize_probabilities_uniform(self):
        """Test probability initialization with uniform frequencies."""
        vocab = ["a", "b", "c"]
        freq = Counter({"a": 10, "b": 10, "c": 10})
        
        probs = initialize_probabilities(vocab, freq)
        
        assert all(abs(p - 1/3) < 1e-10 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-10
    
    def test_initialize_probabilities_nonuniform(self):
        """Test probability initialization with non-uniform frequencies."""
        vocab = ["a", "b"]
        freq = Counter({"a": 30, "b": 10})
        
        probs = initialize_probabilities(vocab, freq)
        
        assert abs(probs["a"] - 0.75) < 1e-10
        assert abs(probs["b"] - 0.25) < 1e-10
        assert abs(sum(probs.values()) - 1.0) < 1e-10
    
    def test_initialize_probabilities_zero_frequency(self):
        """Test that zero total frequency raises error."""
        vocab = ["a", "b"]
        freq = Counter({"a": 0, "b": 0})
        
        with pytest.raises(ValueError, match="Total frequency is zero"):
            initialize_probabilities(vocab, freq)
    
    def test_em_iteration_basic(self):
        """Test basic EM iteration."""
        corpus = ["aa", "bb"]
        vocab = ["a", "b"]
        probs = {"a": 0.5, "b": 0.5}
        
        new_probs = em_iteration(corpus, vocab, probs)
        
        # Should maintain probability distribution
        assert abs(sum(new_probs.values()) - 1.0) < 1e-10
        # Should have equal probabilities since each appears twice
        assert abs(new_probs["a"] - 0.5) < 1e-10
        assert abs(new_probs["b"] - 0.5) < 1e-10
    
    def test_adaptive_pruning_basic(self):
        """Test basic adaptive pruning."""
        corpus = ["hello"]
        vocab = ["h", "e", "l", "o", "hello"]  # Include longer piece
        probs = {"h": 0.2, "e": 0.2, "l": 0.2, "o": 0.2, "hello": 0.2}
        current_ppl = 2.0
        
        V_pruned, ppl_updated = adaptive_pruning(
            corpus, vocab, probs, current_ppl,
            eps=0.15,  # Should prune pieces below threshold
            tau_ppl=10.0,  # Lenient perplexity threshold
            delta_oov=1.0   # Lenient OOV threshold
        )
        
        # Should have maintained all single characters but may prune others
        corpus_chars = set("hello")
        vocab_chars = set(v for v in V_pruned if len(v) == 1)
        assert corpus_chars.issubset(vocab_chars)
    
    def test_adaptive_pruning_no_pruning(self):
        """Test adaptive pruning when no pieces should be pruned."""
        corpus = ["hello"]
        vocab = ["h", "e", "l", "o"]
        probs = {"h": 0.25, "e": 0.25, "l": 0.25, "o": 0.25}
        current_ppl = 2.0
        
        V_pruned, ppl_updated = adaptive_pruning(
            corpus, vocab, probs, current_ppl,
            eps=0.1,  # All probs above threshold
            tau_ppl=0.01,  # Very strict PPL threshold - should prevent pruning
            delta_oov=0.01  # Very strict OOV threshold - should prevent pruning
        )
        
        # Should not have pruned any pieces due to strict thresholds
        assert len(V_pruned) == len(vocab)
    
    def test_induce_vocabulary_basic(self):
        """Test basic vocabulary induction."""
        corpus = ["hello", "world"]
        vocab = ["h", "e", "l", "o", "w", "r", "d"]
        freq = Counter({v: 1 for v in vocab})
        hyperparams = {
            "T_max_iters": 2,
            "min_piece_prob": 0.01,
            "delta_perplexity": 1.0,
            "delta_oov": 0.1
        }
        
        V_final, p_final, stats = induce_vocabulary(corpus, vocab, freq, hyperparams)
        
        # Should return valid vocabulary and probabilities
        assert len(V_final) > 0
        assert len(p_final) == len(V_final)
        assert abs(sum(p_final.values()) - 1.0) < 1e-6
        
        # Should have valid statistics
        assert stats["total_pieces"] == len(V_final)
        assert stats["baseline_ppl"] > 0
        assert stats["final_ppl"] > 0
        assert 0 <= stats["oov_rate"] <= 1
        assert 0 <= stats["coverage"] <= 1
    
    def test_induce_vocabulary_with_output_dir(self):
        """Test vocabulary induction with output directory."""
        corpus = ["hello"]
        vocab = ["h", "e", "l", "o"]
        freq = Counter({v: 1 for v in vocab})
        hyperparams = {
            "T_max_iters": 1,
            "min_piece_prob": 0.01,
            "delta_perplexity": 1.0,
            "delta_oov": 0.1
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            V_final, p_final, stats = induce_vocabulary(
                corpus, vocab, freq, hyperparams, output_dir
            )
            
            # Should create output files
            assert (output_dir / "pruning" / "final_probs.json").exists()
            assert (output_dir / "pruning" / "stats.json").exists()
            
            # Verify file contents
            with open(output_dir / "pruning" / "final_probs.json") as f:
                saved_probs = json.load(f)
                assert len(saved_probs) == len(p_final)


class TestValidation:
    """Test validation functions."""
    
    def test_validate_vocabulary_completeness_valid(self):
        """Test vocabulary completeness validation with valid vocabulary."""
        corpus = ["hello", "world"]
        vocab = ["h", "e", "l", "o", "w", "r", "d"]
        
        # Should not raise any exception
        validate_vocabulary_completeness(corpus, vocab)
    
    def test_validate_vocabulary_completeness_missing_chars(self):
        """Test vocabulary completeness validation with missing characters."""
        corpus = ["hello", "world"]
        vocab = ["h", "e", "l", "o"]  # Missing 'w', 'r', 'd'
        
        with pytest.raises(ValueError, match="missing required single codepoints"):
            validate_vocabulary_completeness(corpus, vocab)
    
    def test_validate_probabilities_valid(self):
        """Test probability validation with valid probabilities."""
        probs = {"a": 0.3, "b": 0.7}
        
        # Should not raise any exception
        validate_probabilities(probs)
    
    def test_validate_probabilities_empty(self):
        """Test probability validation with empty dictionary."""
        probs = {}
        
        with pytest.raises(ValueError, match="empty"):
            validate_probabilities(probs)
    
    def test_validate_probabilities_invalid_type(self):
        """Test probability validation with invalid type."""
        # Test with NaN float value which should fail
        probs = {"a": 0.5}
        probs["b"] = float('nan')  # Invalid float value
        
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_probabilities(probs)
    
    def test_validate_probabilities_zero_probability(self):
        """Test probability validation with zero probability."""
        probs = {"a": 0.0, "b": 1.0}
        
        # Should not raise error for zero probabilities (they get pruned)
        validate_probabilities(probs)
    
    def test_validate_probabilities_greater_than_one(self):
        """Test probability validation with probability > 1."""
        probs = {"a": 1.5, "b": 0.5}
        
        with pytest.raises(ValueError, match="must be <= 1"):
            validate_probabilities(probs)
    
    def test_validate_probabilities_not_normalized(self):
        """Test probability validation with non-normalized probabilities."""
        probs = {"a": 0.6, "b": 0.6}  # Sum = 1.2
        
        with pytest.raises(ValueError, match="do not sum to 1.0"):
            validate_probabilities(probs)
    
    def test_validate_segmentation_consistency_valid(self):
        """Test segmentation consistency validation with valid segmentation."""
        corpus = ["hello", "world"]
        vocab = ["h", "e", "l", "o", "w", "r", "d"]
        probs = {v: 1.0/len(vocab) for v in vocab}
        
        stats = validate_segmentation_consistency(corpus, vocab, probs)
        
        assert stats["total_segments"] == 2
        assert stats["successful_segments"] == 2
        assert stats["failed_segments"] == 0
    
    def test_validate_segmentation_consistency_invalid(self):
        """Test segmentation consistency validation with invalid segmentation."""
        corpus = ["hello"]
        vocab = ["h", "e", "l"]  # Missing 'o'
        probs = {"h": 0.33, "e": 0.33, "l": 0.34}
        
        with pytest.raises(ValueError, match="Segmentation validation failed"):
            validate_segmentation_consistency(corpus, vocab, probs)
    
    def test_validate_vocabulary_structure_valid(self):
        """Test vocabulary structure validation with valid vocabulary."""
        vocab = ["h", "e", "l", "o", "hello"]
        
        stats = validate_vocabulary_structure(vocab)
        
        assert stats["total_pieces"] == 5
        assert stats["single_chars"] == 4
        assert stats["multi_chars"] == 1
        assert stats["avg_length"] == (1+1+1+1+5)/5
    
    def test_validate_vocabulary_structure_empty(self):
        """Test vocabulary structure validation with empty vocabulary."""
        vocab = []
        
        with pytest.raises(ValueError, match="empty"):
            validate_vocabulary_structure(vocab)
    
    def test_validate_vocabulary_structure_empty_pieces(self):
        """Test vocabulary structure validation with empty pieces."""
        vocab = ["h", "", "e"]
        
        with pytest.raises(ValueError, match="empty pieces"):
            validate_vocabulary_structure(vocab)
    
    def test_validate_vocabulary_structure_duplicates(self):
        """Test vocabulary structure validation with duplicate pieces."""
        vocab = ["h", "e", "h"]
        
        with pytest.raises(ValueError, match="duplicate pieces"):
            validate_vocabulary_structure(vocab)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_character_corpus(self):
        """Test with single character corpus."""
        corpus = ["a"]
        vocab = ["a"]
        probs = {"a": 1.0}
        
        seg = viterbi_segment("a", vocab, probs)
        assert seg == ["a"]
        
        ppl = compute_baseline_perplexity(corpus, vocab, probs)
        assert ppl == 1.0  # exp(-log(1.0)) = 1.0
    
    def test_empty_string_corpus(self):
        """Test with empty string in corpus."""
        corpus = [""]
        vocab = ["a"]
        probs = {"a": 1.0}
        
        seg = viterbi_segment("", vocab, probs)
        assert seg == []
        
        # Coverage should be empty
        coverage = compute_coverage("", seg)
        assert coverage == set()
    
    def test_very_long_strings(self):
        """Test with very long strings."""
        corpus = ["a" * 1000]
        vocab = ["a"]
        probs = {"a": 1.0}
        
        seg = viterbi_segment("a" * 1000, vocab, probs)
        assert len(seg) == 1000
        assert all(piece == "a" for piece in seg)
    
    def test_unicode_characters(self):
        """Test with Unicode characters."""
        corpus = ["héllo", "wörld"]
        # Include all Unicode characters
        vocab = ["h", "é", "l", "o", "w", "ö", "r", "d"]
        probs = {v: 1.0/len(vocab) for v in vocab}
        
        seg1 = viterbi_segment("héllo", vocab, probs)
        seg2 = viterbi_segment("wörld", vocab, probs)
        
        assert "".join(seg1) == "héllo"
        assert "".join(seg2) == "wörld"
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency between different perplexity calculations."""
        corpus = ["hello", "world"]
        vocab = ["h", "e", "l", "o", "w", "r", "d"]
        probs = {v: 1.0/len(vocab) for v in vocab}
        
        baseline_ppl = compute_baseline_perplexity(corpus, vocab, probs)
        pruning_ppl, oov = compute_pruning_perplexity_and_oov(corpus, vocab, probs)
        
        # Should be mathematically consistent (same normalization)
        assert abs(baseline_ppl - pruning_ppl) < 1e-10
        
        # OOV should be zero with complete vocabulary
        assert oov == 0.0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline_small(self):
        """Test complete pipeline with small corpus."""
        corpus = ["hello", "world", "hello world"]
        L_max = 5
        M = 10
        
        # Build candidate set
        U_0, freq = build_candidate_set(corpus, L_max, M)
        
        # Validate vocabulary
        validate_vocabulary_completeness(corpus, U_0)
        
        # Induce vocabulary
        hyperparams = {
            "T_max_iters": 3,
            "min_piece_prob": 0.001,
            "delta_perplexity": 2.0,
            "delta_oov": 0.2
        }
        
        V_final, p_final, stats = induce_vocabulary(corpus, U_0, freq, hyperparams)
        
        # Validate results
        validate_probabilities(p_final)
        validate_segmentation_consistency(corpus, V_final, p_final)
        
        # Check statistics
        assert stats["total_pieces"] <= len(U_0)  # Should have pruned some pieces
        assert stats["baseline_ppl"] > 0
        assert stats["final_ppl"] > 0
        assert 0 <= stats["oov_rate"] <= 1
        assert 0 <= stats["coverage"] <= 1
    
    def test_pipeline_convergence(self):
        """Test that pipeline converges to stable solution."""
        corpus = ["abc", "def", "ghi"]
        vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        freq = Counter({v: 1 for v in vocab})
        
        hyperparams = {
            "T_max_iters": 10,
            "min_piece_prob": 0.001,
            "delta_perplexity": 0.5,
            "delta_oov": 0.1
        }
        
        V_final, p_final, stats = induce_vocabulary(corpus, vocab, freq, hyperparams)
        
        # Should converge to stable solution
        assert len(V_final) > 0
        assert stats["em_iterations"] <= 10
        
        # Run again to check consistency
        V_final2, p_final2, stats2 = induce_vocabulary(corpus, vocab, freq, hyperparams)
        
        # Should get same result (deterministic)
        assert V_final == V_final2
        assert stats["final_ppl"] == stats2["final_ppl"]


# Parametrized tests for different corpus sizes and configurations
@pytest.mark.parametrize("corpus_size", [1, 10, 100])
@pytest.mark.parametrize("L_max", [2, 5, 10])
@pytest.mark.parametrize("M", [5, 20, 50])
def test_scalability(corpus_size, L_max, M):
    """Test scalability with different corpus sizes and parameters."""
    # Generate simple corpus
    corpus = [f"text{i}" for i in range(corpus_size)]
    
    U_0, freq = build_candidate_set(corpus, L_max, M)
    
    # Should handle different scales
    assert len(U_0) > 0
    assert len(freq) > 0
    
    # Should include all single characters
    all_chars = set(c for text in corpus for c in text)
    vocab_chars = set(u for u in U_0 if len(u) == 1)
    assert all_chars.issubset(vocab_chars)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
