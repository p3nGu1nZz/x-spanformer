"""
Core mathematical functions for vocabulary induction.

This module contains the fundamental mathematical operations for the
hybrid Unigram-LM vocabulary induction algorithm as described in
Section 3.1 of the X-Spanformer paper.
"""

import math
from typing import List, Dict, Set, Tuple
from collections import Counter


def truncate_string(s: str, max_len: int) -> str:
    """Truncate string for display purposes."""
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + "..."


def viterbi_segment(x: str, V: List[str], p_u: Dict[str, float]) -> List[str]:
    """
    Viterbi segmentation following the paper's formulation.
    
    Returns the best segmentation seg*(x) = argmax_seg ∏_{v∈seg} p(v)
    
    Args:
        x: Input string to segment
        V: Vocabulary list
        p_u: Probability dictionary for vocabulary pieces
        
    Returns:
        List of vocabulary pieces representing the best segmentation
        
    Raises:
        ValueError: If piece not found in probability dictionary or has invalid probability
    """
    T = len(x)
    dp = [-math.inf] * (T + 1)
    back = [None] * (T + 1)
    dp[0] = 0.0

    # Index vocabulary by first character for efficiency
    by_first = {}
    for u in V:
        if u:  # Skip empty strings
            by_first.setdefault(u[0], []).append(u)

    for i in range(T):
        if dp[i] == -math.inf:
            continue
        for u in by_first.get(x[i], []):
            j = i + len(u)
            if j <= T and x[i:j] == u:
                if u not in p_u:
                    raise ValueError(f"Piece '{u}' not found in probability dictionary")
                pu = p_u[u]
                if pu <= 0:
                    # Skip pieces with zero probability (will be pruned)
                    continue
                sc = dp[i] + math.log(pu)
                if sc > dp[j]:
                    dp[j], back[j] = sc, u

    # Reconstruct path - paper assumes all single codepoints are in vocabulary
    seg, ptr = [], T
    while ptr > 0:
        piece = back[ptr]
        if piece is not None:
            seg.append(piece)
            ptr -= len(piece)
        else:
            # This should never happen if vocabulary properly includes all single codepoints
            raise ValueError(f"Segmentation failed at position {ptr} in string '{truncate_string(x, 50)}' - vocabulary incomplete")
    return list(reversed(seg))


def compute_coverage(x: str, segmentation: List[str]) -> Set[int]:
    """
    Compute the set of codepoint indices covered by the segmentation.
    
    Returns cover_V(x) = {i | i is covered by some piece in segmentation}
    
    Args:
        x: Original input string
        segmentation: List of pieces from segmentation
        
    Returns:
        Set of covered codepoint indices
        
    Note: This validates that the segmentation properly reconstructs the input.
    """
    covered = set()
    pos = 0
    
    for piece in segmentation:
        if pos + len(piece) <= len(x) and x[pos:pos + len(piece)] == piece:
            covered.update(range(pos, pos + len(piece)))
            pos += len(piece)
        else:
            # This shouldn't happen with proper Viterbi segmentation
            break
    
    return covered


def compute_baseline_perplexity(corpus: List[str], V: List[str], p_u: Dict[str, float]) -> float:
    """
    Compute baseline perplexity following the paper's corrected formulation.
    
    PPL^(0) = exp(L^(0)/N_p^(0)) where L^(0) is negative log-likelihood, N_p^(0) is total pieces
    
    This uses piece-level normalization for consistency with pruning perplexity.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list
        p_u: Probability dictionary for vocabulary pieces
        
    Returns:
        Baseline perplexity value
    """
    total_pieces = 0  # N_p^(0)
    total_log_prob = 0.0  # L^(0) (will be negative)
    
    for x in corpus:
        segmentation = viterbi_segment(x, V, p_u)
        total_pieces += len(segmentation)
        
        for piece in segmentation:
            if piece not in p_u:
                raise ValueError(f"Piece '{piece}' not found in probability dictionary during baseline perplexity calculation")
            prob = p_u[piece]
            if prob <= 0:
                raise ValueError(f"Invalid probability {prob} for piece '{piece}' - must be positive")
            total_log_prob += math.log(prob)
    
    # Paper's corrected baseline perplexity formula (piece-level normalization)
    ppl = math.exp(-total_log_prob / total_pieces) if total_pieces > 0 else float('inf')
    return ppl


def compute_pruning_perplexity_and_oov(corpus: List[str], V: List[str], p_u: Dict[str, float]) -> Tuple[float, float]:
    """
    Compute pruning perplexity and OOV rate according to the paper's formulation.
    
    PPL' = exp(L'/N_p') where L' is negative log-likelihood, N_p' is total pieces
    OOV' = N_uncov' / N_t where N_uncov' is uncovered positions, N_t is total codepoints
    
    This is piece-level perplexity used during pruning decisions.
    
    Args:
        corpus: List of text segments  
        V: Vocabulary list
        p_u: Probability dictionary for vocabulary pieces
        
    Returns:
        Tuple of (perplexity, oov_rate)
    """
    total_pieces = 0  # N_p'
    total_log_prob = 0.0  # L' (will be negative)
    total_codepoints = 0  # N_t
    uncovered_positions = 0  # N_uncov'
    
    for x in corpus:
        segmentation = viterbi_segment(x, V, p_u)
        coverage = compute_coverage(x, segmentation)
        
        # Update counts
        total_pieces += len(segmentation)
        total_codepoints += len(x)
        uncovered_positions += len(x) - len(coverage)
        
        # Update log probability
        for piece in segmentation:
            if piece not in p_u:
                raise ValueError(f"Piece '{piece}' not found in probability dictionary during pruning perplexity calculation")
            prob = p_u[piece]
            if prob <= 0:
                raise ValueError(f"Invalid probability {prob} for piece '{piece}' - must be positive")
            total_log_prob += math.log(prob)
    
    # Compute metrics according to paper
    ppl = math.exp(-total_log_prob / total_pieces) if total_pieces > 0 else float('inf')
    oov_rate = uncovered_positions / total_codepoints if total_codepoints > 0 else 0.0
    
    return ppl, oov_rate


def compute_corpus_coverage(corpus: List[str], V: List[str], p_u: Dict[str, float]) -> float:
    """
    Compute corpus-level coverage as the percentage of codepoints covered.
    Uses actual vocabulary probabilities for segmentation as specified in paper.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list  
        p_u: Probability dictionary for vocabulary pieces
        
    Returns:
        Coverage percentage (0.0 to 1.0)
    """
    total_covered = 0
    total_positions = 0
    
    for x in corpus:
        segmentation = viterbi_segment(x, V, p_u)
        covered = compute_coverage(x, segmentation)
        total_covered += len(covered)
        total_positions += len(x)
    
    return total_covered / total_positions if total_positions > 0 else 0.0


def build_candidate_set(corpus: List[str], L_max: int, M: int) -> Tuple[List[str], Counter]:
    """
    Build the initial candidate set U_0 following the paper's formulation.
    
    U_0 = {top M substrings} ∪ {all single codepoints}
    
    Args:
        corpus: List of text segments
        L_max: Maximum substring length
        M: Number of top multi-character substrings to keep
        
    Returns:
        Tuple of (candidate_vocabulary, frequency_counter)
    """
    freq = Counter()
    
    for x in corpus:
        T = len(x)
        # Count all substrings up to L_max
        for i in range(T):
            # Single characters
            freq[x[i]] += 1
            # Multi-character substrings
            for length in range(2, min(L_max, T - i) + 1):
                freq[x[i:i + length]] += 1

    # Top M multi-character substrings
    multi_char_substrings = [u for u, _ in freq.most_common() if len(u) > 1][:M]
    
    # All single codepoints
    single_codepoints = list({u for u in freq if len(u) == 1})
    
    # Combine to form U_0
    U_0 = multi_char_substrings + single_codepoints
    
    return U_0, freq
