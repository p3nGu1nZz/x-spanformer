"""
Core mathematical functions for vocabulary induction.

This module contains the fundamental mathematical operations for the
hybrid Unigram-LM vocabulary induction algorithm as described in
Section 3.1 of the X-Spanformer paper.
"""

import math
import logging
import string
from typing import List, Dict, Set, Tuple
from collections import Counter

# Module-level logger
logger = logging.getLogger(__name__)


def truncate_string(s: str, max_len: int) -> str:
    """Truncate string for display purposes."""
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + "..."


def viterbi_segment(x: str, V: List[str], p_u: Dict[str, float], case_handling: str = "normalize") -> List[str]:
    """
    Viterbi segmentation following the paper's formulation.
    
    Returns the best segmentation seg*(x) = argmax_seg ∏_{v∈seg} p(v)
    
    Args:
        x: Input string to segment
        V: Vocabulary list
        p_u: Probability dictionary for vocabulary pieces
        case_handling: Case handling strategy ("normalize" or "preserve")
        
    Returns:
        List of vocabulary pieces representing the best segmentation
        
    Raises:
        ValueError: If piece not found in probability dictionary or has invalid probability
    """
    # Apply case normalization if specified
    if case_handling == "normalize":
        x = x.lower()
    
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
    
    # Track progress through corpus segments
    progress_interval = max(1, len(corpus) // 20)  # Log approximately every 5% of total items
    for i, x in enumerate(corpus):
        if i % progress_interval == 0 or i == len(corpus) - 1:
            progress_pct = (i + 1) / len(corpus) * 100
            logger.info(f"    Progress: {i+1:,}/{len(corpus):,} segments ({progress_pct:.1f}%)")
        
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
    logger.info(f"Computing pruning perplexity and OOV rate for {len(corpus):,} segments")
    
    total_pieces = 0  # N_p'
    total_log_prob = 0.0  # L' (will be negative)
    total_codepoints = 0  # N_t
    uncovered_positions = 0  # N_uncov'
    
    # Track progress through corpus segments  
    progress_interval = max(1, len(corpus) // 20)  # Log every 5% of progress
    for i, x in enumerate(corpus):
        if i % progress_interval == 0 or i == len(corpus) - 1:
            progress_pct = (i + 1) / len(corpus) * 100
            logger.info(f"  Pruning perplexity progress: {i+1:,}/{len(corpus):,} segments ({progress_pct:.1f}%)")
            
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
    
    logger.info(f"Pruning metrics complete: perplexity={ppl:.2f}, OOV rate={oov_rate:.4f}")
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


def is_whitespace_coherent(candidate: str) -> bool:
    """
    Check if a candidate maintains whitespace as atomic units.
    
    Enforces strict whitespace separation using Python's standard string.whitespace
    definition, which includes: space, tab, newline, carriage return, vertical tab,
    and form feed. This follows standard tokenization conventions used by other LMs.
    
    Whitespace sequences must be standalone tokens, never mixed with non-whitespace
    characters. This treats whitespace like repeated character sequences - always atomic.
    
    Args:
        candidate: Candidate substring to validate
        
    Returns:
        True if candidate maintains atomic whitespace separation, False otherwise
    """
    if len(candidate) <= 1:
        return True
    
    # Use Python's standard whitespace definition (includes all 6 standard whitespace chars)
    # This matches conventions used by other language models
    whitespace_chars = set(string.whitespace)  # ' \t\n\r\x0b\x0c'
    
    # Check if this is a pure whitespace sequence (allowed)
    all_whitespace = all(c in whitespace_chars for c in candidate)
    if all_whitespace:
        return True
    
    # Check if this is a pure non-whitespace sequence (allowed) 
    no_whitespace = all(c not in whitespace_chars for c in candidate)
    if no_whitespace:
        return True
    
    # Mixed whitespace and non-whitespace is not allowed
    # This rejects cases like: " the", "ing ", "a\tb", "\nhello", etc.
    return False


def build_candidate_set(corpus: List[str], L_max: int, M: int, case_handling: str = "normalize") -> Tuple[List[str], Counter]:
    """
    Build the initial candidate set U_0 with atomic whitespace handling.
    
    U_0 = {top M substrings} ∪ {all single codepoints}
    
    Enforces atomic whitespace principle: whitespace sequences are treated
    as indivisible units, similar to repeated character sequences.
    
    Args:
        corpus: List of text segments
        L_max: Maximum substring length
        M: Number of top multi-character substrings to keep
        case_handling: Case handling strategy ("normalize" or "preserve")
        
    Returns:
        Tuple of (candidate_vocabulary, frequency_counter)
    """
    freq = Counter()
    
    # Apply case normalization if specified
    if case_handling == "normalize":
        corpus = [x.lower() for x in corpus]
    
    for x in corpus:
        T = len(x)
        # Count all substrings up to L_max
        for i in range(T):
            # Single characters (always valid)
            freq[x[i]] += 1
            
            # Multi-character substrings (check whitespace coherence)
            for length in range(2, min(L_max, T - i) + 1):
                candidate = x[i:i + length]
                
                # Only include candidates that maintain atomic whitespace
                if is_whitespace_coherent(candidate):
                    freq[candidate] += 1

    # Top M multi-character substrings (using raw frequencies - simpler approach)
    multi_char_candidates = [(u, f) for u, f in freq.items() if len(u) > 1]
    multi_char_candidates.sort(key=lambda x: x[1], reverse=True)
    multi_char_substrings = [u for u, _ in multi_char_candidates[:M]]
    
    # All single codepoints
    single_codepoints = list({u for u in freq if len(u) == 1})
    
    # Combine to form U_0
    U_0 = multi_char_substrings + single_codepoints
    
    return U_0, freq
