"""
EM Algorithm implementation for vocabulary induction.

This module implements the Expectation-Maximization algorithm with adaptive
pruning as described in Section 3.1 of the X-Spanformer paper.
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
from pathlib import Path
import json
import logging

from .core import (
    viterbi_segment, 
    compute_pruning_perplexity_and_oov,
    compute_baseline_perplexity,
    compute_corpus_coverage
)

# Module-level logger - will inherit from pipeline setup
logger = logging.getLogger(__name__)


def initialize_probabilities(V: List[str], freq: Counter) -> Dict[str, float]:
    """
    Initialize piece probabilities from frequency counts.
    
    Following paper: p^(0)(u) = freq(u) / Σ_v freq(v)
    
    Args:
        V: Vocabulary list
        freq: Frequency counter for vocabulary pieces
        
    Returns:
        Dictionary mapping pieces to probabilities
    """
    total_freq = sum(freq[u] for u in V)
    if total_freq == 0:
        raise ValueError("Total frequency is zero - cannot initialize probabilities")
    
    probs = {u: freq[u] / total_freq for u in V}
    
    # Log probability statistics
    non_zero_probs = [p for p in probs.values() if p > 0]
    if non_zero_probs:
        logger.debug(f"Prob range: {min(non_zero_probs):.6f} to {max(non_zero_probs):.6f}")
    
    return probs


def em_iteration(corpus: List[str], V: List[str], p_u: Dict[str, float]) -> Dict[str, float]:
    """
    Perform one EM iteration: E-step (Viterbi) + M-step (frequency update).
    
    Args:
        corpus: List of text segments
        V: Current vocabulary list
        p_u: Current probability dictionary
        
    Returns:
        Updated probability dictionary
    """
    # E-step: Compute γ^(t)(u|x) via Viterbi segmentation
    counts = Counter()
    
    # Track progress through corpus segments
    progress_interval = max(1, len(corpus) // 10)  # Log every 10% of progress
    for i, x in enumerate(corpus):
        if i % progress_interval == 0 or i == len(corpus) - 1:
            progress_pct = (i + 1) / len(corpus) * 100
            logger.info(f"    E-step progress: {i+1:,}/{len(corpus):,} segments ({progress_pct:.1f}%)")
        
        segmentation = viterbi_segment(x, V, p_u)
        for piece in segmentation:
            counts[piece] += 1

    # M-step: Update probabilities p^(t+1)(u) 
    total_counts = sum(counts.values())
    if total_counts > 0:
        updated_probs = {u: counts.get(u, 0) / total_counts for u in V}
        return updated_probs
    else:
        logger.warning("M-step: No counts generated, returning original probabilities")
        return p_u.copy()


def adaptive_pruning(
    corpus: List[str], 
    V: List[str], 
    p_next: Dict[str, float], 
    current_ppl: float,
    eps: float,
    tau_ppl: float,
    delta_oov: float
) -> Tuple[List[str], float]:
    """
    Perform adaptive pruning following the paper's criteria.
    
    Args:
        corpus: List of text segments
        V: Current vocabulary list
        p_next: Updated probabilities from M-step
        current_ppl: Current perplexity baseline
        eps: Minimum piece probability threshold
        tau_ppl: Perplexity increase threshold
        delta_oov: Maximum OOV rate threshold
        
    Returns:
        Tuple of (pruned_vocabulary, updated_perplexity)
    """
    V_pruned = V.copy()
    current_ppl_updated = current_ppl
    
    # First, automatically remove pieces with zero probability
    zero_prob_pieces = [u for u in V if p_next[u] <= 0]
    V_pruned = [u for u in V_pruned if p_next[u] > 0]
    
    if zero_prob_pieces:
        logger.debug(f"Removed {len(zero_prob_pieces):,} pieces with zero probability")
    
    # If we removed pieces, renormalize probabilities
    if len(V_pruned) < len(V):
        total_prob_remaining = sum(p_next[v] for v in V_pruned)
        if total_prob_remaining > 0:
            p_next = {v: p_next[v] / total_prob_remaining for v in V_pruned}
        else:
            # All pieces have zero probability - keep original vocabulary
            V_pruned = V.copy()
            logger.warning("All pieces have zero probability, keeping original vocabulary")
    
    # Consider removing pieces with p^(t+1)(u) < ε
    candidates_to_prune = [u for u in V_pruned if p_next[u] < eps]
    
    if not candidates_to_prune:
        logger.debug(f"No candidates below threshold {eps:.6f} for pruning")
        return V_pruned, current_ppl_updated
        
    logger.debug(f"Evaluating {len(candidates_to_prune):,} pruning candidates")
    
    # Precompute the set of all characters in the corpus
    corpus_chars = set(c for x in corpus for c in x)
    
    pruned_count = 0
    for u in candidates_to_prune:
        # Tentative removal: V' = V \ {u}
        V_prime = [v for v in V_pruned if v != u]
        
        if not V_prime:  # Don't remove all pieces
            continue
        
        # Ensure V_prime still contains all required single characters
        vocab_chars = set(v for v in V_prime if len(v) == 1)
        if not corpus_chars.issubset(vocab_chars):
            # Cannot remove this piece as it would make vocabulary incomplete
            continue
            
        # Create renormalized probabilities for reduced vocabulary V'
        # Following paper: probabilities must be renormalized after removal
        total_prob_remaining = sum(p_next[v] for v in V_prime)
        if total_prob_remaining == 0:
            continue
            
        p_prime = {v: p_next[v] / total_prob_remaining for v in V_prime}
        
        # Simulate removal and compute new metrics using pruning formula
        ppl_prime, oov_prime = compute_pruning_perplexity_and_oov(corpus, V_prime, p_prime)
        
        # Check pruning criteria from the paper
        ppl_increase = ppl_prime - current_ppl_updated
        if ppl_increase < tau_ppl and oov_prime <= delta_oov:
            V_pruned = V_prime
            current_ppl_updated = ppl_prime
            pruned_count += 1
    
    if pruned_count > 0:
        logger.debug(f"Pruned {pruned_count:,} additional pieces")
    
    return V_pruned, current_ppl_updated


def induce_vocabulary(
    corpus: List[str], 
    V: List[str], 
    freq: Counter, 
    hyperparams: Dict,
    output_dir: Optional[Path] = None
) -> Tuple[List[str], Dict[str, float], Dict]:
    """
    Complete EM-based vocabulary induction with adaptive pruning.
    
    Implements the complete algorithm from Section 3.1 including:
    - Proper initialization of piece probabilities
    - EM iterations with Viterbi E-step and frequency-based M-step  
    - Adaptive pruning with PPL and OOV thresholds
    
    Args:
        corpus: List of text segments
        V: Initial candidate vocabulary
        freq: Frequency counter for vocabulary pieces
        hyperparams: Dictionary with T_max_iters, min_piece_prob, delta_perplexity, delta_oov
        output_dir: Optional output directory for intermediate results
        
    Returns:
        Tuple of (final_vocabulary, final_probabilities, statistics)
    """
    logger.info("Starting EM-based vocabulary induction")
    logger.info(f"Corpus: {len(corpus):,} segments, Vocabulary: {len(V):,} candidates")
    
    # Store initial vocabulary size for statistics
    V_init = V.copy()
    
    # Extract hyperparameters
    T_max = hyperparams["T_max_iters"]
    eps = hyperparams["min_piece_prob"]
    tau_ppl = hyperparams["delta_perplexity"]
    delta_oov = hyperparams["delta_oov"]
    
    logger.info(f"Hyperparameters: T_max={T_max}, min_prob={eps}, delta_ppl={tau_ppl}, delta_oov={delta_oov}")

    # Initialize piece probabilities
    logger.info("Step 1: Initializing probabilities...")
    p_u = initialize_probabilities(V, freq)
    non_zero_initial = sum(1 for p in p_u.values() if p > 0)
    logger.info(f"  → {non_zero_initial:,}/{len(V):,} pieces initialized")

    # Compute baseline perplexity
    logger.info("Step 2: Computing baseline perplexity...")
    baseline_ppl = compute_baseline_perplexity(corpus, V, p_u)
    _, baseline_oov = compute_pruning_perplexity_and_oov(corpus, V, p_u)
    logger.info(f"  → Baseline: PPL={baseline_ppl:.2f}, OOV={baseline_oov:.4f}")

    current_ppl = baseline_ppl
    final_iteration = 0
    
    # EM iterations
    logger.info(f"Step 3: Starting EM iterations (max {T_max})...")
    for iteration in range(1, T_max + 1):
        logger.info(f"  Iteration {iteration}/{T_max}: {len(V):,} pieces")
        final_iteration = iteration
        
        # EM iteration
        p_next = em_iteration(corpus, V, p_u)
        
        # Adaptive pruning
        V, current_ppl = adaptive_pruning(
            corpus, V, p_next, current_ppl, eps, tau_ppl, delta_oov
        )
        logger.info(f"  → After pruning: {len(V):,} pieces, PPL={current_ppl:.2f}")

        # Update probabilities for next iteration
        # Only keep probabilities for remaining vocabulary pieces
        p_u = {u: p_next[u] for u in V if u in p_next}

    # Final vocabulary statistics
    logger.info("Computing final statistics...")
    final_coverage = compute_corpus_coverage(corpus, V, p_u)
    final_ppl, final_oov_rate = compute_pruning_perplexity_and_oov(corpus, V, p_u)
    
    stats = {
        "total_pieces": len(V),
        "baseline_ppl": baseline_ppl,
        "final_ppl": final_ppl, 
        "oov_rate": final_oov_rate,
        "coverage": final_coverage,
        "em_iterations": final_iteration,
        "pruned_pieces": len(V_init) - len(V),
        "baseline_oov": baseline_oov
    }
    
    logger.info(f"EM complete: {len(V_init):,} → {len(V):,} pieces ({final_iteration} iterations)")
    logger.info(f"Final metrics: PPL {baseline_ppl:.2f} → {final_ppl:.2f}, OOV {baseline_oov:.4f} → {final_oov_rate:.4f}")
    
    # Save intermediate results if output directory provided
    if output_dir:
        prune_dir = output_dir / "pruning"
        prune_dir.mkdir(exist_ok=True)
        
        with open(prune_dir / "final_probs.json", "w", encoding="utf-8") as f:
            json.dump({u: p_u[u] for u in V}, f, ensure_ascii=False, indent=2)
            
        with open(prune_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return V, p_u, stats
