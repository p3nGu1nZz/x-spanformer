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
import asyncio
import concurrent.futures
import multiprocessing
import math
from functools import partial

from .core import (
    viterbi_segment, 
    compute_pruning_perplexity_and_oov,
    compute_baseline_perplexity,
    compute_corpus_coverage
)

# Module-level logger - will inherit from pipeline setup
logger = logging.getLogger(__name__)

# Constants for worker configuration
MAX_WORKERS_LIMIT = 32  # Maximum number of workers allowed, chosen to prevent excessive resource usage
WORKER_CPU_OFFSET = 4   # Additional workers to account for I/O-bound tasks

# Get optimal number of workers based on CPU cores
MAX_WORKERS = min(MAX_WORKERS_LIMIT, (multiprocessing.cpu_count() or 1) + WORKER_CPU_OFFSET)


def _process_segment_batch(segment_batch: List[str], V: List[str], p_u: Dict[str, float]) -> Counter:
    """
    Process a batch of segments for E-step (used by worker processes).
    
    Args:
        segment_batch: Batch of text segments to process
        V: Vocabulary list
        p_u: Probability dictionary
        
    Returns:
        Counter of piece counts for this batch
    """
    batch_counts = Counter()
    for segment in segment_batch:
        segmentation = viterbi_segment(segment, V, p_u)
        for piece in segmentation:
            batch_counts[piece] += 1
    return batch_counts


async def _process_segments_parallel(
    corpus: List[str], 
    V: List[str], 
    p_u: Dict[str, float], 
    batch_size: Optional[int] = None
) -> Counter:
    """
    Process corpus segments in parallel for E-step computation.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list  
        p_u: Probability dictionary
        batch_size: Size of batches to process (auto-calculated if None)
        
    Returns:
        Combined counter of all piece occurrences
    """
    if batch_size is None:
        # Calculate optimal batch size: aim for ~2x more batches than workers for better load balancing
        batch_size = max(1, len(corpus) // (MAX_WORKERS * 2))
    
    # Split corpus into batches
    batches = []
    for i in range(0, len(corpus), batch_size):
        batches.append(corpus[i:i + batch_size])
    
    logger.info(f"    Processing {len(batches)} batches across {MAX_WORKERS} workers")
    
    # Process batches in parallel using ProcessPoolExecutor
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create partial function with fixed V and p_u parameters
        process_func = partial(_process_segment_batch, V=V, p_u=p_u)
        
        # Submit all batch processing tasks at once for better throughput
        tasks = [
            loop.run_in_executor(executor, process_func, batch)
            for batch in batches
        ]
        
        # Process results as they complete, with progress tracking
        logger.info(f"    Submitting {len(tasks)} parallel tasks...")
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            
            # Log progress every 25% or when all tasks complete
            if completed % max(1, len(tasks) // 4) == 0 or completed == len(tasks):
                progress_pct = completed / len(tasks) * 100
                logger.info(f"        E-step batch progress: {completed}/{len(tasks)} batches ({progress_pct:.1f}%)")
        
        # Combine all results
        combined_counts = Counter()
        for batch_counts in results:
            combined_counts.update(batch_counts)
        
        logger.info(f"    E-step batch processing complete: {len(results)} batches processed")
    
    return combined_counts


def _compute_segment_perplexity_batch(
    segment_batch: List[str], 
    V: List[str], 
    p_u: Dict[str, float]
) -> Tuple[float, int, int]:
    """
    Compute perplexity metrics for a batch of segments (used by worker processes).
    
    Args:
        segment_batch: Batch of text segments
        V: Vocabulary list
        p_u: Probability dictionary
        
    Returns:
        Tuple of (total_log_prob, total_pieces, total_oov_positions)
    """
    total_log_prob = 0.0
    total_pieces = 0
    total_oov_positions = 0
    
    for segment in segment_batch:
        segmentation = viterbi_segment(segment, V, p_u)
        
        # Count pieces for this segment (piece-level normalization per paper)
        total_pieces += len(segmentation)
        
        # Compute log probability for this segment
        segment_log_prob = 0.0
        covered_positions = 0
        
        for piece in segmentation:
            if piece in p_u and p_u[piece] > 0:
                segment_log_prob += math.log(p_u[piece])
                covered_positions += len(piece)
        
        total_log_prob += segment_log_prob
        
        # Count OOV positions (uncovered codepoint positions)
        oov_positions = len(segment) - covered_positions
        total_oov_positions += oov_positions
    
    return total_log_prob, total_pieces, total_oov_positions


def _compute_coverage_batch(
    segment_batch: List[str], 
    V: List[str], 
    p_u: Dict[str, float]
) -> Tuple[int, int]:
    """
    Compute coverage metrics for a batch of segments (used by worker processes).
    
    Args:
        segment_batch: Batch of text segments
        V: Vocabulary list
        p_u: Probability dictionary
        
    Returns:
        Tuple of (total_covered_positions, total_positions)
    """
    from .core import viterbi_segment, compute_coverage
    
    total_covered = 0
    total_positions = 0
    
    for segment in segment_batch:
        segmentation = viterbi_segment(segment, V, p_u)
        covered = compute_coverage(segment, segmentation)
        total_covered += len(covered)
        total_positions += len(segment)
    
    return total_covered, total_positions


async def compute_corpus_coverage_parallel(
    corpus: List[str], 
    V: List[str], 
    p_u: Dict[str, float],
    batch_size: Optional[int] = None
) -> float:
    """
    Compute corpus-level coverage in parallel.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list
        p_u: Probability dictionary
        batch_size: Size of batches to process (auto-calculated if None)
        
    Returns:
        Coverage percentage (0.0 to 1.0)
    """
    if batch_size is None:
        # Calculate optimal batch size for coverage computation
        batch_size = max(1, len(corpus) // (MAX_WORKERS * 2))
    
    # Split corpus into batches
    batches = []
    for i in range(0, len(corpus), batch_size):
        batches.append(corpus[i:i + batch_size])
    
    logger.info(f"    Computing coverage across {len(batches)} batches using {MAX_WORKERS} workers")
    
    # Process batches in parallel
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create partial function with fixed parameters
        coverage_func = partial(_compute_coverage_batch, V=V, p_u=p_u)
        
        # Submit all batch processing tasks
        tasks = [
            loop.run_in_executor(executor, coverage_func, batch)
            for batch in batches
        ]
        
        # Process results as they complete, with progress tracking
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            
            # Log progress every 25% or when all tasks complete
            if completed % max(1, len(tasks) // 4) == 0 or completed == len(tasks):
                progress_pct = completed / len(tasks) * 100
                logger.info(f"    Coverage progress: {completed}/{len(tasks)} batches ({progress_pct:.1f}%)")
    
    # Combine results from all batches
    total_covered = sum(result[0] for result in results)
    total_positions = sum(result[1] for result in results)
    
    # Calculate coverage percentage
    coverage = total_covered / total_positions if total_positions > 0 else 0.0
    
    return coverage


def _compute_oov_batch(
    segment_batch: List[str], 
    V: List[str], 
    p_u: Dict[str, float]
) -> Tuple[int, int]:
    """
    Compute OOV metrics for a batch of segments (used by worker processes).
    
    Args:
        segment_batch: Batch of text segments
        V: Vocabulary list
        p_u: Probability dictionary
        
    Returns:
        Tuple of (total_positions, total_oov_positions)
    """
    total_positions = 0
    total_oov_positions = 0
    
    for segment in segment_batch:
        segmentation = viterbi_segment(segment, V, p_u)
        covered_positions = sum(len(piece) for piece in segmentation)
        oov_positions = len(segment) - covered_positions
        
        total_positions += len(segment)
        total_oov_positions += oov_positions
    
    return total_positions, total_oov_positions


async def compute_baseline_oov_parallel(
    corpus: List[str], 
    V: List[str], 
    p_u: Dict[str, float],
    batch_size: Optional[int] = None
) -> float:
    """
    Compute baseline OOV rate in parallel.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list
        p_u: Probability dictionary
        batch_size: Size of batches to process (auto-calculated if None)
        
    Returns:
        Baseline OOV rate
    """
    if batch_size is None:
        # Calculate optimal batch size for OOV computation
        batch_size = max(1, len(corpus) // (MAX_WORKERS * 2))
    
    # Split corpus into batches
    batches = []
    for i in range(0, len(corpus), batch_size):
        batches.append(corpus[i:i + batch_size])
    
    logger.info(f"    Computing baseline OOV across {len(batches)} batches using {MAX_WORKERS} workers")
    
    # Process batches in parallel
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create partial function with fixed parameters
        oov_func = partial(_compute_oov_batch, V=V, p_u=p_u)
        
        # Submit all batch processing tasks
        tasks = [
            loop.run_in_executor(executor, oov_func, batch)
            for batch in batches
        ]
        
        # Process results as they complete, with progress tracking
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            
            # Log progress every 25% or when all tasks complete
            if completed % max(1, len(tasks) // 4) == 0 or completed == len(tasks):
                progress_pct = completed / len(tasks) * 100
                logger.info(f"    Baseline OOV progress: {completed}/{len(tasks)} batches ({progress_pct:.1f}%)")
    
    # Combine results from all batches
    total_positions = sum(result[0] for result in results)
    total_oov_positions = sum(result[1] for result in results)
    
    # Calculate OOV rate
    baseline_oov = total_oov_positions / total_positions if total_positions > 0 else 0.0
    
    return baseline_oov


def _compute_baseline_batch(
    segment_batch: List[str], 
    V: List[str], 
    p_u: Dict[str, float]
) -> Tuple[int, float]:
    """
    Compute baseline perplexity metrics for a batch of segments (used by worker processes).
    
    Args:
        segment_batch: Batch of text segments
        V: Vocabulary list
        p_u: Probability dictionary
        
    Returns:
        Tuple of (total_pieces, total_log_prob)
    """
    total_pieces = 0
    total_log_prob = 0.0
    
    for segment in segment_batch:
        segmentation = viterbi_segment(segment, V, p_u)
        total_pieces += len(segmentation)
        
        for piece in segmentation:
            if piece not in p_u:
                raise ValueError(f"Piece '{piece}' not found in probability dictionary during baseline perplexity calculation")
            prob = p_u[piece]
            if prob <= 0:
                raise ValueError(f"Invalid probability {prob} for piece '{piece}' - must be positive")
            total_log_prob += math.log(prob)
    
    return total_pieces, total_log_prob


async def compute_baseline_perplexity_parallel(
    corpus: List[str], 
    V: List[str], 
    p_u: Dict[str, float],
    batch_size: Optional[int] = None
) -> float:
    """
    Compute baseline perplexity in parallel.
    
    PPL^(0) = exp(L^(0)/N_p^(0)) where L^(0) is negative log-likelihood, N_p^(0) is total pieces
    
    Args:
        corpus: List of text segments
        V: Vocabulary list
        p_u: Probability dictionary for vocabulary pieces
        batch_size: Size of batches to process (auto-calculated if None)
        
    Returns:
        Baseline perplexity value
    """
    if batch_size is None:
        # Calculate optimal batch size for baseline computation
        batch_size = max(1, len(corpus) // (MAX_WORKERS * 2))
    
    # Split corpus into batches
    batches = []
    for i in range(0, len(corpus), batch_size):
        batches.append(corpus[i:i + batch_size])
    
    logger.info(f"    Computing baseline across {len(batches)} batches using {MAX_WORKERS} workers")
    
    # Process batches in parallel
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create partial function with fixed parameters
        baseline_func = partial(_compute_baseline_batch, V=V, p_u=p_u)
        
        # Submit all batch processing tasks
        tasks = [
            loop.run_in_executor(executor, baseline_func, batch)
            for batch in batches
        ]
        
        # Process results as they complete, with progress tracking
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            
            # Log progress every 25% or when all tasks complete
            if completed % max(1, len(tasks) // 4) == 0 or completed == len(tasks):
                progress_pct = completed / len(tasks) * 100
                logger.info(f"    Baseline batch progress: {completed}/{len(tasks)} batches ({progress_pct:.1f}%)")
    
    # Combine results from all batches
    total_pieces = sum(result[0] for result in results)
    total_log_prob = sum(result[1] for result in results)
    
    # Paper's corrected baseline perplexity formula (piece-level normalization) with overflow protection
    if total_pieces > 0:
        avg_log_prob_per_piece = -total_log_prob / total_pieces
        
        # Log statistics for debugging
        logger.debug(f"Baseline calculation: total_pieces={total_pieces}, total_log_prob={total_log_prob:.2f}")
        logger.debug(f"Average log prob per piece: {avg_log_prob_per_piece:.6f}")
        
        # Safe exponentiation to avoid overflow errors
        max_exp_arg = OVERFLOW_PROTECTION_LIMIT  # More conservative limit to prevent overflow
        
        if avg_log_prob_per_piece > max_exp_arg:
            logger.warning(f"Baseline perplexity calculation would overflow (exp_arg={avg_log_prob_per_piece:.2f})")
            logger.info(f"This indicates very low probability vocabulary pieces - using simplified calculation")
            # Use a more conservative calculation for very high perplexity
            ppl = PERPLEXITY_CAP  # Cap at 1M instead of inf for better numerical stability
        elif avg_log_prob_per_piece == 0:
            ppl = 1.0  # exp(0) = 1
        else:
            try:
                ppl = math.exp(avg_log_prob_per_piece)
                # Additional check for overflow result
                if not math.isfinite(ppl) or ppl > 1e100:
                    logger.warning(f"Baseline perplexity result overflow detected: {ppl}")
                    ppl = 1000000.0
            except OverflowError:
                logger.warning(f"Math overflow in exp({avg_log_prob_per_piece:.2f})")
                ppl = 1000000.0
    else:
        ppl = float('inf')
        
    return ppl


async def _compute_perplexity_parallel(
    corpus: List[str], 
    V: List[str], 
    p_u: Dict[str, float],
    batch_size: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute perplexity and OOV rate in parallel.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list
        p_u: Probability dictionary  
        batch_size: Size of batches to process (auto-calculated if None)
        
    Returns:
        Tuple of (perplexity, oov_rate)
    """
    if batch_size is None:
        # Calculate optimal batch size for perplexity computation
        batch_size = max(1, len(corpus) // (MAX_WORKERS * 2))
    
    # Split corpus into batches
    batches = []
    for i in range(0, len(corpus), batch_size):
        batches.append(corpus[i:i + batch_size])
    
    # Process batches in parallel
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create partial function with fixed parameters
        compute_func = partial(_compute_segment_perplexity_batch, V=V, p_u=p_u)
        
        # Submit all batch processing tasks
        tasks = [
            loop.run_in_executor(executor, compute_func, batch)
            for batch in batches
        ]
        
        # Collect results
        results = await asyncio.gather(*tasks)
    
    # Combine results from all batches
    total_log_prob = sum(result[0] for result in results)
    total_pieces = sum(result[1] for result in results)
    total_oov_positions = sum(result[2] for result in results)
    
    # Calculate final metrics with overflow protection using piece-level normalization (consistent with paper and baseline)
    if total_pieces > 0:
        # Use piece-based normalization consistent with baseline perplexity calculation
        avg_log_prob_per_piece = -total_log_prob / total_pieces
        
        # Log statistics for debugging
        logger.debug(f"Final perplexity calculation: total_pieces={total_pieces}, total_log_prob={total_log_prob:.2f}")
        logger.debug(f"Average log prob per piece: {avg_log_prob_per_piece:.6f}")
        
        # Safe exponentiation to avoid overflow errors
        max_exp_arg = 500  # More conservative limit to prevent overflow
        
        if avg_log_prob_per_piece > max_exp_arg:
            logger.warning(f"Perplexity calculation would overflow (exp_arg={avg_log_prob_per_piece:.2f})")
            logger.info(f"This indicates very low probability vocabulary pieces - using simplified calculation")
            # Use a more conservative calculation for very high perplexity
            perplexity = 1000000.0  # Cap at 1M instead of inf for better numerical stability
        elif avg_log_prob_per_piece == 0:
            perplexity = 1.0  # exp(0) = 1
        else:
            try:
                perplexity = math.exp(avg_log_prob_per_piece)
                # Additional check for overflow result
                if not math.isfinite(perplexity) or perplexity > 1e100:
                    logger.warning(f"Perplexity result overflow detected: {perplexity}")
                    perplexity = 1000000.0
            except OverflowError:
                logger.warning(f"Math overflow in exp({avg_log_prob_per_piece:.2f})")
                perplexity = 1000000.0
    else:
        perplexity = float('inf')
    
    # Calculate OOV rate
    total_positions = sum(len(segment) for segment in corpus)
    oov_rate = total_oov_positions / total_positions if total_positions > 0 else 0.0
    
    return perplexity, oov_rate


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
    # E-step: Compute γ^(t)(u|x) via parallel Viterbi segmentation
    logger.info(f"    E-step: Processing {len(corpus):,} segments in parallel")
    
    # Run parallel segmentation
    counts = asyncio.run(_process_segments_parallel(corpus, V, p_u))
    
    logger.info(f"    E-step complete: {sum(counts.values()):,} total pieces counted")

    # M-step: Update probabilities p^(t+1)(u) 
    total_counts = sum(counts.values())
    if total_counts > 0:
        updated_probs = {u: counts.get(u, 0) / total_counts for u in V}
        
        # Log statistics about probability updates
        non_zero_updated = sum(1 for p in updated_probs.values() if p > 0)
        logger.info(f"    M-step: Updated {non_zero_updated:,}/{len(V):,} piece probabilities")
        
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
        ppl_prime, oov_prime = asyncio.run(_compute_perplexity_parallel(corpus, V_prime, p_prime))
        
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
    
    # Extract hyperparameters with proper type conversion
    T_max = int(hyperparams["T_max_iters"])
    eps = float(hyperparams["min_piece_prob"])
    tau_ppl = float(hyperparams["delta_perplexity"])
    delta_oov = float(hyperparams["delta_oov"])
    
    logger.info(f"Hyperparameters: T_max={T_max}, min_prob={eps}, delta_ppl={tau_ppl}, delta_oov={delta_oov}")

    # Initialize piece probabilities
    logger.info("Step 1: Initializing probabilities...")
    p_u = initialize_probabilities(V, freq)
    non_zero_initial = sum(1 for p in p_u.values() if p > 0)
    logger.info(f"  → {non_zero_initial:,}/{len(V):,} pieces initialized")

    # Compute baseline perplexity
    logger.info("Step 2: Computing baseline perplexity...")
    baseline_ppl = asyncio.run(compute_baseline_perplexity_parallel(corpus, V, p_u))
    
    # Compute baseline OOV rate using parallel processing
    logger.info("Step 2b: Computing baseline OOV rate...")
    baseline_oov = asyncio.run(compute_baseline_oov_parallel(corpus, V, p_u))
    
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

    # Final vocabulary statistics - run both computations concurrently
    logger.info("Computing final statistics...")
    logger.info("  Running parallel coverage and perplexity computations...")
    
    # Run both final computations in parallel using asyncio.run
    async def compute_final_stats():
        coverage_task = compute_corpus_coverage_parallel(corpus, V, p_u)
        perplexity_task = _compute_perplexity_parallel(corpus, V, p_u)
        return await asyncio.gather(coverage_task, perplexity_task)
    
    final_coverage, (final_ppl, final_oov_rate) = asyncio.run(compute_final_stats())
    
    logger.info("  Final statistics computation complete")
    
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
