"""
Validation functions for vocabulary induction.

This module provides validation functions to ensure the vocabulary and
segmentation processes meet the requirements specified in the X-Spanformer paper.
"""

from typing import List, Dict, Set, Union
from .core import viterbi_segment, compute_coverage


def validate_vocabulary_completeness(corpus: List[str], V: List[str]) -> None:
    """
    Validate that vocabulary includes all single codepoints from corpus.
    
    According to the paper, U_0 must include all single codepoints to ensure
    complete coverage without fallbacks.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list
        
    Raises:
        ValueError: If vocabulary is missing required single codepoints
    """
    corpus_chars = set()
    for x in corpus:
        corpus_chars.update(x)
    
    vocab_chars = set(u for u in V if len(u) == 1)
    
    missing_chars = corpus_chars - vocab_chars
    if missing_chars:
        sorted_missing_chars = sorted(missing_chars)
        raise ValueError(f"Vocabulary missing required single codepoints: {sorted_missing_chars}")


def validate_probabilities(p_u: Dict[str, float]) -> None:
    """
    Validate that probabilities are well-formed.
    
    Args:
        p_u: Probability dictionary
        
    Raises:
        ValueError: If probabilities are invalid
    """
    if not p_u:
        raise ValueError("Probability dictionary is empty")
        
    for piece, prob in p_u.items():
        if not isinstance(prob, (int, float)):
            raise ValueError(f"Invalid probability type for piece '{piece}': {type(prob)}")
        # Allow zero probabilities during EM iterations (will be pruned later)
        if prob < 0:
            raise ValueError(f"Invalid probability {prob} for piece '{piece}' - must be non-negative")
        if prob > 1:
            raise ValueError(f"Invalid probability {prob} for piece '{piece}' - must be <= 1")
        # Check for NaN or infinity
        if not math.isfinite(prob):  # Check for NaN or infinity
            raise ValueError(f"Invalid probability {prob} for piece '{piece}' - must be a finite number")
    
    # Check if probabilities sum to approximately 1
    total_prob = sum(p_u.values())
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(f"Probabilities do not sum to 1.0: {total_prob}")


def validate_segmentation_consistency(
    corpus: List[str], 
    V: List[str], 
    p_u: Dict[str, float]
) -> Dict[str, int]:
    """
    Validate that segmentation is consistent across the corpus.
    
    Args:
        corpus: List of text segments
        V: Vocabulary list
        p_u: Probability dictionary
        
    Returns:
        Dictionary with validation statistics
        
    Raises:
        ValueError: If segmentation is inconsistent
    """
    stats = {
        "total_segments": len(corpus),
        "successful_segments": 0,
        "failed_segments": 0,
        "total_coverage": 0,
        "total_positions": 0
    }
    
    for i, x in enumerate(corpus):
        try:
            segmentation = viterbi_segment(x, V, p_u)
            coverage = compute_coverage(x, segmentation)
            
            # Check if segmentation reconstructs original string
            reconstructed = ''.join(segmentation)
            if reconstructed != x:
                raise ValueError(f"Segmentation failed to reconstruct string at index {i}: '{x}' != '{reconstructed}'")
                
            stats["successful_segments"] += 1
            stats["total_coverage"] += len(coverage)
            stats["total_positions"] += len(x)
            
        except Exception as e:
            stats["failed_segments"] += 1
            raise ValueError(f"Segmentation validation failed at corpus index {i}: {e}")
    
    return stats


def validate_vocabulary_structure(V: List[str]) -> Dict[str, Union[int, float]]:
    """
    Validate the structure of the vocabulary.
    
    Args:
        V: Vocabulary list
        
    Returns:
        Dictionary with vocabulary statistics
        
    Raises:
        ValueError: If vocabulary structure is invalid
    """
    if not V:
        raise ValueError("Vocabulary is empty")
    
    single_chars = set()
    multi_chars = []
    empty_pieces = []
    
    for piece in V:
        if not piece:
            empty_pieces.append(piece)
        elif len(piece) == 1:
            single_chars.add(piece)
        else:
            multi_chars.append(piece)
    
    if empty_pieces:
        raise ValueError(f"Vocabulary contains {len(empty_pieces)} empty pieces")
    
    # Check for duplicates
    if len(set(V)) != len(V):
        duplicates = [piece for piece in set(V) if V.count(piece) > 1]
        raise ValueError(f"Vocabulary contains duplicate pieces: {duplicates}")
    
    return {
        "total_pieces": len(V),
        "single_chars": len(single_chars),
        "multi_chars": len(multi_chars),
        "avg_length": sum(len(piece) for piece in V) / len(V)
    }
