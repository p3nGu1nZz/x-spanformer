"""
X-Spanformer Vocabulary Induction Package

This package contains the core mathematical functions and utilities for
vocabulary induction following the hybrid Unigram-LM approach described
in Section 3.1 of the X-Spanformer paper.

Key modules:
- core: Core mathematical functions (Viterbi, perplexity calculations)
- em_algorithm: EM-based vocabulary induction with adaptive pruning
- validation: Vocabulary validation and consistency checks
"""

from .core import (
    viterbi_segment,
    compute_coverage,
    compute_baseline_perplexity,
    compute_pruning_perplexity_and_oov,
    compute_corpus_coverage,
    build_candidate_set,
    is_whitespace_coherent
)

from .em_algorithm import (
    initialize_probabilities,
    em_iteration,
    adaptive_pruning,
    induce_vocabulary
)

from .validation import (
    validate_vocabulary_completeness,
    validate_probabilities,
    validate_segmentation_consistency,
    validate_vocabulary_structure
)

__all__ = [
    # Core functions
    "viterbi_segment",
    "compute_coverage", 
    "compute_baseline_perplexity",
    "compute_pruning_perplexity_and_oov",
    "compute_corpus_coverage",
    "build_candidate_set",
    "is_whitespace_coherent",
    
    # EM Algorithm
    "initialize_probabilities",
    "em_iteration",
    "adaptive_pruning", 
    "induce_vocabulary",
    
    # Validation
    "validate_vocabulary_completeness",
    "validate_probabilities",
    "validate_segmentation_consistency",
    "validate_vocabulary_structure"
]
