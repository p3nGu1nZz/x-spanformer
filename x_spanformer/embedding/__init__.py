#!/usr/bin/env python3
"""
embedding/__init__.py

Embedding module initialization for X-Spanformer.
Provides utilities for vocabulary-aware embedding generation and span candidate filtering.
"""

# Core utilities that don't require extra dependencies
from .embedding_utils import (
    load_embedding_results,
    analyze_embedding_quality, 
    compute_embedding_similarity,
    extract_span_features
)

from .span_analysis import (
    SpanAnalyzer,
    analyze_span_hierarchy,
    compute_span_coverage,
    generate_span_statistics
)

# Visualization utilities
from .embedding_viz import (
    plot_soft_probabilities,
    plot_embedding_space,
    plot_candidate_distribution,
    create_span_heatmap,
    plot_embedding_comparison,
    create_comprehensive_visualization
)

# Core functionality always available
__all__ = [
    # Embedding utilities
    'load_embedding_results',
    'analyze_embedding_quality',
    'compute_embedding_similarity',
    'extract_span_features',
    
    # Span analysis
    'SpanAnalyzer',
    'analyze_span_hierarchy', 
    'compute_span_coverage',
    'generate_span_statistics',
    
    # Visualization
    'plot_soft_probabilities',
    'plot_embedding_space', 
    'plot_candidate_distribution',
    'create_span_heatmap',
    'plot_embedding_comparison',
    'create_comprehensive_visualization',
]
