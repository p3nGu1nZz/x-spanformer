#!/usr/bin/env python3
"""
embedding/embedding_viz.py

Visualization utilities for vocab2embedding pipeline outputs.
Creates plots and heatmaps to understand span patterns, embedding spaces, and quality metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING, Any
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_soft_probabilities(soft_probs: np.ndarray, sequence: str, 
                          vocab_pieces: Optional[List[str]] = None,
                          max_pieces: int = 20, max_positions: int = 50,
                          save_path: Optional[Union[str, Path]] = None) -> "Figure":
    """
    Create a heatmap visualization of soft piece probabilities.
    
    Args:
        soft_probs: Soft probability matrix (T, |V|)
        sequence: Original input sequence
        vocab_pieces: List of vocabulary pieces (for labeling)
        max_pieces: Maximum number of pieces to display
        max_positions: Maximum number of positions to display
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    T, V = soft_probs.shape
    
    # Truncate if necessary
    display_T = min(T, max_positions)
    display_V = min(V, max_pieces)
    
    # Select top pieces by total probability
    piece_totals = np.sum(soft_probs, axis=0)
    top_pieces_idx = np.argsort(piece_totals)[-display_V:][::-1]
    
    # Create display matrix
    display_probs = soft_probs[:display_T, top_pieces_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [1, 4]})
    
    # Top panel: sequence
    ax1.text(0.05, 0.5, f"Sequence: {sequence[:max_positions * 2]}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='center')
    if len(sequence) > max_positions * 2:
        ax1.text(0.05, 0.2, "...", transform=ax1.transAxes, fontsize=10)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Bottom panel: heatmap
    im = ax2.imshow(display_probs.T, cmap='viridis', aspect='auto', 
                    interpolation='nearest')
    
    # Labels
    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Vocabulary Pieces (Top Probability)')
    ax2.set_title(f'Soft Piece Probabilities\n({display_T} positions × {display_V} pieces)')
    
    # Ticks
    if vocab_pieces:
        piece_labels = [vocab_pieces[i] for i in top_pieces_idx]
        # Set both ticks and labels to avoid matplotlib warning
        ax2.set_yticks(range(min(len(piece_labels), 20)))
        ax2.set_yticklabels(piece_labels[:20])  # Limit labels for readability
    
    # Position ticks (show every 5th position)
    pos_ticks = list(range(0, display_T, max(1, display_T // 10)))
    ax2.set_xticks(pos_ticks)
    ax2.set_xticklabels(pos_ticks)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Probability', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_embedding_space(embeddings: np.ndarray, method: str = 'pca',
                        span_candidates: Optional[List[Tuple[int, int]]] = None,
                        sequence: Optional[str] = None,
                        save_path: Optional[Union[str, Path]] = None) -> "Figure":
    """
    Visualize embedding space using dimensionality reduction.
    
    Args:
        embeddings: Embedding matrix (T, d)
        method: Reduction method ('pca', 'tsne')
        span_candidates: Optional span positions for highlighting
        sequence: Optional sequence for labeling
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    T, d = embeddings.shape
    
    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        
    elif method.lower() == 'tsne':
        perplexity = min(30, max(5, T // 4))  # Adaptive perplexity
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(embeddings)
            
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Base scatter plot
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                        c=np.arange(T), cmap='viridis', 
                        alpha=0.7, s=50)
    
    # Add position labels for some points
    if T <= 50:
        for i in range(0, T, max(1, T // 10)):
            ax.annotate(f'{i}', (reduced[i, 0], reduced[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Highlight span candidates if provided
    if span_candidates and len(span_candidates) <= 20:
        for i, (start, end) in enumerate(span_candidates):
            if start < T and end <= T:
                span_points = reduced[start:end]
                ax.plot(span_points[:, 0], span_points[:, 1], 
                       'r-', alpha=0.6, linewidth=2, label=f'Span {i+1}' if i < 5 else "")
    
    title_suffix = f"({method.upper()})" if not sequence else f"({method.upper()}, {len(sequence)} chars)"
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Embedding Space Visualization {title_suffix}')
    
    # Colorbar for position
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sequence Position', rotation=270, labelpad=15)
    
    # Add sequence info if available
    if sequence and len(sequence) <= 100:
        ax.text(0.02, 0.98, f"Sequence: {sequence}", transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_candidate_distribution(span_candidates: List[Tuple[int, int]], 
                               sequence_length: int,
                               save_path: Optional[Union[str, Path]] = None) -> "Figure":
    """
    Visualize the distribution of span candidates.
    
    Args:
        span_candidates: List of (start, end) span positions
        sequence_length: Length of the original sequence
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    if not span_candidates:
        raise ValueError("No span candidates provided")
    
    # Extract span properties
    starts = [start for start, end in span_candidates]
    ends = [end for start, end in span_candidates]
    lengths = [end - start for start, end in span_candidates]
    centers = [(start + end) / 2 for start, end in span_candidates]
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Length distribution
    ax1.hist(lengths, bins=min(20, max(lengths) - min(lengths) + 1), 
             alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Span Length')
    ax1.set_ylabel('Count')
    ax1.set_title('Span Length Distribution')
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(lengths):.1f}')
    ax1.legend()
    
    # 2. Start position distribution
    start_bins = max(1, min(30, sequence_length // 5))
    ax2.hist(starts, bins=start_bins, 
             alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Start Position')
    ax2.set_ylabel('Count')
    ax2.set_title('Start Position Distribution')
    ax2.axvline(np.mean(starts), color='red', linestyle='--',
               label=f'Mean: {np.mean(starts):.1f}')
    ax2.legend()
    
    # 3. End position distribution  
    end_bins = max(1, min(30, sequence_length // 5))
    ax3.hist(ends, bins=end_bins,
             alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('End Position')
    ax3.set_ylabel('Count')
    ax3.set_title('End Position Distribution')
    ax3.axvline(np.mean(ends), color='red', linestyle='--',
               label=f'Mean: {np.mean(ends):.1f}')
    ax3.legend()
    
    # 4. Span density across sequence
    density = np.zeros(sequence_length)
    for start, end in span_candidates:
        density[start:end] += 1
    
    positions = np.arange(sequence_length)
    ax4.plot(positions, density, color='purple', linewidth=2)
    ax4.fill_between(positions, density, alpha=0.3, color='purple')
    ax4.set_xlabel('Sequence Position')
    ax4.set_ylabel('Span Coverage Depth')
    ax4.set_title('Span Coverage Density')
    ax4.axhline(np.mean(density), color='red', linestyle='--',
               label=f'Mean depth: {np.mean(density):.1f}')
    ax4.legend()
    
    plt.suptitle(f'Span Analysis for {len(span_candidates)} candidates', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_span_heatmap(sequence: str, span_candidates: List[Tuple[int, int]], 
                       max_spans: int = 50,
                       save_path: Optional[Union[str, Path]] = None) -> "Figure":
    """
    Create a heatmap showing span coverage over the sequence.
    
    Args:
        sequence: Input sequence
        span_candidates: List of (start, end) span positions
        max_spans: Maximum number of spans to display
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    if not span_candidates:
        raise ValueError("No span candidates provided")
    
    sequence_length = len(sequence)
    
    # Select spans to display (sorted by length, then position)
    display_spans = sorted(span_candidates, key=lambda x: (x[1] - x[0], x[0]))
    display_spans = display_spans[:max_spans]
    
    # Create span matrix (spans × positions)
    span_matrix = np.zeros((len(display_spans), sequence_length))
    
    for i, (start, end) in enumerate(display_spans):
        if start < sequence_length and end <= sequence_length:
            span_matrix[i, start:end] = 1
    
    # Create figure
    fig_height = max(6, len(display_spans) * 0.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, fig_height),
                                  gridspec_kw={'height_ratios': [1, 5]})
    
    # Top panel: sequence characters (if short enough)
    if sequence_length <= 100:
        ax1.text(0.02, 0.5, ' '.join(sequence), transform=ax1.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='center')
    else:
        ax1.text(0.02, 0.5, f"Sequence: {sequence[:50]}... (length: {sequence_length})", 
                transform=ax1.transAxes, fontsize=10, verticalalignment='center')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Bottom panel: span heatmap
    im = ax2.imshow(span_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    
    # Labels and ticks
    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Span Candidates')
    ax2.set_title(f'Span Coverage Heatmap ({len(display_spans)} spans shown)')
    
    # X-axis ticks (positions)
    x_ticks = list(range(0, sequence_length, max(1, sequence_length // 20)))
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticks)
    
    # Y-axis ticks (span info)
    if len(display_spans) <= 20:
        y_labels = [f'({s},{e})' for s, e in display_spans]
        ax2.set_yticks(range(len(display_spans)))
        ax2.set_yticklabels(y_labels)
    else:
        y_ticks = list(range(0, len(display_spans), max(1, len(display_spans) // 10)))
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels([f'Span {i+1}' for i in y_ticks])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_embedding_comparison(seed_embeddings: np.ndarray, 
                             contextual_embeddings: np.ndarray,
                             save_path: Optional[Union[str, Path]] = None) -> "Figure":
    """
    Compare seed embeddings vs contextual embeddings.
    
    Args:
        seed_embeddings: Seed embedding matrix (T, d)
        contextual_embeddings: Contextual embedding matrix (T, d)
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Compute similarity between seed and contextual embeddings
    similarities = []
    for i in range(len(seed_embeddings)):
        sim = np.dot(seed_embeddings[i], contextual_embeddings[i]) / (
            np.linalg.norm(seed_embeddings[i]) * np.linalg.norm(contextual_embeddings[i])
        )
        similarities.append(sim)
    
    # Compute norms
    seed_norms = np.linalg.norm(seed_embeddings, axis=1)
    context_norms = np.linalg.norm(contextual_embeddings, axis=1)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Similarity distribution
    ax1.hist(similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title('Seed vs Contextual Embedding Similarity')
    ax1.axvline(np.mean(similarities), color='red', linestyle='--',
               label=f'Mean: {np.mean(similarities):.3f}')
    ax1.legend()
    
    # 2. Norm comparison
    positions = np.arange(len(seed_norms))
    ax2.plot(positions, seed_norms, 'b-', alpha=0.7, label='Seed Embeddings')
    ax2.plot(positions, context_norms, 'r-', alpha=0.7, label='Contextual Embeddings')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('L2 Norm')
    ax2.set_title('Embedding Norms Comparison')
    ax2.legend()
    
    # 3. Norm distributions
    ax3.hist(seed_norms, bins=20, alpha=0.5, color='blue', label='Seed', edgecolor='black')
    ax3.hist(context_norms, bins=20, alpha=0.5, color='red', label='Contextual', edgecolor='black')
    ax3.set_xlabel('L2 Norm')
    ax3.set_ylabel('Count')
    ax3.set_title('Embedding Norm Distributions')
    ax3.legend()
    
    # 4. Similarity vs position
    ax4.scatter(positions, similarities, alpha=0.6, color='green')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Similarity by Position')
    ax4.axhline(np.mean(similarities), color='red', linestyle='--',
               label=f'Mean: {np.mean(similarities):.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_comprehensive_visualization(result_dir: Union[str, Path], 
                                     sequence_id: int,
                                     output_dir: Optional[Union[str, Path]] = None) -> Dict[str, "Figure"]:
    """
    Create a comprehensive set of visualizations for a single sequence result.
    
    Args:
        result_dir: Directory containing embedding pipeline outputs
        sequence_id: Sequence ID to visualize
        output_dir: Optional directory to save all plots
        
    Returns:
        Dictionary mapping plot names to figure objects
    """
    from .embedding_utils import load_embedding_results
    
    # Load results
    result = load_embedding_results(result_dir, sequence_id)
    metadata = result['metadata']
    
    sequence = metadata['sequence']
    span_candidates = metadata['span_candidates']
    
    figures = {}
    
    # 1. Span candidate distribution
    try:
        fig = plot_candidate_distribution(span_candidates, len(sequence))
        figures['candidate_distribution'] = fig
        if output_dir:
            save_path = Path(output_dir) / f"candidate_distribution_{sequence_id:06d}.png"
            plt.figure(fig.number)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        logger.warning(f"Could not create candidate distribution plot: {e}")
    
    # 2. Span heatmap
    try:
        fig = create_span_heatmap(sequence, span_candidates)
        figures['span_heatmap'] = fig
        if output_dir:
            save_path = Path(output_dir) / f"span_heatmap_{sequence_id:06d}.png"
            plt.figure(fig.number)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        logger.warning(f"Could not create span heatmap: {e}")
    
    # 3. Soft probabilities (if available)
    if result['soft_probabilities'] is not None:
        try:
            fig = plot_soft_probabilities(result['soft_probabilities'], sequence)
            figures['soft_probabilities'] = fig
            if output_dir:
                save_path = Path(output_dir) / f"soft_probabilities_{sequence_id:06d}.png"
                plt.figure(fig.number)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Could not create soft probabilities plot: {e}")
    
    # 4. Contextual embedding space
    if result['contextual_embeddings'] is not None:
        try:
            fig = plot_embedding_space(result['contextual_embeddings'], 
                                     span_candidates=span_candidates, sequence=sequence)
            figures['embedding_space'] = fig
            if output_dir:
                save_path = Path(output_dir) / f"embedding_space_{sequence_id:06d}.png"
                plt.figure(fig.number)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Could not create embedding space plot: {e}")
    
    # 5. Seed vs contextual comparison
    if (result['seed_embeddings'] is not None and 
        result['contextual_embeddings'] is not None):
        try:
            fig = plot_embedding_comparison(result['seed_embeddings'], 
                                          result['contextual_embeddings'])
            figures['embedding_comparison'] = fig
            if output_dir:
                save_path = Path(output_dir) / f"embedding_comparison_{sequence_id:06d}.png"
                plt.figure(fig.number)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Could not create embedding comparison plot: {e}")
    
    logger.info(f"Created {len(figures)} visualizations for sequence {sequence_id}")
    return figures
