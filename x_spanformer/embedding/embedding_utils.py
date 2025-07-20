#!/usr/bin/env python3
"""
embedding/embedding_utils.py

Utility functions for working with vocab2embedding pipeline outputs.
Provides analysis, loading, and processing functions for embedding results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embedding_results(result_dir: Union[str, Path], sequence_id: int) -> Dict:
    """
    Load complete embedding results for a specific sequence.
    
    Args:
        result_dir: Directory containing embedding pipeline outputs
        sequence_id: Sequence ID to load
        
    Returns:
        Dictionary containing all embedding data and metadata
    """
    result_dir = Path(result_dir)
    sequence_id_str = f"{sequence_id:06d}"
    
    # Load metadata
    metadata_file = result_dir / f"embedding_{sequence_id_str}.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load numpy arrays
    result = {
        'metadata': metadata,
        'soft_probabilities': None,
        'seed_embeddings': None,
        'contextual_embeddings': None
    }
    
    # Load soft probabilities
    soft_probs_file = result_dir / f"soft_probs_{sequence_id_str}.npy"
    if soft_probs_file.exists():
        result['soft_probabilities'] = np.load(soft_probs_file)
    
    # Load seed embeddings  
    seed_emb_file = result_dir / f"seed_emb_{sequence_id_str}.npy"
    if seed_emb_file.exists():
        result['seed_embeddings'] = np.load(seed_emb_file)
    
    # Load contextual embeddings
    context_emb_file = result_dir / f"context_emb_{sequence_id_str}.npy"
    if context_emb_file.exists():
        result['contextual_embeddings'] = np.load(context_emb_file)
    
    return result


def analyze_embedding_quality(embeddings: np.ndarray) -> Dict:
    """
    Analyze the quality of embedding representations.
    
    Args:
        embeddings: Embedding matrix of shape (T, d)
        
    Returns:
        Dictionary containing quality metrics
    """
    T, d = embeddings.shape
    
    # Basic statistics
    mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
    std_norm = np.std(np.linalg.norm(embeddings, axis=1))
    
    # Embedding variance per dimension
    dim_variances = np.var(embeddings, axis=0)
    mean_var = np.mean(dim_variances)
    min_var = np.min(dim_variances)
    max_var = np.max(dim_variances)
    
    # Pairwise similarity statistics
    similarity_matrix = np.corrcoef(embeddings)
    off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    mean_similarity = np.mean(off_diagonal)
    std_similarity = np.std(off_diagonal)
    
    return {
        'sequence_length': T,
        'embedding_dim': d,
        'mean_embedding_norm': float(mean_norm),
        'std_embedding_norm': float(std_norm),
        'mean_dimension_variance': float(mean_var),
        'min_dimension_variance': float(min_var),
        'max_dimension_variance': float(max_var),
        'dimension_variance_ratio': float(max_var / min_var) if min_var > 0 else float('inf'),
        'mean_pairwise_similarity': float(mean_similarity),
        'std_pairwise_similarity': float(std_similarity)
    }


def compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray, 
                                method: str = 'cosine') -> np.ndarray:
    """
    Compute similarity between two embedding matrices.
    
    Args:
        emb1: First embedding matrix (T1, d)
        emb2: Second embedding matrix (T2, d) 
        method: Similarity method ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        Similarity matrix of shape (T1, T2)
    """
    if method == 'cosine':
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        return np.dot(emb1_norm, emb2_norm.T)
    
    elif method == 'euclidean':
        # Compute pairwise Euclidean distances
        distances = np.linalg.norm(emb1[:, None] - emb2[None, :], axis=2)
        # Convert to similarities (higher = more similar)
        return 1 / (1 + distances)
    
    elif method == 'manhattan':
        # Compute pairwise Manhattan distances
        distances = np.sum(np.abs(emb1[:, None] - emb2[None, :]), axis=2)
        return 1 / (1 + distances)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def extract_span_features(embeddings: np.ndarray, span_candidates: List[Tuple[int, int]],
                         aggregation: str = 'mean') -> np.ndarray:
    """
    Extract features for span candidates from contextual embeddings.
    
    Args:
        embeddings: Contextual embeddings (T, d)
        span_candidates: List of (start, end) span positions
        aggregation: How to aggregate embeddings within spans ('mean', 'max', 'sum')
        
    Returns:
        Span feature matrix (num_spans, d)
    """
    T, d = embeddings.shape
    span_features = []
    
    for start, end in span_candidates:
        if start >= T or end > T or start >= end:
            logger.warning(f"Invalid span ({start}, {end}) for sequence length {T}")
            # Add zero vector for invalid spans
            span_features.append(np.zeros(d))
            continue
        
        span_embeddings = embeddings[start:end]
        
        if aggregation == 'mean':
            span_feat = np.mean(span_embeddings, axis=0)
        elif aggregation == 'max':
            span_feat = np.max(span_embeddings, axis=0)
        elif aggregation == 'sum':
            span_feat = np.sum(span_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        span_features.append(span_feat)
    
    return np.array(span_features)


def visualize_span_candidates(sequence: str, span_candidates: List[Tuple[int, int]], 
                             max_display: int = 20) -> str:
    """
    Create a text visualization of span candidates over a sequence.
    
    Args:
        sequence: Input text sequence
        span_candidates: List of (start, end) span positions
        max_display: Maximum number of spans to display
        
    Returns:
        Formatted string showing span overlays
    """
    if not span_candidates:
        return f"Sequence: {sequence}\nNo span candidates found."
    
    # Sort candidates by start position, then by length
    sorted_candidates = sorted(span_candidates[:max_display], 
                             key=lambda x: (x[0], x[1] - x[0]))
    
    output_lines = [f"Sequence: {sequence}"]
    output_lines.append(f"Length: {len(sequence)} characters")
    output_lines.append(f"Candidates shown: {len(sorted_candidates)} / {len(span_candidates)}")
    output_lines.append("")
    
    for i, (start, end) in enumerate(sorted_candidates):
        if start >= len(sequence) or end > len(sequence):
            continue
            
        # Create span visualization line
        span_line = [' '] * len(sequence)
        span_text = sequence[start:end]
        
        # Mark span boundaries
        span_line[start] = '['
        if end - 1 < len(span_line):
            span_line[end - 1] = ']'
        
        # Fill span interior
        for j in range(start + 1, min(end - 1, len(span_line))):
            if span_line[j] == ' ':
                span_line[j] = '-'
        
        span_line_str = ''.join(span_line)
        output_lines.append(f"Span {i+1:2d}: {span_line_str}")
        output_lines.append(f"         Text: '{span_text}' ({start}:{end})")
        output_lines.append("")
    
    return '\n'.join(output_lines)


def batch_analyze_embeddings(result_dir: Union[str, Path], 
                            sequence_ids: Optional[List[int]] = None) -> Dict:
    """
    Analyze embedding quality across multiple sequences.
    
    Args:
        result_dir: Directory containing embedding results
        sequence_ids: List of sequence IDs to analyze (None for all)
        
    Returns:
        Dictionary containing aggregate quality metrics
    """
    result_dir = Path(result_dir)
    
    if sequence_ids is None:
        # Find all available sequence IDs
        metadata_files = list(result_dir.glob("embedding_*.json"))
        sequence_ids = []
        for f in metadata_files:
            try:
                seq_id = int(f.stem.split('_')[1])
                sequence_ids.append(seq_id)
            except (IndexError, ValueError):
                continue
        sequence_ids.sort()
    
    logger.info(f"Analyzing {len(sequence_ids)} sequences")
    
    # Collect quality metrics for each sequence
    all_metrics = []
    embedding_norms = []
    similarity_scores = []
    
    for seq_id in sequence_ids:
        try:
            result = load_embedding_results(result_dir, seq_id)
            
            if result['contextual_embeddings'] is not None:
                metrics = analyze_embedding_quality(result['contextual_embeddings'])
                all_metrics.append(metrics)
                
                embedding_norms.extend(np.linalg.norm(result['contextual_embeddings'], axis=1))
                
                if result['seed_embeddings'] is not None:
                    # Compare seed vs contextual embeddings
                    sim = compute_embedding_similarity(result['seed_embeddings'], 
                                                     result['contextual_embeddings'])
                    similarity_scores.extend(np.diag(sim))  # Self-similarities
                    
        except Exception as e:
            logger.warning(f"Failed to analyze sequence {seq_id}: {e}")
            continue
    
    if not all_metrics:
        return {'error': 'No valid sequences found'}
    
    # Aggregate statistics
    aggregate_stats = {
        'num_sequences': len(all_metrics),
        'mean_sequence_length': np.mean([m['sequence_length'] for m in all_metrics]),
        'embedding_dim': all_metrics[0]['embedding_dim'],
        
        # Embedding norm statistics
        'global_mean_norm': np.mean(embedding_norms),
        'global_std_norm': np.std(embedding_norms),
        
        # Quality metrics across sequences
        'mean_dimension_variance': np.mean([m['mean_dimension_variance'] for m in all_metrics]),
        'mean_pairwise_similarity': np.mean([m['mean_pairwise_similarity'] for m in all_metrics]),
        
        # Seed vs contextual similarity (if available)
        'seed_context_similarity': {
            'mean': np.mean(similarity_scores) if similarity_scores else None,
            'std': np.std(similarity_scores) if similarity_scores else None
        }
    }
    
    return aggregate_stats


def export_embeddings_to_numpy(result_dir: Union[str, Path], output_file: Union[str, Path],
                               embedding_type: str = 'contextual') -> None:
    """
    Export all embeddings of a specific type to a single numpy file.
    
    Args:
        result_dir: Directory containing embedding results
        output_file: Output numpy file path
        embedding_type: Type of embeddings to export ('contextual', 'seed', 'soft_probs')
    """
    result_dir = Path(result_dir)
    output_file = Path(output_file)
    
    # Find all sequences
    if embedding_type == 'contextual':
        pattern = "context_emb_*.npy"
    elif embedding_type == 'seed':
        pattern = "seed_emb_*.npy"
    elif embedding_type == 'soft_probs':
        pattern = "soft_probs_*.npy"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    embedding_files = sorted(result_dir.glob(pattern))
    
    if not embedding_files:
        raise FileNotFoundError(f"No {embedding_type} files found in {result_dir}")
    
    # Load and concatenate embeddings
    all_embeddings = []
    sequence_lengths = []
    
    for emb_file in embedding_files:
        embeddings = np.load(emb_file)
        all_embeddings.append(embeddings)
        sequence_lengths.append(embeddings.shape[0])
    
    # Create concatenated array with metadata
    concatenated = np.concatenate(all_embeddings, axis=0)
    
    # Save with metadata
    output_data = {
        'embeddings': concatenated,
        'sequence_lengths': sequence_lengths,
        'num_sequences': len(embedding_files),
        'total_positions': concatenated.shape[0],
        'embedding_dim': concatenated.shape[1],
        'embedding_type': embedding_type
    }
    
    np.savez_compressed(output_file, **output_data)
    logger.info(f"Exported {concatenated.shape[0]} {embedding_type} embeddings to {output_file}")
