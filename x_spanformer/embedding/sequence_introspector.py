"""
X-Spanformer Sequence Introspector

A CLI tool for inspecting and analyzing processed sequences from the vocab2embedding pipeline.
This tool allows deep introspection of the various neural network layers and representations
generated during the embedding pipeline processing.

Mathematical Foundation (from Section 3.2):
- H⁰: Seed embeddings from soft probabilities P·W_emb
- H: Contextual embeddings from multi-scale dilated convolutions  
- P: Soft probability matrix (T × |V|) from forward-backward algorithm
- Span candidates: Filtered candidate set with dynamic w_max

Usage:
    python -m x_spanformer.embedding.sequence_introspector --id 1 --output data/embedding/out
    python -m x_spanformer.embedding.sequence_introspector --id 5 --output data/embedding/out --analyze
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import sys


class SequenceIntrospector:
    """
    Introspects a single processed sequence from the vocab2embedding pipeline.
    
    This class provides access to all neural network layers and representations:
    - Seed embeddings (H⁰): Initial dense representations from soft probabilities
    - Contextual embeddings (H): Multi-scale convolutionally enhanced representations
    - Soft probabilities (P): Forward-backward probability matrix
    - Span candidates: Filtered spans for boundary prediction
    """
    
    def __init__(self, output_dir: Path):
        """Initialize introspector with output directory."""
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "json"
        self.seed_dir = self.output_dir / "seed"
        self.context_dir = self.output_dir / "context"
        self.soft_prob_dir = self.output_dir / "soft_prob"
        
        # Validate essential directories exist
        essential_dirs = [self.context_dir]  # Only contextual embeddings are essential
        for dir_path in essential_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Essential directory not found: {dir_path}")
        
        # Optional directories (may not exist based on config)
        self.json_available = self.json_dir.exists()
        self.seed_available = self.seed_dir.exists()
        self.soft_prob_available = self.soft_prob_dir.exists()
    
    def get_sequence_count(self) -> int:
        """Get total number of processed sequences."""
        if self.json_available:
            json_files = list(self.json_dir.glob("embedding_*.json"))
            return len(json_files)
        elif self.seed_available:
            # Fall back to counting seed embedding files
            seed_files = list(self.seed_dir.glob("seed_emb_*.npy"))
            return len(seed_files)
        else:
            # Final fallback to contextual embedding files
            context_files = list(self.context_dir.glob("context_emb_*.npy"))
            return len(context_files)
    
    def load_sequence(self, seq_id: int) -> Dict[str, Any]:
        """
        Load all data for a specific sequence ID.
        
        Args:
            seq_id: Sequence identifier (1-based)
            
        Returns:
            Dictionary containing all sequence data:
            - metadata: JSON metadata including span candidates
            - seed_embeddings: H⁰ matrix (T × 512)
            - contextual_embeddings: H matrix (T × 512)  
            - soft_probabilities: P matrix (T × |V|)
            
        Raises:
            FileNotFoundError: If sequence files don't exist
            ValueError: If seq_id is invalid
        """
        if seq_id < 1:
            raise ValueError("Sequence ID must be >= 1")
        
        # Check if sequence exists
        total_sequences = self.get_sequence_count()
        if seq_id > total_sequences:
            raise ValueError(f"Sequence ID {seq_id} exceeds total sequences {total_sequences}")
        
        # Load metadata if available
        metadata = None
        if self.json_available:
            json_file = self.json_dir / f"embedding_{seq_id:06d}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
        
        # Load numpy arrays
        seed_file = self.seed_dir / f"seed_emb_{seq_id:06d}.npy"
        context_file = self.context_dir / f"context_emb_{seq_id:06d}.npy"
        soft_prob_file = self.soft_prob_dir / f"soft_probs_{seq_id:06d}.npy"
        
        # Essential files that must exist
        essential_files = [
            ("contextual embeddings", context_file)
        ]
        
        for name, file_path in essential_files:
            if not file_path.exists():
                raise FileNotFoundError(f"{name} file not found: {file_path}")
        
        # Load arrays
        data = {
            'metadata': metadata,
            'contextual_embeddings': np.load(context_file),
        }
        
        # Seed embeddings are optional (may be disabled for performance)
        if self.seed_available and seed_file.exists():
            data['seed_embeddings'] = np.load(seed_file)
        else:
            data['seed_embeddings'] = None
        
        # Soft probabilities are optional (may be disabled for performance)  
        if self.soft_prob_available and soft_prob_file.exists():
            data['soft_probabilities'] = np.load(soft_prob_file)
        else:
            data['soft_probabilities'] = None
        
        return data
    
    def get_file_sizes(self, seq_id: int) -> Dict[str, float]:
        """Get file sizes in KB for all files related to a sequence."""
        sizes = {}
        
        # Essential files
        context_file = self.context_dir / f"context_emb_{seq_id:06d}.npy"
        if context_file.exists():
            sizes['contextual_embeddings'] = context_file.stat().st_size / 1024
        
        # Optional files
        if self.seed_available:
            seed_file = self.seed_dir / f"seed_emb_{seq_id:06d}.npy"
            if seed_file.exists():
                sizes['seed_embeddings'] = seed_file.stat().st_size / 1024
        
        if self.json_available:
            json_file = self.json_dir / f"embedding_{seq_id:06d}.json"
            if json_file.exists():
                sizes['json_metadata'] = json_file.stat().st_size / 1024
        
        if self.soft_prob_available:
            soft_prob_file = self.soft_prob_dir / f"soft_probs_{seq_id:06d}.npy"
            if soft_prob_file.exists():
                sizes['soft_probabilities'] = soft_prob_file.stat().st_size / 1024
        
        # Calculate total
        sizes['total'] = sum(sizes.values())
        
        return sizes

    def analyze_sequence(self, seq_id: int, detailed: bool = False) -> Dict[str, Any]:
        """
        Perform analysis on a sequence.
        
        Args:
            seq_id: Sequence identifier
            detailed: Include detailed statistical analysis
            
        Returns:
            Analysis dictionary with statistics and insights
        """
        data = self.load_sequence(seq_id)
        metadata = data['metadata']
        
        # Get file sizes
        file_sizes = self.get_file_sizes(seq_id)
        
        # Handle missing metadata gracefully
        if metadata:
            analysis = {
                'sequence_id': seq_id,
                'sequence_length': metadata['sequence_length'],
                'num_span_candidates': metadata['num_candidates'],
                'sequence_preview': metadata['sequence'][:100] + "..." if len(metadata['sequence']) > 100 else metadata['sequence']
            }
        else:
            # Fallback when no JSON metadata available
            context_shape = data['contextual_embeddings'].shape
            analysis = {
                'sequence_id': seq_id,
                'sequence_length': context_shape[0],  # Infer from embedding shape
                'num_span_candidates': 'Unknown (no metadata)',
                'sequence_preview': 'Not available (no metadata)'
            }
        
        # Array shape analysis
        seed_shape = data['seed_embeddings'].shape if data['seed_embeddings'] is not None else None
        context_shape = data['contextual_embeddings'].shape
        soft_prob_shape = data['soft_probabilities'].shape if data['soft_probabilities'] is not None else None
        
        analysis.update({
            'seed_embeddings_shape': seed_shape,  # Should be (T, 512) or None
            'contextual_embeddings_shape': context_shape,  # Should be (T, 512)
            'soft_probabilities_shape': soft_prob_shape,  # Should be (T, |V|) or None
            'vocabulary_size': soft_prob_shape[1] if soft_prob_shape is not None and len(soft_prob_shape) > 1 else None,
            'seed_embeddings_available': data['seed_embeddings'] is not None,
            'soft_probabilities_available': data['soft_probabilities'] is not None,
            'file_sizes': file_sizes
        })
        
        if detailed:
            # Statistical analysis
            seed_emb = data['seed_embeddings']
            context_emb = data['contextual_embeddings']
            soft_probs = data['soft_probabilities']
            
            analysis['detailed_stats'] = {
                'contextual_embeddings': {
                    'mean': float(context_emb.mean()),
                    'std': float(context_emb.std()),
                    'min': float(context_emb.min()),
                    'max': float(context_emb.max()),
                    'sparsity': float((context_emb == 0).mean())
                }
            }
            
            # Add seed embeddings stats only if available
            if seed_emb is not None:
                analysis['detailed_stats']['seed_embeddings'] = {
                    'mean': float(seed_emb.mean()),
                    'std': float(seed_emb.std()),
                    'min': float(seed_emb.min()),
                    'max': float(seed_emb.max()),
                    'sparsity': float((seed_emb == 0).mean())
                }
            
            # Add soft probabilities stats only if available
            if soft_probs is not None:
                analysis['detailed_stats']['soft_probabilities'] = {
                    'mean': float(soft_probs.mean()),
                    'std': float(soft_probs.std()),
                    'min': float(soft_probs.min()),
                    'max': float(soft_probs.max()),
                    'sparsity': float((soft_probs == 0).mean()),
                    'row_sums_mean': float(soft_probs.sum(axis=1).mean()),
                    'row_sums_std': float(soft_probs.sum(axis=1).std())
                }
            
            # Span analysis (only if metadata available)
            if metadata:
                span_candidates = metadata.get('span_candidates', [])
                if span_candidates:
                    span_lengths = []
                    for candidate in span_candidates:
                        if len(candidate) == 2:  # [start, end] format
                            span_lengths.append(candidate[1] - candidate[0])
                    
                    if span_lengths:
                        span_stats = {
                            'total_candidates': len(span_candidates),
                            'avg_span_length': float(np.mean(span_lengths)),
                            'min_span_length': int(min(span_lengths)),
                            'max_span_length': int(max(span_lengths)),
                            'length_distribution': {
                                'q25': float(np.percentile(span_lengths, 25)),
                                'q50': float(np.percentile(span_lengths, 50)),
                                'q75': float(np.percentile(span_lengths, 75))
                            }
                        }
                        analysis['detailed_stats']['span_candidates'] = span_stats
        
        return analysis
    
    def print_summary(self, seq_id: int, analyze: bool = False, verbose: bool = False):
        """Print a formatted summary of the sequence."""
        try:
            analysis = self.analyze_sequence(seq_id, detailed=analyze)
            
            print(f"\n{'='*80}")
            print(f"X-SPANFORMER SEQUENCE INTROSPECTOR - SEQUENCE {seq_id}")
            print(f"{'='*80}")
            
            print(f"\n== SEQUENCE INFORMATION:")
            print(f"   Sequence Length: {analysis['sequence_length']} characters")
            print(f"   Span Candidates: {analysis['num_span_candidates']}")
            print(f"   Vocabulary Size: {analysis['vocabulary_size']}")
            
            print(f"\n== NEURAL REPRESENTATIONS:")
            if analysis['seed_embeddings_available']:
                print(f"   Seed Embeddings (H0):     {analysis['seed_embeddings_shape']}")
            else:
                print(f"   Seed Embeddings (H0):     Not saved (performance optimization)")
            print(f"   Contextual Embeddings (H): {analysis['contextual_embeddings_shape']}")  
            if analysis['soft_probabilities_available']:
                print(f"   Soft Probabilities (P):    {analysis['soft_probabilities_shape']}")
            else:
                print(f"   Soft Probabilities (P):    Not saved (performance optimization)")
            
            print(f"\n== FILE SIZES:")
            file_sizes = analysis['file_sizes']
            if 'json_metadata' in file_sizes:
                print(f"   JSON Metadata: {file_sizes['json_metadata']:.1f} KB")
            else:
                print(f"   JSON Metadata: Not saved (disabled in config)")
            if 'seed_embeddings' in file_sizes:
                print(f"   Seed Embeddings: {file_sizes['seed_embeddings']:.1f} KB")
            else:
                print(f"   Seed Embeddings: Not saved (performance optimization)")
            print(f"   Contextual Embeddings: {file_sizes['contextual_embeddings']:.1f} KB")
            if 'soft_probabilities' in file_sizes:
                print(f"   Soft Probabilities: {file_sizes['soft_probabilities']:.1f} KB")
            else:
                print(f"   Soft Probabilities: Not saved (performance optimization)")
            print(f"   Total Size: {file_sizes['total']:.1f} KB")
            
            print(f"\n== SEQUENCE PREVIEW:")
            print(f"   {repr(analysis['sequence_preview'])}")
            
            # Always show partial arrays (removed verbose requirement)
            data = self.load_sequence(seq_id)
            
            # Show seed embeddings sample (if available)
            if data['seed_embeddings'] is not None:
                seed_embeddings = data['seed_embeddings']
                print(f"\n== SEED EMBEDDINGS (H0) - SAMPLE VALUES:")
                print(f"   Shape: {seed_embeddings.shape}")
                if seed_embeddings.size > 0:
                    # Show first 3 positions with first 10 dimensions each
                    sample_positions = min(3, seed_embeddings.shape[0])
                    sample_dims = min(10, seed_embeddings.shape[1])
                    print(f"   Sample values (first {sample_positions} positions, first {sample_dims} dimensions):")
                    for i in range(sample_positions):
                        values = seed_embeddings[i, :sample_dims]
                        values_str = ' '.join([f"{v:8.6f}" for v in values])
                        print(f"     Position {i:3d}: [{values_str}...]")
            else:
                print(f"\n== SEED EMBEDDINGS (H0):")
                print(f"   Not available (disabled for performance optimization)")
            
            # Show contextual embeddings sample
            contextual_embeddings = data['contextual_embeddings']
            print(f"\n== CONTEXTUAL EMBEDDINGS (H) - SAMPLE VALUES:")
            print(f"   Shape: {contextual_embeddings.shape}")
            if contextual_embeddings.size > 0:
                # Show first 3 positions with first 10 dimensions each
                sample_positions = min(3, contextual_embeddings.shape[0])
                sample_dims = min(10, contextual_embeddings.shape[1])
                print(f"   Sample values (first {sample_positions} positions, first {sample_dims} dimensions):")
                for i in range(sample_positions):
                    values = contextual_embeddings[i, :sample_dims]
                    values_str = ' '.join([f"{v:8.6f}" for v in values])
                    print(f"     Position {i:3d}: [{values_str}...]")
            
            # Show soft probabilities sample if available
            if data['soft_probabilities'] is not None:
                soft_probs = data['soft_probabilities']
                print(f"\n== SOFT PROBABILITIES (P) - SAMPLE VALUES:")
                print(f"   Shape: {soft_probs.shape}")
                if soft_probs.size > 0:
                    # Show first 2 positions with top 5 probabilities each
                    sample_positions = min(2, soft_probs.shape[0])
                    print(f"   Top probabilities (first {sample_positions} positions, highest 5 values):")
                    for i in range(sample_positions):
                        row_probs = soft_probs[i, :]
                        top_indices = np.argsort(row_probs)[-5:][::-1]  # Top 5 indices
                        print(f"     Position {i:3d}: Top vocab indices and probs:")
                        for j, vocab_idx in enumerate(top_indices):
                            prob = row_probs[vocab_idx]
                            print(f"       {vocab_idx:5d}: {prob:10.8f}")
            else:
                print(f"\n== SOFT PROBABILITIES (P):")
                print(f"   Not available (disabled for performance optimization)")
            
            if verbose:
                # Show complete arrays in verbose mode
                print(f"\n== COMPLETE ARRAYS (VERBOSE MODE):")
                
                if data['seed_embeddings'] is not None:
                    print(f"\n--- SEED EMBEDDINGS (COMPLETE) ---")
                    np.set_printoptions(precision=6, suppress=True, linewidth=200)
                    print(data['seed_embeddings'])
                else:
                    print(f"\n--- SEED EMBEDDINGS (COMPLETE) ---")
                    print("Not available (disabled for performance optimization)")
                
                print(f"\n--- CONTEXTUAL EMBEDDINGS (COMPLETE) ---")
                print(contextual_embeddings)
                
                if data['soft_probabilities'] is not None:
                    print(f"\n--- SOFT PROBABILITIES (COMPLETE) ---")
                    print(data['soft_probabilities'])
                
                # Reset to defaults
                np.set_printoptions()
            
            if analyze and 'detailed_stats' in analysis:
                stats = analysis['detailed_stats']
                
                print(f"\n== STATISTICAL ANALYSIS:")
                
                # Show seed embeddings stats only if available
                if 'seed_embeddings' in stats:
                    print(f"   Seed Embeddings:")
                    print(f"     Mean: {stats['seed_embeddings']['mean']:.6f}, Std: {stats['seed_embeddings']['std']:.6f}")
                    print(f"     Range: [{stats['seed_embeddings']['min']:.6f}, {stats['seed_embeddings']['max']:.6f}]")
                    print(f"     Sparsity: {stats['seed_embeddings']['sparsity']:.2%}")
                else:
                    print(f"   Seed Embeddings: Not available (disabled for performance)")
                
                print(f"   Contextual Embeddings:")
                print(f"     Mean: {stats['contextual_embeddings']['mean']:.6f}, Std: {stats['contextual_embeddings']['std']:.6f}")
                print(f"     Range: [{stats['contextual_embeddings']['min']:.6f}, {stats['contextual_embeddings']['max']:.6f}]")
                print(f"     Sparsity: {stats['contextual_embeddings']['sparsity']:.2%}")
                
                if 'soft_probabilities' in stats:
                    print(f"   Soft Probabilities:")
                    print(f"     Mean: {stats['soft_probabilities']['mean']:.6f}, Std: {stats['soft_probabilities']['std']:.6f}")
                    print(f"     Row Sums: {stats['soft_probabilities']['row_sums_mean']:.6f} ± {stats['soft_probabilities']['row_sums_std']:.6f}")
                    print(f"     Sparsity: {stats['soft_probabilities']['sparsity']:.2%}")
                else:
                    print(f"   Soft Probabilities: Not available (disabled for performance)")
                
                if 'span_candidates' in stats:
                    span_stats = stats['span_candidates']
                    print(f"   Span Candidates:")
                    print(f"     Total: {span_stats['total_candidates']}")
                    print(f"     Length: Avg {span_stats['avg_span_length']:.1f}, Range [{span_stats['min_span_length']}, {span_stats['max_span_length']}]")
                    print(f"     Quartiles: {span_stats['length_distribution']['q25']:.1f}, {span_stats['length_distribution']['q50']:.1f}, {span_stats['length_distribution']['q75']:.1f}")
            
            print(f"\n{'='*80}")
            
        except Exception as e:
            print(f"Error analyzing sequence {seq_id}: {e}", file=sys.stderr)
            return False
        
        return True


def main():
    """CLI entry point for sequence introspection."""
    parser = argparse.ArgumentParser(
        description="X-Spanformer Sequence Introspector - Analyze processed embedding sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --id 1 --output data/embedding/out
  %(prog)s --id 5 --output data/embedding/out --analyze
  %(prog)s --id 10 --output data/embedding/out --list-total
  %(prog)s --id 1 --output data/embedding/out -v --analyze
        """
    )
    
    parser.add_argument(
        "--id", type=int, required=True,
        help="Sequence ID to introspect (1-based indexing)"
    )
    
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to embedding output directory containing seed/, context/, and optional json/, soft_prob/ subdirectories"
    )
    
    parser.add_argument(
        "--analyze", action="store_true",
        help="Include detailed statistical analysis of embeddings and probabilities"
    )
    
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show complete arrays (very long output - use with caution)"
    )
    
    parser.add_argument(
        "--list-total", action="store_true",
        help="Show total number of processed sequences"
    )
    
    args = parser.parse_args()
    
    try:
        introspector = SequenceIntrospector(args.output)
        
        if args.list_total:
            total = introspector.get_sequence_count()
            print(f"Total processed sequences: {total}")
            if args.id > total:
                print(f"Error: Sequence ID {args.id} exceeds available sequences (max: {total})")
                return 1
        
        success = introspector.print_summary(args.id, args.analyze, args.verbose)
        return 0 if success else 1
        
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
