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

# Add the parent directory to the path to import schema modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.embedding.embedding_chunk import ChunkManager


class SequenceIntrospector:
    """
    Introspects a single processed sequence from the vocab2embedding pipeline.
    
    This class provides access to all neural network layers and representations:
    - Seed embeddings (H⁰): Initial dense representations from soft probabilities
    - Contextual embeddings (H): Multi-scale convolutionally enhanced representations
    - Soft probabilities (P): Forward-backward probability matrix
    - Span candidates: Filtered spans for boundary prediction
    
    Uses the new chunk-based storage system.
    """
    
    def __init__(self, output_dir: Path):
        """Initialize introspector with output directory."""
        self.output_dir = Path(output_dir)
        
        # Check for chunk-based storage
        self.chunks_dir = self.output_dir / "chunks"
        self.metadata_file = self.output_dir / "metadata.json"
        
        if not (self.chunks_dir.exists() and self.metadata_file.exists()):
            raise FileNotFoundError(
                f"Chunk-based storage not found in {output_dir}. "
                f"Expected chunks/ directory and metadata.json file."
            )
        
        # Initialize chunk manager
        self.chunk_manager = ChunkManager(self.output_dir)
        self._total_sequences = None  # Will be computed from metadata
    
    def get_sequence_count(self) -> int:
        """Get total number of processed sequences."""
        if self._total_sequences is None:
            # Calculate from chunk metadata
            existing_sequences = self.chunk_manager.get_existing_sequences()
            self._total_sequences = len(existing_sequences)
        return self._total_sequences
    
    def load_sequence(self, seq_id: int) -> Dict[str, Any]:
        """
        Load all data for a specific sequence ID from chunks.
        
        Args:
            seq_id: Sequence identifier (1-based)
            
        Returns:
            Dictionary containing all sequence data:
            - metadata: JSON metadata including span candidates (if saved)
            - seed_embeddings: H⁰ matrix (T × 512) (if saved)
            - contextual_embeddings: H matrix (T × 512)  
            - soft_probabilities: P matrix (T × |V|) (if saved)
            
        Raises:
            FileNotFoundError: If sequence doesn't exist in chunks
            ValueError: If seq_id is invalid
        """
        if seq_id < 1:
            raise ValueError("Sequence ID must be >= 1")
        
        print(f"Loading sequence {seq_id}...")
        
        # Check if sequence exists
        total_sequences = self.get_sequence_count()
        if seq_id > total_sequences:
            raise ValueError(f"Sequence ID {seq_id} exceeds total sequences {total_sequences}")
        
        # Use the efficient single-sequence loader
        sequence_data = self.chunk_manager.load_single_sequence(seq_id)
        if sequence_data is None:
            raise FileNotFoundError(f"Sequence {seq_id} not found in chunks")
        
        print(f"Sequence {seq_id} loaded successfully")
        
        # Extract sequence text and build metadata if available
        sequence_text = sequence_data.get('sequence', '')
        span_candidates = sequence_data.get('span_candidates', [])
        
        # Build metadata-like structure from available data
        extracted_metadata = {
            'sequence': sequence_text,
            'sequence_length': len(sequence_text) if sequence_text else None,
            'num_candidates': len(span_candidates) if span_candidates else None,
            'span_candidates': span_candidates
        }
        
        # Return the data with standardized keys
        return {
            'metadata': extracted_metadata,
            'seed_embeddings': sequence_data.get('seed_embeddings'),
            'contextual_embeddings': sequence_data.get('contextual_embeddings'),
            'soft_probabilities': sequence_data.get('soft_probabilities')
        }
    
    def get_file_sizes(self, seq_id: int) -> Dict[str, float]:
        """Get file sizes in KB for chunk storage related to a sequence."""
        sizes = {}
        
        # For chunk-based storage, we can estimate the sequence's contribution
        # to the chunk file size, but it's not exact since it's compressed
        chunk_id = self.chunk_manager.get_chunk_id(seq_id)
        chunk_file = self.chunk_manager.get_chunk_file_path(chunk_id)
        
        if chunk_file.exists():
            chunk_size_kb = chunk_file.stat().st_size / 1024
            
            # Load the chunk to see how many sequences it contains
            chunk_data = self.chunk_manager.load_chunk(chunk_id)
            if chunk_data:
                num_sequences_in_chunk = len(chunk_data)
                estimated_size_per_sequence = chunk_size_kb / num_sequences_in_chunk
                sizes['chunk_contribution'] = estimated_size_per_sequence
                sizes['total_chunk_size'] = chunk_size_kb
                sizes['sequences_in_chunk'] = num_sequences_in_chunk
            else:
                sizes['chunk_contribution'] = 0.0
                sizes['total_chunk_size'] = chunk_size_kb
                sizes['sequences_in_chunk'] = 0
        
        # Total is just the estimated contribution
        sizes['total'] = sizes.get('chunk_contribution', 0.0)
        
        return sizes
    
    def get_file_sizes_fast(self, seq_id: int) -> Dict[str, float]:
        """Get file sizes in KB for chunk storage without loading chunk data."""
        sizes = {}
        
        # Just get the chunk file size without loading the data
        chunk_id = self.chunk_manager.get_chunk_id(seq_id)
        chunk_file = self.chunk_manager.get_chunk_file_path(chunk_id)
        
        if chunk_file.exists():
            chunk_size_kb = chunk_file.stat().st_size / 1024
            
            # Estimate based on chunk size (100 sequences per chunk typically)
            estimated_sequences_per_chunk = self.chunk_manager.chunk_size
            estimated_size_per_sequence = chunk_size_kb / estimated_sequences_per_chunk
            
            sizes['chunk_contribution'] = estimated_size_per_sequence
            sizes['total_chunk_size'] = chunk_size_kb
            sizes['sequences_in_chunk'] = estimated_sequences_per_chunk
        else:
            sizes['chunk_contribution'] = 0.0
            sizes['total_chunk_size'] = 0.0
            sizes['sequences_in_chunk'] = 0
        
        # Total is just the estimated contribution
        sizes['total'] = sizes.get('chunk_contribution', 0.0)
        
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
        
        # Get file sizes (but avoid loading chunk again)
        file_sizes = self.get_file_sizes_fast(seq_id)
        
        # Handle missing metadata gracefully
        if metadata:
            analysis = {
                'sequence_id': seq_id,
                'sequence_length': metadata.get('sequence_length', 'Unknown'),
                'num_span_candidates': metadata.get('num_candidates', 'Unknown'),
                'sequence_preview': metadata.get('sequence', 'Not available')[:100] + "..." if metadata.get('sequence') and len(metadata.get('sequence', '')) > 100 else metadata.get('sequence', 'Not available')
            }
        else:
            # Fallback when no JSON metadata available
            context_emb = data['contextual_embeddings']
            if context_emb is not None:
                context_shape = context_emb.shape
                analysis = {
                    'sequence_id': seq_id,
                    'sequence_length': context_shape[0],  # Infer from embedding shape
                    'num_span_candidates': 'Unknown (no metadata)',
                    'sequence_preview': 'Not available (no metadata)'
                }
            else:
                analysis = {
                    'sequence_id': seq_id,
                    'sequence_length': 'Unknown',
                    'num_span_candidates': 'Unknown (no metadata)',
                    'sequence_preview': 'Not available (no metadata)'
                }
        
        # Array shape analysis
        seed_emb = data['seed_embeddings']
        context_emb = data['contextual_embeddings']
        soft_probs = data['soft_probabilities']
        
        seed_shape = seed_emb.shape if seed_emb is not None else None
        context_shape = context_emb.shape if context_emb is not None else None
        soft_prob_shape = soft_probs.shape if soft_probs is not None else None
        
        analysis.update({
            'seed_embeddings_shape': seed_shape,  # Should be (T, 512) or None
            'contextual_embeddings_shape': context_shape,  # Should be (T, 512)
            'soft_probabilities_shape': soft_prob_shape,  # Should be (T, |V|) or None
            'vocabulary_size': soft_prob_shape[1] if soft_prob_shape is not None and len(soft_prob_shape) > 1 else None,
            'seed_embeddings_available': seed_emb is not None,
            'soft_probabilities_available': soft_probs is not None,
            'file_sizes': file_sizes
        })
        
        if detailed:
            # Statistical analysis
            analysis['detailed_stats'] = {}
            
            # Contextual embeddings stats (should always be available)
            if context_emb is not None:
                analysis['detailed_stats']['contextual_embeddings'] = {
                    'mean': float(context_emb.mean()),
                    'std': float(context_emb.std()),
                    'min': float(context_emb.min()),
                    'max': float(context_emb.max()),
                    'sparsity': float((context_emb == 0).mean())
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
            
            print(f"\n== FILE STORAGE (CHUNK-BASED):")
            file_sizes = analysis['file_sizes']
            print(f"   Chunk Contribution: {file_sizes.get('chunk_contribution', 0.0):.1f} KB (estimated)")
            print(f"   Total Chunk Size: {file_sizes.get('total_chunk_size', 0.0):.1f} KB")
            print(f"   Sequences in Chunk: {file_sizes.get('sequences_in_chunk', 0)}")
            print(f"   Storage Type: Compressed chunk (.npz)")
            
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
            if data['contextual_embeddings'] is not None:
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
            else:
                print(f"\n== CONTEXTUAL EMBEDDINGS (H):")
                print(f"   Not available (unexpected - this should always be present)")
            
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
                
                if data['contextual_embeddings'] is not None:
                    print(f"\n--- CONTEXTUAL EMBEDDINGS (COMPLETE) ---")
                    print(data['contextual_embeddings'])
                
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
        help="Path to embedding output directory containing chunks/ subdirectory and metadata.json (chunk-based storage)"
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
