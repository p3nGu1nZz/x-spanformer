#!/usr/bin/env python3
"""
embedding/analyze_results.py

Sample script demonstrating how to analyze vocab2embedding pipeline outputs
using the embedding utilities and visualization tools.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.embedding.embedding_utils import (
    load_embedding_results, 
    analyze_embedding_quality,
    batch_analyze_embeddings,
    export_embeddings_to_numpy
)
from x_spanformer.embedding.span_analysis import (
    SpanAnalyzer,
    generate_span_statistics
)
from x_spanformer.embedding.embedding_viz import (
    create_comprehensive_visualization
)


def analyze_single_sequence(results_dir: Path, sequence_id: int) -> None:
    """Analyze a single sequence result comprehensively."""
    print(f"\n🔍 Analyzing Sequence {sequence_id}")
    print("=" * 50)
    
    try:
        # Load results
        result = load_embedding_results(results_dir, sequence_id)
        metadata = result['metadata']
        
        # Basic info
        print(f"Sequence: {metadata['sequence'][:100]}...")
        print(f"Length: {metadata['sequence_length']} characters")
        print(f"Candidates: {metadata['num_candidates']} spans")
        
        # Analyze embeddings quality
        if result['contextual_embeddings'] is not None:
            quality = analyze_embedding_quality(result['contextual_embeddings'])
            print(f"\n📊 Embedding Quality:")
            print(f"  Mean norm: {quality['mean_embedding_norm']:.3f}")
            print(f"  Dimension variance ratio: {quality['dimension_variance_ratio']:.3f}")
            print(f"  Mean pairwise similarity: {quality['mean_pairwise_similarity']:.3f}")
        
        # Analyze spans
        sequence = metadata['sequence']
        span_candidates = metadata['span_candidates']
        analyzer = SpanAnalyzer(sequence, span_candidates)
        
        # Coverage analysis
        coverage = analyzer.compute_coverage_statistics()
        print(f"\n📍 Span Coverage:")
        print(f"  Coverage density: {coverage['coverage_density']:.1%}")
        print(f"  Average depth: {coverage['average_coverage_depth']:.2f}")
        print(f"  Gaps: {coverage['num_gaps']}")
        
        # Length analysis
        lengths = analyzer.analyze_span_lengths()
        if 'mean_length' in lengths:
            print(f"\n📏 Span Lengths:")
            print(f"  Mean length: {lengths['mean_length']:.2f}")
            print(f"  Range: {lengths['min_length']} - {lengths['max_length']}")
            print(f"  Most common: {lengths['most_common_lengths'][:3]}")
        
        # Hierarchy analysis
        hierarchy = analyzer.compute_span_hierarchy()
        print(f"\n🌳 Span Hierarchy:")
        print(f"  Max nesting depth: {hierarchy['max_nesting_depth']}")
        print(f"  Root spans: {hierarchy['num_root_spans']}")
        print(f"  Leaf spans: {hierarchy['num_leaf_spans']}")
        
    except Exception as e:
        print(f"❌ Error analyzing sequence {sequence_id}: {e}")


def analyze_batch(results_dir: Path, max_sequences: int = 10) -> None:
    """Analyze multiple sequences and show aggregate statistics."""
    print(f"\n📈 Batch Analysis (up to {max_sequences} sequences)")
    print("=" * 60)
    
    try:
        # Find available sequences
        metadata_files = list(results_dir.glob("embedding_*.json"))
        sequence_ids = []
        for f in metadata_files:
            try:
                seq_id = int(f.stem.split('_')[1])
                sequence_ids.append(seq_id)
            except (ValueError, IndexError):
                continue
        
        sequence_ids = sorted(sequence_ids)[:max_sequences]
        print(f"Found {len(sequence_ids)} sequences to analyze")
        
        # Batch embedding analysis
        embedding_stats = batch_analyze_embeddings(results_dir, sequence_ids)
        if 'error' not in embedding_stats:
            print(f"\n🧠 Embedding Statistics:")
            print(f"  Sequences: {embedding_stats['num_sequences']}")
            print(f"  Mean sequence length: {embedding_stats['mean_sequence_length']:.1f}")
            print(f"  Embedding dimension: {embedding_stats['embedding_dim']}")
            print(f"  Global mean norm: {embedding_stats['global_mean_norm']:.3f}")
            print(f"  Mean dimension variance: {embedding_stats['mean_dimension_variance']:.6f}")
            
            if embedding_stats['seed_context_similarity']['mean'] is not None:
                print(f"  Seed-Context similarity: {embedding_stats['seed_context_similarity']['mean']:.3f}")
        
        # Batch span analysis
        span_stats = generate_span_statistics(results_dir, sequence_ids)
        if 'error' not in span_stats:
            print(f"\n📍 Span Statistics:")
            print(f"  Total spans: {span_stats['total_spans']}")
            print(f"  Spans per sequence: {span_stats['average_spans_per_sequence']:.1f}")
            print(f"  Span density: {span_stats['span_density']:.3f} spans/char")
            print(f"  Mean coverage ratio: {span_stats['coverage_stats']['mean_coverage_ratio']:.1%}")
            print(f"  Mean nesting depth: {span_stats['hierarchy_stats']['mean_nesting_depth']:.2f}")
    
    except Exception as e:
        print(f"❌ Error in batch analysis: {e}")


def create_visualizations(results_dir: Path, sequence_id: int, output_dir: Optional[Path] = None) -> None:
    """Create comprehensive visualizations for a sequence."""
    print(f"\n🎨 Creating Visualizations for Sequence {sequence_id}")
    print("=" * 55)
    
    try:
        if output_dir:
            output_dir.mkdir(exist_ok=True)
        
        figures = create_comprehensive_visualization(results_dir, sequence_id, output_dir)
        
        print(f"Created {len(figures)} visualizations:")
        for plot_name in figures:
            print(f"  ✅ {plot_name}")
        
        if output_dir:
            print(f"📁 Saved to: {output_dir}")
        else:
            print("📊 Figures created in memory (use --output-dir to save)")
    
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


def export_data(results_dir: Path, output_path: Path, embedding_type: str = 'contextual') -> None:
    """Export embeddings to numpy format."""
    print(f"\n💾 Exporting {embedding_type} embeddings")
    print("=" * 40)
    
    try:
        export_embeddings_to_numpy(results_dir, output_path, embedding_type)
        print(f"✅ Exported to: {output_path}")
    
    except Exception as e:
        print(f"❌ Error exporting: {e}")


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze vocab2embedding pipeline outputs"
    )
    
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing embedding pipeline outputs"
    )
    
    parser.add_argument(
        "--sequence-id",
        type=int,
        help="Analyze specific sequence ID"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch analysis across multiple sequences"
    )
    
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=10,
        help="Maximum sequences for batch analysis"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations (requires sequence-id)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save visualizations"
    )
    
    parser.add_argument(
        "--export",
        type=str,
        choices=['contextual', 'seed', 'soft_probs'],
        help="Export embeddings to numpy format"
    )
    
    parser.add_argument(
        "--export-path",
        type=Path,
        help="Path for exported numpy file"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Results directory not found: {args.results_dir}")
        return
    
    print(f"🚀 X-Spanformer Embedding Analysis")
    print(f"📂 Results directory: {args.results_dir}")
    
    # Single sequence analysis
    if args.sequence_id is not None:
        analyze_single_sequence(args.results_dir, args.sequence_id)
        
        if args.visualize:
            create_visualizations(args.results_dir, args.sequence_id, args.output_dir)
    
    # Batch analysis
    if args.batch:
        analyze_batch(args.results_dir, args.max_sequences)
    
    # Export embeddings
    if args.export:
        if args.export_path is None:
            args.export_path = args.results_dir / f"{args.export}_embeddings.npz"
        export_data(args.results_dir, args.export_path, args.export)
    
    # Default behavior if no specific action
    if (args.sequence_id is None and not args.batch and args.export is None):
        print("\n💡 No specific action requested. Running sample batch analysis...")
        analyze_batch(args.results_dir, 5)


if __name__ == "__main__":
    main()
