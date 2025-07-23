#!/usr/bin/env python3
"""
embedding/vocab2embedding_analysis.py

Analysis utilities specifically for vocab2embedding pipeline outputs.
Provides comprehensive analysis of embeddings, span candidates, and pipeline quality.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import logging
import argparse

import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class Vocab2EmbeddingAnalyzer:
    """
    Comprehensive analyzer for vocab2embedding pipeline outputs.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize analyzer with pipeline output directory.
        
        Args:
            output_dir: Directory containing pipeline outputs
        """
        self.output_dir = Path(output_dir)
        self.results = {}
        self._load_all_results()
    
    def _load_all_results(self):
        """Load all embedding results from output directory."""
        logger.info(f"Loading results from {self.output_dir}")
        
        for json_file in self.output_dir.glob('embedding_*.json'):
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                seq_id = metadata['sequence_id']
                
                # Load corresponding numpy arrays
                soft_probs_file = self.output_dir / f"soft_probs_{seq_id:06d}.npy"
                seed_emb_file = self.output_dir / f"seed_emb_{seq_id:06d}.npy"
                context_emb_file = self.output_dir / f"context_emb_{seq_id:06d}.npy"
                
                if all(f.exists() for f in [soft_probs_file, seed_emb_file, context_emb_file]):
                    self.results[seq_id] = {
                        'metadata': metadata,
                        'soft_probs': np.load(soft_probs_file),
                        'seed_embeddings': np.load(seed_emb_file),
                        'contextual_embeddings': np.load(context_emb_file)
                    }
                else:
                    logger.warning(f"Missing numpy files for sequence {seq_id}")
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.results)} complete result sets")
    
    def analyze_pipeline_quality(self) -> Dict:
        """
        Comprehensive pipeline quality analysis.
        
        Returns:
            Dictionary with quality metrics
        """
        if not self.results:
            return {'error': 'No results loaded'}
        
        analysis = {
            'sequence_count': len(self.results),
            'embedding_stats': {},
            'candidate_stats': {},
            'span_coverage': {},
            'probability_health': {}
        }
        
        # Aggregate statistics
        all_candidate_densities = []
        all_sequence_lengths = []
        all_embedding_norms = []
        all_max_probs = []
        
        for seq_id, data in self.results.items():
            metadata = data['metadata']
            soft_probs = data['soft_probs']
            context_emb = data['contextual_embeddings']
            
            # Candidate density
            density = metadata['num_candidates'] / metadata['sequence_length']
            all_candidate_densities.append(density)
            all_sequence_lengths.append(metadata['sequence_length'])
            
            # Embedding quality
            emb_norms = np.linalg.norm(context_emb, axis=1)
            all_embedding_norms.extend(emb_norms.tolist())
            
            # Probability health
            max_probs = np.max(soft_probs, axis=1)
            all_max_probs.extend(max_probs.tolist())
        
        # Summary statistics
        analysis['candidate_stats'] = {
            'mean_density': np.mean(all_candidate_densities),
            'std_density': np.std(all_candidate_densities),
            'min_density': np.min(all_candidate_densities),
            'max_density': np.max(all_candidate_densities)
        }
        
        analysis['embedding_stats'] = {
            'mean_norm': np.mean(all_embedding_norms),
            'std_norm': np.std(all_embedding_norms),
            'dimension_variance': np.var([np.var(data['contextual_embeddings'], axis=0).mean() 
                                        for data in self.results.values()])
        }
        
        analysis['probability_health'] = {
            'mean_max_prob': np.mean(all_max_probs),
            'low_confidence_rate': np.mean(np.array(all_max_probs) < 0.1),
            'high_confidence_rate': np.mean(np.array(all_max_probs) > 0.8)
        }
        
        return analysis
    
    def visualize_span_candidates(self, sequence_id: int, save_path: Optional[Path] = None):
        """
        Create visualization of span candidates for a specific sequence.
        
        Args:
            sequence_id: ID of sequence to visualize
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if sequence_id not in self.results:
            raise ValueError(f"Sequence {sequence_id} not found in results")
        
        data = self.results[sequence_id]
        metadata = data['metadata']
        sequence = metadata['sequence']
        candidates = metadata['span_candidates']
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot sequence as horizontal bar
        ax.barh(0, len(sequence), height=0.1, color='lightgray', alpha=0.5, label='Sequence')
        
        # Plot span candidates with different colors by length
        span_lengths = [end - start for start, end in candidates]
        unique_lengths = sorted(set(span_lengths))
        
        # Use a simple color cycle instead of tab10
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_lengths)))
        length_to_color = {length: color for length, color in zip(unique_lengths, colors)}
        
        for i, (start, end) in enumerate(candidates):
            y_pos = 0.2 + (i % 20) * 0.05  # Stack spans vertically
            width = end - start
            color = length_to_color[width]
            ax.barh(y_pos, width, left=start, height=0.03, 
                   alpha=0.7, color=color, label=f'Length {width}' if i == 0 else '')
        
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Span Candidates')
        ax.set_title(f'Span Candidates for Sequence {sequence_id} (length={len(sequence)})')
        
        # Add text showing first part of sequence
        text_preview = sequence[:50] + "..." if len(sequence) > 50 else sequence
        ax.text(0.02, 0.98, f"Text: {text_preview}", transform=ax.transAxes, 
               verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_summary_report(self, output_path: Path):
        """
        Export comprehensive summary report to JSON.
        
        Args:
            output_path: Path for output JSON file
        """
        report = {
            'pipeline_analysis': self.analyze_pipeline_quality(),
            'sequence_details': {}
        }
        
        # Add per-sequence details
        for seq_id, data in self.results.items():
            metadata = data['metadata']
            report['sequence_details'][seq_id] = {
                'sequence_length': metadata['sequence_length'],
                'num_candidates': metadata['num_candidates'],
                'candidate_density': metadata['num_candidates'] / metadata['sequence_length'],
                'soft_probs_shape': data['soft_probs'].shape,
                'seed_embeddings_shape': data['seed_embeddings'].shape,
                'contextual_embeddings_shape': data['contextual_embeddings'].shape,
                'mean_embedding_norm': np.linalg.norm(data['contextual_embeddings'], axis=1).mean(),
                'max_probability_per_position': np.max(data['soft_probs'], axis=1).mean()
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=float)
        
        logger.info(f"Summary report exported to {output_path}")


def analyze_embeddings_batch(output_dir: Union[str, Path], 
                           export_report: bool = True,
                           create_visualizations: bool = True) -> Dict:
    """
    Batch analysis of vocab2embedding pipeline outputs.
    
    Args:
        output_dir: Directory containing pipeline outputs
        export_report: Whether to export summary report
        create_visualizations: Whether to create visualization plots
        
    Returns:
        Analysis results dictionary
    """
    analyzer = Vocab2EmbeddingAnalyzer(output_dir)
    
    if not analyzer.results:
        logger.error("No results found for analysis")
        return {'error': 'No results found'}
    
    # Run comprehensive analysis
    analysis = analyzer.analyze_pipeline_quality()
    
    # Export report if requested
    if export_report:
        report_path = analyzer.output_dir / 'analysis_report.json'
        analyzer.export_summary_report(report_path)
    
    # Create visualizations for first few sequences
    if create_visualizations and analyzer.results:
        viz_dir = analyzer.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Visualize first 5 sequences
        sequence_ids = sorted(analyzer.results.keys())[:5]
        for seq_id in sequence_ids:
            try:
                fig = analyzer.visualize_span_candidates(seq_id)
                fig.savefig(viz_dir / f'spans_sequence_{seq_id:06d}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating visualization for sequence {seq_id}: {e}")
        
        logger.info(f"Created visualizations in {viz_dir}")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze vocab2embedding pipeline outputs")
    parser.add_argument("output_dir", help="Directory containing pipeline outputs")
    parser.add_argument("--no-report", action="store_true", help="Skip report export")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run analysis
    results = analyze_embeddings_batch(
        args.output_dir,
        export_report=not args.no_report,
        create_visualizations=not args.no_viz
    )
    
    print("\nAnalysis Summary:")
    print(f"Sequences analyzed: {results.get('sequence_count', 0)}")
    if 'candidate_stats' in results:
        print(f"Mean candidate density: {results['candidate_stats']['mean_density']:.3f}")
    if 'embedding_stats' in results:
        print(f"Mean embedding norm: {results['embedding_stats']['mean_norm']:.3f}")
