#!/usr/bin/env python3
"""
embedding/span_analysis.py

Advanced span analysis utilities for understanding span candidate patterns,
hierarchies, and coverage statistics in the vocab2embedding pipeline outputs.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import defaultdict, Counter
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpanAnalyzer:
    """
    Comprehensive span analysis toolkit for vocab2embedding pipeline outputs.
    """
    
    def __init__(self, sequence: str, span_candidates: List[Tuple[int, int]]):
        """
        Initialize span analyzer.
        
        Args:
            sequence: Original input sequence
            span_candidates: List of (start, end) span positions
        """
        self.sequence = sequence
        self.span_candidates = span_candidates
        self.sequence_length = len(sequence)
        
        # Validate spans
        self.valid_spans = []
        for start, end in span_candidates:
            if 0 <= start < end <= self.sequence_length:
                self.valid_spans.append((start, end))
            else:
                logger.warning(f"Invalid span ({start}, {end}) for sequence length {self.sequence_length}")
        
        logger.info(f"Initialized SpanAnalyzer: {len(self.valid_spans)} valid spans out of {len(span_candidates)}")
    
    def compute_span_hierarchy(self) -> Dict:
        """
        Analyze hierarchical relationships between spans.
        
        Returns:
            Dictionary containing hierarchy statistics and relationships
        """
        # Find nested relationships
        nested_pairs = []
        contains_map = defaultdict(list)  # parent -> list of children
        contained_by_map = defaultdict(list)  # child -> list of parents
        
        for i, (start1, end1) in enumerate(self.valid_spans):
            for j, (start2, end2) in enumerate(self.valid_spans):
                if i != j:
                    # Check if span j is contained in span i
                    if start1 <= start2 and end2 <= end1:
                        nested_pairs.append((i, j))
                        contains_map[i].append(j)
                        contained_by_map[j].append(i)
        
        # Compute hierarchy statistics
        max_nesting_depth = 0
        span_depths = {}
        
        for i in range(len(self.valid_spans)):
            depth = len(contained_by_map[i])
            span_depths[i] = depth
            max_nesting_depth = max(max_nesting_depth, depth)
        
        # Find root spans (not contained by any other span)
        root_spans = [i for i in range(len(self.valid_spans)) if not contained_by_map[i]]
        
        # Find leaf spans (don't contain any other span)
        leaf_spans = [i for i in range(len(self.valid_spans)) if not contains_map[i]]
        
        return {
            'nested_pairs': nested_pairs,
            'contains_map': dict(contains_map),
            'contained_by_map': dict(contained_by_map),
            'max_nesting_depth': max_nesting_depth,
            'span_depths': span_depths,
            'root_spans': root_spans,
            'leaf_spans': leaf_spans,
            'num_root_spans': len(root_spans),
            'num_leaf_spans': len(leaf_spans),
            'average_depth': np.mean(list(span_depths.values())) if span_depths else 0
        }
    
    def compute_coverage_statistics(self) -> Dict:
        """
        Analyze how well spans cover the input sequence.
        
        Returns:
            Dictionary containing coverage metrics
        """
        # Create coverage array
        coverage = np.zeros(self.sequence_length, dtype=int)
        
        for start, end in self.valid_spans:
            coverage[start:end] += 1
        
        # Coverage statistics
        total_positions = self.sequence_length
        covered_positions = np.sum(coverage > 0)
        uncovered_positions = total_positions - covered_positions
        
        coverage_density = covered_positions / total_positions if total_positions > 0 else 0
        average_coverage = np.mean(coverage)
        max_coverage = np.max(coverage)
        
        # Find gaps (uncovered regions)
        gaps = []
        gap_start = None
        
        for i, cov in enumerate(coverage):
            if cov == 0 and gap_start is None:
                gap_start = i
            elif cov > 0 and gap_start is not None:
                gaps.append((gap_start, i))
                gap_start = None
        
        # Handle gap at end of sequence
        if gap_start is not None:
            gaps.append((gap_start, self.sequence_length))
        
        return {
            'total_positions': total_positions,
            'covered_positions': int(covered_positions),
            'uncovered_positions': int(uncovered_positions),
            'coverage_density': float(coverage_density),
            'average_coverage_depth': float(average_coverage),
            'max_coverage_depth': int(max_coverage),
            'num_gaps': len(gaps),
            'gaps': gaps,
            'coverage_distribution': Counter(coverage.tolist()),
            'coverage_array': coverage.tolist()
        }
    
    def analyze_span_lengths(self) -> Dict:
        """
        Analyze the distribution of span lengths.
        
        Returns:
            Dictionary containing length statistics
        """
        lengths = [end - start for start, end in self.valid_spans]
        
        if not lengths:
            return {'error': 'No valid spans to analyze'}
        
        length_counter = Counter(lengths)
        
        return {
            'num_spans': len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'length_distribution': dict(length_counter),
            'most_common_lengths': length_counter.most_common(10)
        }
    
    def analyze_span_positions(self) -> Dict:
        """
        Analyze the distribution of span start and end positions.
        
        Returns:
            Dictionary containing positional statistics
        """
        starts = [start for start, _ in self.valid_spans]
        ends = [end for _, end in self.valid_spans]
        
        # Position densities
        start_density = np.zeros(self.sequence_length)
        end_density = np.zeros(self.sequence_length)
        
        for start in starts:
            if start < self.sequence_length:
                start_density[start] += 1
        
        for end in ends:
            if end <= self.sequence_length:
                end_density[min(end - 1, self.sequence_length - 1)] += 1
        
        return {
            'start_positions': {
                'mean': np.mean(starts),
                'std': np.std(starts),
                'distribution': Counter(starts),
                'density_array': start_density.tolist()
            },
            'end_positions': {
                'mean': np.mean(ends),
                'std': np.std(ends),
                'distribution': Counter(ends),
                'density_array': end_density.tolist()
            },
            'span_center_positions': {
                'centers': [(start + end) / 2 for start, end in self.valid_spans],
                'mean_center': np.mean([(start + end) / 2 for start, end in self.valid_spans]),
                'std_center': np.std([(start + end) / 2 for start, end in self.valid_spans])
            }
        }
    
    def find_overlapping_spans(self) -> Dict:
        """
        Find all overlapping span pairs and analyze overlap patterns.
        
        Returns:
            Dictionary containing overlap analysis
        """
        overlapping_pairs = []
        overlap_matrix = np.zeros((len(self.valid_spans), len(self.valid_spans)), dtype=bool)
        
        for i, (start1, end1) in enumerate(self.valid_spans):
            for j, (start2, end2) in enumerate(self.valid_spans):
                if i != j:
                    # Check for overlap (not containment)
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    
                    if overlap_start < overlap_end:
                        # There is overlap
                        overlap_length = overlap_end - overlap_start
                        if not (start1 <= start2 and end2 <= end1) and not (start2 <= start1 and end1 <= end2):
                            # Not a containment relationship
                            overlapping_pairs.append({
                                'span1_idx': i,
                                'span2_idx': j,
                                'span1': (start1, end1),
                                'span2': (start2, end2),
                                'overlap_region': (overlap_start, overlap_end),
                                'overlap_length': overlap_length
                            })
                            overlap_matrix[i, j] = True
        
        return {
            'overlapping_pairs': overlapping_pairs,
            'num_overlapping_pairs': len(overlapping_pairs),
            'overlap_matrix': overlap_matrix.tolist(),
            'spans_with_overlaps': list(set([pair['span1_idx'] for pair in overlapping_pairs] + 
                                          [pair['span2_idx'] for pair in overlapping_pairs])),
            'average_overlap_length': np.mean([pair['overlap_length'] for pair in overlapping_pairs]) 
                                    if overlapping_pairs else 0
        }
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate a comprehensive analysis report combining all metrics.
        
        Returns:
            Dictionary containing complete span analysis
        """
        report = {
            'sequence_info': {
                'sequence': self.sequence,
                'sequence_length': self.sequence_length,
                'num_candidate_spans': len(self.span_candidates),
                'num_valid_spans': len(self.valid_spans)
            }
        }
        
        try:
            report['hierarchy'] = self.compute_span_hierarchy()
        except Exception as e:
            report['hierarchy'] = {'error': str(e)}
        
        try:
            report['coverage'] = self.compute_coverage_statistics()
        except Exception as e:
            report['coverage'] = {'error': str(e)}
        
        try:
            report['lengths'] = self.analyze_span_lengths()
        except Exception as e:
            report['lengths'] = {'error': str(e)}
        
        try:
            report['positions'] = self.analyze_span_positions()
        except Exception as e:
            report['positions'] = {'error': str(e)}
        
        try:
            report['overlaps'] = self.find_overlapping_spans()
        except Exception as e:
            report['overlaps'] = {'error': str(e)}
        
        return report


def analyze_span_hierarchy(span_candidates: List[Tuple[int, int]]) -> Dict:
    """
    Standalone function to analyze span hierarchy patterns.
    
    Args:
        span_candidates: List of (start, end) span positions
        
    Returns:
        Dictionary containing hierarchy analysis
    """
    nested_relationships = []
    
    for i, (start1, end1) in enumerate(span_candidates):
        for j, (start2, end2) in enumerate(span_candidates):
            if i != j and start1 <= start2 and end2 <= end1:
                nested_relationships.append((i, j))  # j nested in i
    
    # Build hierarchy tree
    children = defaultdict(list)
    parents = defaultdict(list)
    
    for parent, child in nested_relationships:
        children[parent].append(child)
        parents[child].append(parent)
    
    # Find roots and leaves
    roots = [i for i in range(len(span_candidates)) if i not in parents]
    leaves = [i for i in range(len(span_candidates)) if i not in children]
    
    return {
        'nested_relationships': nested_relationships,
        'children': dict(children),
        'parents': dict(parents),
        'roots': roots,
        'leaves': leaves,
        'max_depth': max([len(parents[i]) for i in range(len(span_candidates))], default=0)
    }


def compute_span_coverage(sequence_length: int, span_candidates: List[Tuple[int, int]]) -> Dict:
    """
    Compute coverage statistics for span candidates over a sequence.
    
    Args:
        sequence_length: Length of the input sequence
        span_candidates: List of (start, end) span positions
        
    Returns:
        Dictionary containing coverage metrics
    """
    coverage = np.zeros(sequence_length, dtype=int)
    
    for start, end in span_candidates:
        if 0 <= start < end <= sequence_length:
            coverage[start:end] += 1
    
    covered_positions = np.sum(coverage > 0)
    coverage_ratio = covered_positions / sequence_length if sequence_length > 0 else 0
    
    return {
        'total_positions': sequence_length,
        'covered_positions': int(covered_positions),
        'coverage_ratio': float(coverage_ratio),
        'average_depth': float(np.mean(coverage)),
        'max_depth': int(np.max(coverage)),
        'coverage_histogram': dict(Counter(coverage.tolist()))
    }


def generate_span_statistics(results_dir: Union[str, Path], 
                           sequence_ids: Optional[List[int]] = None) -> Dict:
    """
    Generate aggregate statistics across multiple span analysis results.
    
    Args:
        results_dir: Directory containing embedding pipeline outputs
        sequence_ids: List of sequence IDs to analyze (None for all)
        
    Returns:
        Dictionary containing aggregate span statistics
    """
    import json
    
    results_dir = Path(results_dir)
    
    if sequence_ids is None:
        # Find all available sequences
        metadata_files = list(results_dir.glob("embedding_*.json"))
        sequence_ids = []
        for f in metadata_files:
            try:
                seq_id = int(f.stem.split('_')[1])
                sequence_ids.append(seq_id)
            except (ValueError, IndexError):
                continue
    
    all_stats = []
    total_spans = 0
    total_sequence_length = 0
    
    for seq_id in sequence_ids:
        try:
            # Load metadata
            metadata_file = results_dir / f"embedding_{seq_id:06d}.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            sequence = metadata['sequence']
            span_candidates = metadata['span_candidates']
            
            # Analyze this sequence
            analyzer = SpanAnalyzer(sequence, span_candidates)
            stats = {
                'sequence_id': seq_id,
                'sequence_length': len(sequence),
                'num_spans': len(span_candidates),
                'hierarchy': analyzer.compute_span_hierarchy(),
                'coverage': analyzer.compute_coverage_statistics(),
                'lengths': analyzer.analyze_span_lengths()
            }
            all_stats.append(stats)
            total_spans += len(span_candidates)
            total_sequence_length += len(sequence)
            
        except Exception as e:
            logger.warning(f"Failed to analyze sequence {seq_id}: {e}")
            continue
    
    if not all_stats:
        return {'error': 'No sequences could be analyzed'}
    
    # Aggregate statistics
    aggregate = {
        'num_sequences': len(all_stats),
        'total_spans': total_spans,
        'total_sequence_length': total_sequence_length,
        'average_spans_per_sequence': total_spans / len(all_stats),
        'average_sequence_length': total_sequence_length / len(all_stats),
        'span_density': total_spans / total_sequence_length if total_sequence_length > 0 else 0,
        
        'coverage_stats': {
            'mean_coverage_ratio': np.mean([s['coverage']['coverage_density'] for s in all_stats]),
            'mean_coverage_depth': np.mean([s['coverage']['average_coverage_depth'] for s in all_stats]),
            'mean_gaps_per_sequence': np.mean([s['coverage']['num_gaps'] for s in all_stats])
        },
        
        'hierarchy_stats': {
            'mean_nesting_depth': np.mean([s['hierarchy']['max_nesting_depth'] for s in all_stats]),
            'mean_root_spans': np.mean([s['hierarchy']['num_root_spans'] for s in all_stats]),
            'mean_leaf_spans': np.mean([s['hierarchy']['num_leaf_spans'] for s in all_stats])
        },
        
        'length_stats': {
            'global_mean_length': np.mean([s['lengths']['mean_length'] for s in all_stats 
                                         if 'mean_length' in s['lengths']]),
            'global_std_length': np.mean([s['lengths']['std_length'] for s in all_stats 
                                        if 'std_length' in s['lengths']])
        }
    }
    
    return aggregate
