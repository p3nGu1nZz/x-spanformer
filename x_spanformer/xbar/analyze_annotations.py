"""
Clean version of annotation analyzer with logging integration.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional

from .xbar_map import XBarLabelMap

# Configure logger
logger = logging.getLogger(__name__)


class AnnotationAnalyzer:
    """Clean annotation analyzer with logging integration."""
    
    def __init__(self, spans_file: str = "data/annotations/spans.jsonl"):
        """
        Initialize analyzer with spans file.
        
        Args:
            spans_file: Path to spans.jsonl (position-based format)
        """
        self.annotations_file = spans_file
        self.annotations = []
    
    @staticmethod
    def _get_hierarchical_categories() -> Dict[str, List[str]]:
        """Get hierarchical categorization of X-bar labels based on linguistic theory."""
        # Word-level spans (X-bar terminals): Individual lexical items
        word_level = []
        
        # Natural language terminals
        word_level.extend(['noun', 'verb', 'adjective', 'adverb', 'determiner', 
                          'preposition', 'pronoun', 'conjunction', 'punctuation'])
        
        # Code terminals  
        word_level.extend(['keyword', 'identifier', 'operator', 'literal', 
                          'delimiter', 'type_name', 'comment'])
        
        # Mixed domain terminals
        word_level.append('inline_code')
        
        # Phrase-level spans (X-bar intermediate projections): Multi-word constituents
        phrase_level = []
        
        # Natural language phrases
        phrase_level.extend(['noun_phrase', 'verb_phrase', 'adjective_phrase', 
                            'adverb_phrase', 'prepositional_phrase'])
        
        # Code phrases
        phrase_level.extend(['expression', 'function_call', 'assignment', 
                            'parameter_list', 'argument_list'])
        
        # Mixed domain phrases
        phrase_level.extend(['code_block', 'documentation_comment', 'api_reference'])
        
        # Clause-level spans (X-bar maximal projections): Complete constructions
        clause_level = []
        
        # Natural language clauses
        clause_level.extend(['main_clause', 'subordinate_clause', 'relative_clause'])
        
        # Code clauses
        clause_level.extend(['if_statement', 'loop_statement', 'function_definition',
                            'class_definition', 'import_statement', 'return_statement'])
        
        return {
            'word_level': word_level,
            'phrase_level': phrase_level,
            'clause_level': clause_level
        }
    
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations from JSONL file."""
        annotations_path = Path(self.annotations_file)
        if not annotations_path.exists():
            logger.error(f"Annotations file not found: {annotations_path}")
            return []
        
        annotations = []
        with open(annotations_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        annotations.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse annotation line: {e}")
        
        self.annotations = annotations
        logger.info(f"Loaded {len(annotations)} annotations from {annotations_path}")
        return annotations
    
    def analyze_hierarchical_structure(self) -> Dict[str, Any]:
        """Analyze hierarchical X-bar structure and return summary statistics."""
        by_sequence = defaultdict(list)
        for ann in self.annotations:
            by_sequence[ann['sequence_number']].append(ann)
        
        # Get hierarchical categories from centralized mapping
        categories = self._get_hierarchical_categories()
        
        hierarchy_analysis = {}
        for seq_num, seq_anns in by_sequence.items():
            # Normalize labels and categorize spans
            word_level = [a for a in seq_anns 
                         if XBarLabelMap.normalize_label(a['xbar_label']) in categories['word_level']]
            
            phrase_level = [a for a in seq_anns 
                           if XBarLabelMap.normalize_label(a['xbar_label']) in categories['phrase_level']]
            
            clause_level = [a for a in seq_anns 
                           if XBarLabelMap.normalize_label(a['xbar_label']) in categories['clause_level']]
            
            hierarchy_analysis[seq_num] = {
                'word_level': len(word_level),
                'phrase_level': len(phrase_level),
                'clause_level': len(clause_level),
                'total': len(seq_anns)
            }
        
        return hierarchy_analysis

    def find_anomalies(self) -> list:
        """Find potential anomalies in span annotations."""
        anomalies = []
        
        # Get all valid labels from XBarLabelMap
        all_valid_labels = set()
        all_valid_labels.update(XBarLabelMap.NATURAL_LABELS.keys())
        all_valid_labels.update(XBarLabelMap.CODE_LABELS.keys())
        all_valid_labels.update(XBarLabelMap.MIXED_LABELS.keys())
        
        # Group by sequence
        by_sequence = defaultdict(list)
        for ann in self.annotations:
            by_sequence[ann['sequence_number']].append(ann)
        
        for seq_num, seq_anns in by_sequence.items():
            # Check for invalid labels
            for ann in seq_anns:
                normalized_label = XBarLabelMap.normalize_label(ann['xbar_label'])
                if normalized_label not in all_valid_labels and normalized_label != 'unknown':
                    anomalies.append({
                        'type': 'invalid_label',
                        'sequence': seq_num,
                        'span': ann,
                        'normalized_label': normalized_label,
                        'severity': 'medium'
                    })
            
            # Check for exact duplicates
            seen_exact = set()
            for ann in seq_anns:
                exact_key = (ann['start_pos'], ann['end_pos'], ann['text'], ann['xbar_label'])
                if exact_key in seen_exact:
                    anomalies.append({
                        'type': 'exact_duplicate',
                        'sequence': seq_num,
                        'span': ann,
                        'severity': 'high'
                    })
                seen_exact.add(exact_key)
            
            # Check for boundary duplicates (same position + text, different labels)
            boundary_groups = defaultdict(list)
            for ann in seq_anns:
                boundary_key = (ann['start_pos'], ann['end_pos'], ann['text'])
                boundary_groups[boundary_key].append(ann)
            
            for boundary_key, duplicate_anns in boundary_groups.items():
                if len(duplicate_anns) > 1:
                    labels = set(ann['xbar_label'] for ann in duplicate_anns)
                    if len(labels) > 1:
                        anomalies.append({
                            'type': 'boundary_duplicate',
                            'sequence': seq_num,
                            'text': boundary_key[2],
                            'positions': (boundary_key[0], boundary_key[1]),
                            'labels': list(labels),
                            'count': len(duplicate_anns),
                            'severity': 'medium'
                        })
        
        return anomalies

    def analyze_overlap_patterns(self) -> Dict[str, Any]:
        """Analyze span overlap patterns and coverage."""
        by_sequence = defaultdict(list)
        for ann in self.annotations:
            by_sequence[ann['sequence_number']].append(ann)
        
        total_overlaps = 0
        total_coverage_chars = 0
        total_text_chars = 0
        overlap_details = []
        
        for seq_num, seq_anns in by_sequence.items():
            # Sort spans by start position
            sorted_spans = sorted(seq_anns, key=lambda x: x['start_pos'])
            
            # Find overlaps within sequence
            seq_overlaps = 0
            for i, span1 in enumerate(sorted_spans):
                for span2 in sorted_spans[i+1:]:
                    # Check if spans overlap
                    if (span1['start_pos'] < span2['end_pos'] and 
                        span2['start_pos'] < span1['end_pos']):
                        seq_overlaps += 1
                        overlap_details.append({
                            'sequence': seq_num,
                            'span1': span1,
                            'span2': span2,
                            'overlap_chars': min(span1['end_pos'], span2['end_pos']) - 
                                           max(span1['start_pos'], span2['start_pos'])
                        })
            
            total_overlaps += seq_overlaps
            
            # Calculate coverage for this sequence
            if seq_anns:
                raw_text = seq_anns[0]['raw']
                total_text_chars += len(raw_text)
                
                # Calculate unique covered characters
                covered_positions = set()
                for span in seq_anns:
                    covered_positions.update(range(span['start_pos'], span['end_pos']))
                total_coverage_chars += len(covered_positions)
        
        coverage_rate = total_coverage_chars / total_text_chars if total_text_chars > 0 else 0
        
        return {
            'total_overlaps': total_overlaps,
            'overlap_rate': total_overlaps / len(self.annotations) if self.annotations else 0,
            'coverage_rate': coverage_rate,
            'overlap_details': overlap_details[:10]  # Sample of overlaps
        }
    
    def analyze_domain_distribution(self) -> Dict[str, Any]:
        """Analyze distribution by domain type."""
        by_sequence = defaultdict(list)
        for ann in self.annotations:
            by_sequence[ann['sequence_number']].append(ann)
        
        domain_stats = defaultdict(lambda: {'sequences': 0, 'spans': 0, 'avg_spans': 0.0})
        
        for seq_num, seq_anns in by_sequence.items():
            if seq_anns:
                # Get domain from first annotation in sequence
                domain = seq_anns[0].get('domain_type', 'unknown')
                domain_stats[domain]['sequences'] += 1
                domain_stats[domain]['spans'] += len(seq_anns)
        
        # Calculate averages
        for domain, stats in domain_stats.items():
            if stats['sequences'] > 0:
                stats['avg_spans'] = stats['spans'] / stats['sequences']
        
        return dict(domain_stats)
    
    def analyze_consistency_patterns(self) -> Dict[str, Any]:
        """Analyze label consistency for identical text spans."""
        text_to_labels = defaultdict(set)
        text_counts = defaultdict(int)
        
        for ann in self.annotations:
            text = ann['text'].strip().lower()
            if len(text) > 1:  # Skip single characters
                text_to_labels[text].add(XBarLabelMap.normalize_label(ann['xbar_label']))
                text_counts[text] += 1
        
        # Find inconsistent labeling
        inconsistent_texts = []
        consistent_texts = 0
        
        for text, labels in text_to_labels.items():
            if len(labels) > 1 and text_counts[text] > 1:
                inconsistent_texts.append({
                    'text': text,
                    'labels': list(labels),
                    'count': text_counts[text]
                })
            elif text_counts[text] > 1:
                consistent_texts += 1
        
        # Sort by frequency
        inconsistent_texts.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'consistent_texts': consistent_texts,
            'inconsistent_texts': len(inconsistent_texts),
            'consistency_rate': consistent_texts / (consistent_texts + len(inconsistent_texts)) if (consistent_texts + len(inconsistent_texts)) > 0 else 1.0,
            'top_inconsistencies': inconsistent_texts[:10]
        }
    
    def analyze_working_files(self, output_dir: str) -> Dict[str, Any]:
        """Analyze working files for processing patterns and performance."""
        working_dir = Path(output_dir) / "working"
        if not working_dir.exists():
            return {'error': 'Working directory not found'}
        
        working_files = list(working_dir.glob("*.json"))
        processing_times = []
        error_patterns = defaultdict(int)
        domain_performance = defaultdict(lambda: {'total': 0, 'successful': 0, 'success_rate': 0.0})
        zero_span_sequences = 0
        
        for working_file in working_files:
            try:
                with open(working_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                domain = data.get('domain_type', 'unknown')
                domain_performance[domain]['total'] += 1
                
                if data.get('status') == 'completed':
                    span_count = len(data.get('span_annotations', []))
                    if span_count == 0:
                        zero_span_sequences += 1
                        # These should be treated as needing retry
                        error_patterns['zero_spans'] += 1
                    else:
                        domain_performance[domain]['successful'] += 1
                    # Could extract processing time if available in agent_metadata
                elif data.get('error_message'):
                    error_msg = data['error_message']
                    # Categorize error types
                    if 'timeout' in error_msg.lower():
                        error_patterns['timeout'] += 1
                    elif 'json' in error_msg.lower():
                        error_patterns['json_parse'] += 1
                    elif 'connection' in error_msg.lower():
                        error_patterns['connection'] += 1
                    else:
                        error_patterns['other'] += 1
                        
            except Exception as e:
                error_patterns['file_read_error'] += 1
        
        # Calculate success rates by domain
        for domain, stats in domain_performance.items():
            if stats['total'] > 0:
                stats['success_rate'] = stats['successful'] / stats['total']
        
        return {
            'total_working_files': len(working_files),
            'zero_span_sequences': zero_span_sequences,
            'error_patterns': dict(error_patterns),
            'domain_performance': dict(domain_performance)
        }

    def analyze_and_report(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis and return summary statistics."""
        if not self.annotations:
            self.load_annotations()
        
        # Basic statistics
        sequences = set(ann['sequence_number'] for ann in self.annotations)
        min_seq = min(sequences) if sequences else 0
        max_seq = max(sequences) if sequences else 0
        
        # Hierarchical analysis
        hierarchy = self.analyze_hierarchical_structure()
        total_word_spans = sum(h['word_level'] for h in hierarchy.values())
        total_phrase_spans = sum(h['phrase_level'] for h in hierarchy.values())
        total_clause_spans = sum(h['clause_level'] for h in hierarchy.values())
        total_sequences = len(hierarchy)
        
        # Label distribution (with normalization)
        raw_label_counts = Counter(ann['xbar_label'] for ann in self.annotations)
        normalized_label_counts = Counter(XBarLabelMap.normalize_label(ann['xbar_label']) 
                                        for ann in self.annotations)
        
        # Label validation statistics
        all_valid_labels = set()
        all_valid_labels.update(XBarLabelMap.NATURAL_LABELS.keys())
        all_valid_labels.update(XBarLabelMap.CODE_LABELS.keys())
        all_valid_labels.update(XBarLabelMap.MIXED_LABELS.keys())
        
        valid_labels = sum(1 for ann in self.annotations 
                          if XBarLabelMap.normalize_label(ann['xbar_label']) in all_valid_labels)
        invalid_labels = len(self.annotations) - valid_labels
        
        # Span length statistics
        span_lengths = [len(ann['text']) for ann in self.annotations]
        avg_length = sum(span_lengths) / len(span_lengths) if span_lengths else 0
        min_length = min(span_lengths) if span_lengths else 0
        max_length = max(span_lengths) if span_lengths else 0
        
        # Enhanced analysis
        overlap_analysis = self.analyze_overlap_patterns()
        domain_analysis = self.analyze_domain_distribution()
        consistency_analysis = self.analyze_consistency_patterns()
        
        # Working files analysis (if output_dir provided)
        working_analysis = None
        if output_dir:
            working_analysis = self.analyze_working_files(output_dir)
        
        # Anomaly detection
        anomalies = self.find_anomalies()
        anomaly_counts = Counter(a.get('severity', 'unknown') for a in anomalies)
        
        # Boundary alignment check
        boundary_issues = 0
        for ann in self.annotations:
            raw_text = ann['raw']
            start, end = ann['start_pos'], ann['end_pos']
            if start < len(raw_text) and end <= len(raw_text):
                actual_text = raw_text[start:end]
                if actual_text != ann['text']:
                    boundary_issues += 1
        
        # Log clean summary
        logger.info("=" * 40)
        logger.info("ANNOTATION ANALYSIS SUMMARY")
        logger.info("=" * 40)
        logger.info(f"Total annotations: {len(self.annotations)}")
        logger.info(f"Sequences analyzed: {total_sequences} (seq {min_seq} to {max_seq})")
        
        logger.info("Hierarchical distribution:")
        logger.info(f"  Word-level spans: {total_word_spans} ({total_word_spans/len(self.annotations)*100:.1f}%)")
        logger.info(f"  Phrase-level spans: {total_phrase_spans} ({total_phrase_spans/len(self.annotations)*100:.1f}%)")
        logger.info(f"  Clause-level spans: {total_clause_spans} ({total_clause_spans/len(self.annotations)*100:.1f}%)")
        
        logger.info("Span statistics:")
        logger.info(f"  Average spans per sequence: {len(self.annotations)/total_sequences:.1f}")
        logger.info(f"  Average span length: {avg_length:.1f} characters")
        logger.info(f"  Span length range: {min_length}-{max_length} characters")
        
        # Enhanced metrics logging
        logger.info("Coverage and overlap analysis:")
        logger.info(f"  Text coverage rate: {overlap_analysis['coverage_rate']*100:.1f}%")
        logger.info(f"  Span overlaps: {overlap_analysis['total_overlaps']} ({overlap_analysis['overlap_rate']*100:.1f}% of spans)")
        
        logger.info("Domain distribution:")
        for domain, stats in domain_analysis.items():
            logger.info(f"  {domain}: {stats['sequences']} sequences, {stats['spans']} spans, {stats['avg_spans']:.1f} avg/seq")
        
        logger.info("Label consistency:")
        logger.info(f"  Consistent texts: {consistency_analysis['consistent_texts']}")
        logger.info(f"  Inconsistent texts: {consistency_analysis['inconsistent_texts']}")
        logger.info(f"  Consistency rate: {consistency_analysis['consistency_rate']*100:.1f}%")
        if consistency_analysis['top_inconsistencies']:
            logger.info("  Top inconsistencies:")
            for item in consistency_analysis['top_inconsistencies'][:3]:
                logger.info(f"    '{item['text']}': {item['labels']} ({item['count']} occurrences)")
        
        logger.info("Label validation:")
        logger.info(f"  Valid labels: {valid_labels} ({valid_labels/len(self.annotations)*100:.1f}%)")
        if invalid_labels > 0:
            logger.info(f"  Invalid labels: {invalid_labels} ({invalid_labels/len(self.annotations)*100:.1f}%)")
        
        logger.info("Top labels (normalized):")
        for label, count in normalized_label_counts.most_common(5):
            percentage = count / len(self.annotations) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Working files analysis (if available)
        if working_analysis and 'error' not in working_analysis:
            logger.info("Working files analysis:")
            logger.info(f"  Total working files: {working_analysis['total_working_files']}")
            if working_analysis.get('zero_span_sequences', 0) > 0:
                logger.info(f"  Zero-span sequences: {working_analysis['zero_span_sequences']} (will be retried)")
            if working_analysis['error_patterns']:
                logger.info("  Error patterns:")
                for error_type, count in working_analysis['error_patterns'].items():
                    logger.info(f"    {error_type}: {count}")
            if working_analysis['domain_performance']:
                logger.info("  Domain performance:")
                for domain, stats in working_analysis['domain_performance'].items():
                    logger.info(f"    {domain}: {stats['successful']}/{stats['total']} ({stats['success_rate']*100:.1f}%)")
        
        if anomalies:
            logger.info(f"Anomalies detected: {len(anomalies)} total")
            for severity, count in anomaly_counts.items():
                logger.info(f"  {severity}: {count}")
        else:
            logger.info("No anomalies detected")
            
        if boundary_issues == 0:
            logger.info("Boundary alignment: Perfect (100% accurate)")
        else:
            logger.info(f"Boundary alignment: {boundary_issues} misalignments found")
        
        # Return summary data
        return {
            'total_annotations': len(self.annotations),
            'total_sequences': total_sequences,
            'sequence_range': (min_seq, max_seq),
            'hierarchical_stats': {
                'word_level': total_word_spans,
                'phrase_level': total_phrase_spans, 
                'clause_level': total_clause_spans
            },
            'span_stats': {
                'avg_per_sequence': len(self.annotations) / total_sequences if total_sequences > 0 else 0,
                'avg_length': avg_length,
                'length_range': (min_length, max_length)
            },
            'overlap_analysis': overlap_analysis,
            'domain_analysis': domain_analysis,
            'consistency_analysis': consistency_analysis,
            'working_analysis': working_analysis,
            'label_validation': {
                'valid_labels': valid_labels,
                'invalid_labels': invalid_labels,
                'validation_rate': valid_labels / len(self.annotations) if self.annotations else 0
            },
            'top_labels_raw': dict(raw_label_counts.most_common(5)),
            'top_labels_normalized': dict(normalized_label_counts.most_common(5)),
            'anomalies': {
                'total': len(anomalies),
                'by_severity': dict(anomaly_counts)
            },
            'boundary_issues': boundary_issues
        }


def analyze_annotations(spans_file: str = "data/annotations/spans.jsonl") -> Dict[str, Any]:
    """Run annotation analysis and return summary."""
    analyzer = AnnotationAnalyzer(spans_file)
    return analyzer.analyze_and_report()

