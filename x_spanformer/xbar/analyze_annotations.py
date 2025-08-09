"""
Clean version of annotation analyzer with logging integration.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List

from .xbar_map import XBarLabelMap

# Configure logger
logger = logging.getLogger(__name__)


class AnnotationAnalyzer:
    """Clean annotation analyzer with logging integration."""
    
    def __init__(self, annotations_file: str):
        self.annotations_file = annotations_file
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

    def analyze_and_report(self) -> Dict[str, Any]:
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
        
        logger.info("Label validation:")
        logger.info(f"  Valid labels: {valid_labels} ({valid_labels/len(self.annotations)*100:.1f}%)")
        if invalid_labels > 0:
            logger.info(f"  Invalid labels: {invalid_labels} ({invalid_labels/len(self.annotations)*100:.1f}%)")
        
        logger.info("Top labels (normalized):")
        for label, count in normalized_label_counts.most_common(5):
            percentage = count / len(self.annotations) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
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


def analyze_annotations(annotations_file: str = "data/annotations/annotations.jsonl") -> Dict[str, Any]:
    """Run annotation analysis and return summary."""
    analyzer = AnnotationAnalyzer(annotations_file)
    return analyzer.analyze_and_report()

