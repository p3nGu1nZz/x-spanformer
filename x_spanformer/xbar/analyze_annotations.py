#!/usr/bin/env python3
"""
Span Annotation Analysis Tool
Analyzes span annotations against the theoretical framework from Section 3.3 of the paper.
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

def load_annotations(file_path):
    """Load annotations from JSONL file."""
    annotations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations

def analyze_overlapping_spans(annotations):
    """Analyze overlapping spans - should allow multiple labels per position."""
    overlaps = defaultdict(list)
    position_coverage = defaultdict(set)
    
    for ann in annotations:
        seq_num = ann['sequence_number']
        start, end = ann['start_pos'], ann['end_pos']
        label = ann['xbar_label']
        text = ann['text']
        
        # Track position coverage
        for pos in range(start, end):
            position_coverage[seq_num].add(pos)
        
        # Check for overlaps with other spans
        for other in annotations:
            if (other['sequence_number'] == seq_num and 
                other != ann and
                not (other['end_pos'] <= start or other['start_pos'] >= end)):
                overlaps[seq_num].append((ann, other))
    
    return overlaps, position_coverage

def analyze_hierarchical_structure(annotations):
    """Analyze if annotations follow hierarchical X-bar structure."""
    by_sequence = defaultdict(list)
    for ann in annotations:
        by_sequence[ann['sequence_number']].append(ann)
    
    hierarchy_analysis = {}
    for seq_num, seq_anns in by_sequence.items():
        word_level = [a for a in seq_anns if a['xbar_label'] in 
                     ['noun', 'verb', 'adjective', 'adverb', 'determiner', 
                      'preposition', 'pronoun', 'conjunction', 'keyword', 
                      'identifier', 'operator', 'literal', 'inline_code']]
        
        phrase_level = [a for a in seq_anns if a['xbar_label'] in 
                       ['noun_phrase', 'verb_phrase', 'expression', 
                        'function_call', 'code_block', 'documentation_comment']]
        
        clause_level = [a for a in seq_anns if a['xbar_label'] in 
                       ['main_clause', 'subordinate_clause', 'if_statement', 
                        'loop_statement', 'function_definition']]
        
        hierarchy_analysis[seq_num] = {
            'word_level': len(word_level),
            'phrase_level': len(phrase_level),
            'clause_level': len(clause_level),
            'total': len(seq_anns)
        }
    
    return hierarchy_analysis

def analyze_span_quality(annotations):
    """
    Analyze the quality and patterns of span annotations without modifying them.
    
    This function analyzes:
    1. Multiple occurrences of the same text at different positions (legitimate)
    2. Potential duplicate patterns that might indicate annotation issues
    3. Consistency of labeling across occurrences
    """
    # Group by sequence
    by_sequence = defaultdict(list)
    for ann in annotations:
        by_sequence[ann['sequence_number']].append(ann)
    
    quality_stats = {
        'multiple_occurrences': [],
        'potential_issues': [],
        'consistency_analysis': []
    }
    
    for seq_num, seq_anns in by_sequence.items():
        # Sort annotations by start position for analysis
        seq_anns_sorted = sorted(seq_anns, key=lambda x: (x['start_pos'], x['end_pos']))
        
        # Group by text to find multiple occurrences
        text_occurrences = defaultdict(list)
        for ann in seq_anns:
            text_occurrences[ann['text']].append(ann)
        
        # Analyze multiple occurrences
        for text, occurrences in text_occurrences.items():
            if len(occurrences) > 1:
                # Sort by position
                occurrences_sorted = sorted(occurrences, key=lambda x: x['start_pos'])
                
                positions = [(ann['start_pos'], ann['end_pos']) for ann in occurrences_sorted]
                labels = [ann['xbar_label'] for ann in occurrences_sorted]
                unique_labels = set(labels)
                
                # Check for consecutive occurrences that might be problematic
                consecutive_pairs = []
                for i in range(len(positions) - 1):
                    curr_start, curr_end = positions[i]
                    next_start, next_end = positions[i + 1]
                    gap = next_start - curr_end
                    
                    # Flag if very close together (especially for single chars)
                    is_single_char = len(text) == 1
                    max_allowed_gap = 0 if is_single_char else 1
                    
                    if gap <= max_allowed_gap:
                        consecutive_pairs.append({
                            'positions': [(curr_start, curr_end), (next_start, next_end)],
                            'gap': gap,
                            'labels': [labels[i], labels[i + 1]]
                        })
                
                quality_stats['multiple_occurrences'].append({
                    'sequence': seq_num,
                    'text': text,
                    'count': len(occurrences),
                    'positions': positions,
                    'labels': labels,
                    'unique_labels': list(unique_labels),
                    'is_consistent': len(unique_labels) == 1,
                    'consecutive_pairs': consecutive_pairs
                })
                
                # Flag potential issues
                if consecutive_pairs:
                    quality_stats['potential_issues'].append({
                        'type': 'consecutive_duplicates',
                        'sequence': seq_num,
                        'text': text,
                        'count': len(consecutive_pairs),
                        'details': consecutive_pairs
                    })
                
                if len(unique_labels) > 1:
                    quality_stats['consistency_analysis'].append({
                        'sequence': seq_num,
                        'text': text,
                        'labels': list(unique_labels),
                        'occurrences': len(occurrences)
                    })
    
    return quality_stats

def find_anomalies(annotations):
    """Find potential anomalies in span annotations with enhanced position-aware detection."""
    anomalies = []
    
    # Group by sequence
    by_sequence = defaultdict(list)
    for ann in annotations:
        by_sequence[ann['sequence_number']].append(ann)
    
    for seq_num, seq_anns in by_sequence.items():
        # Sort by position for better analysis
        seq_anns_sorted = sorted(seq_anns, key=lambda x: (x['start_pos'], x['end_pos']))
        
        # Check for exact duplicates (boundary + text + label - true duplicates)
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
        
        # Enhanced: Check for overlapping spans with same text (potential over-annotation)
        for i in range(len(seq_anns_sorted) - 1):
            curr = seq_anns_sorted[i]
            for j in range(i + 1, len(seq_anns_sorted)):
                next_ann = seq_anns_sorted[j]
                
                # Skip if next span starts after current ends
                if next_ann['start_pos'] >= curr['end_pos']:
                    break
                
                # Check for overlapping spans with same text
                if (curr['text'] == next_ann['text'] and 
                    curr['xbar_label'] == next_ann['xbar_label'] and
                    curr['start_pos'] != next_ann['start_pos']):
                    
                    overlap_start = max(curr['start_pos'], next_ann['start_pos'])
                    overlap_end = min(curr['end_pos'], next_ann['end_pos'])
                    overlap_length = overlap_end - overlap_start
                    
                    if overlap_length > 0:
                        anomalies.append({
                            'type': 'overlapping_identical_spans',
                            'sequence': seq_num,
                            'text': curr['text'],
                            'label': curr['xbar_label'],
                            'span1': (curr['start_pos'], curr['end_pos']),
                            'span2': (next_ann['start_pos'], next_ann['end_pos']),
                            'overlap_length': overlap_length,
                            'severity': 'medium'
                        })
        
        # Check for suspiciously short spans (refined logic)
        for ann in seq_anns:
            # Single-character spans that are typically valid
            valid_short_labels = [
                'operator', 'literal', 'identifier', 'punctuation', 
                'conjunction', 'preposition', 'determiner', 'pronoun'
            ]
            
            # Common single-character words that are valid
            valid_single_chars = ['a', 'i', 's', 'j', 'c', 'x', 'y', 'z', 'n', 'm', 'k', 
                                 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', '.', ',', ';', 
                                 ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', "'"]
            
            domain_type = ann.get('domain_type', 'mixed')
            
            if (len(ann['text']) <= 1 and 
                ann['xbar_label'] not in valid_short_labels and
                ann['text'].lower() not in valid_single_chars and
                domain_type == 'natural'):
                
                anomalies.append({
                    'type': 'suspicious_short_span',
                    'sequence': seq_num,
                    'span': ann,
                    'severity': 'low'
                })
        
        # Enhanced: Check for truly problematic repetitive patterns
        # Only flag texts that appear with identical positions (actual duplicates)
        # or unusually high frequencies that suggest annotation errors
        text_counts = Counter(ann['text'] for ann in seq_anns)
        for text, count in text_counts.items():
            # Get all positions for this text
            positions = [(ann['start_pos'], ann['end_pos']) for ann in seq_anns if ann['text'] == text]
            
            # Check for position duplicates (true anomalies)
            position_counts = Counter(positions)
            position_duplicates = [pos for pos, cnt in position_counts.items() if cnt > 1]
            
            if position_duplicates:
                # This is a real anomaly - same text at same position multiple times
                anomalies.append({
                    'type': 'position_duplicate',
                    'sequence': seq_num,
                    'text': text,
                    'duplicate_positions': position_duplicates,
                    'count': count,
                    'severity': 'high'
                })
            else:
                # Only flag extremely high frequencies that suggest annotation errors
                # Be very lenient - only flag if it's clearly excessive
                is_single_char = len(text) == 1
                is_punctuation = text in [',', '.', ';', ':', '!', '?', '(', ')', '"', "'"]
                is_common_word = text.lower() in ['the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were']
                
                # Very high thresholds - only flag truly excessive cases
                if is_single_char and is_punctuation:
                    threshold = 50  # Very high for single punctuation
                elif is_single_char:
                    threshold = 25  # High for single characters
                elif is_common_word:
                    threshold = 20  # High for common words
                else:
                    threshold = 15  # High for other words
                
                if count > threshold:
                    anomalies.append({
                        'type': 'excessive_repetition',
                        'sequence': seq_num,
                        'text': text,
                        'count': count,
                        'positions': positions[:3],  # Show first 3 positions
                        'severity': 'low'
                    })
        
        # Enhanced: Check for inconsistent labeling across positions
        text_to_labels = defaultdict(set)
        text_to_positions = defaultdict(list)
        for ann in seq_anns:
            text_to_labels[ann['text']].add(ann['xbar_label'])
            text_to_positions[ann['text']].append((ann['start_pos'], ann['end_pos'], ann['xbar_label']))
        
        for text, labels in text_to_labels.items():
            # Allow reasonable variations but flag excessive inconsistency
            if len(labels) > 3:  # More than 3 different labels for same text
                positions_with_labels = text_to_positions[text]
                
                anomalies.append({
                    'type': 'inconsistent_labeling',
                    'sequence': seq_num,
                    'text': text,
                    'labels': list(labels),
                    'occurrences': len(positions_with_labels),
                    'examples': positions_with_labels[:3],  # Show first 3 examples
                    'severity': 'medium'
                })
        
        # Check for insufficient hierarchical coverage
        word_level = [a for a in seq_anns if a['xbar_label'] in 
                     ['noun', 'verb', 'adjective', 'adverb', 'determiner', 
                      'preposition', 'pronoun', 'conjunction', 'keyword', 
                      'identifier', 'operator', 'literal', 'inline_code', 'punctuation']]
        
        phrase_level = [a for a in seq_anns if a['xbar_label'] in 
                       ['noun_phrase', 'verb_phrase', 'expression', 
                        'function_call', 'code_block', 'documentation_comment', 'adverb_phrase']]
        
        clause_level = [a for a in seq_anns if a['xbar_label'] in 
                       ['main_clause', 'subordinate_clause', 'if_statement', 
                        'loop_statement', 'function_definition', 'relative_clause']]
        
        # Flag sequences with insufficient hierarchical coverage
        missing_levels = []
        if len(word_level) < 1:
            missing_levels.append('word')
        if len(phrase_level) < 1:
            missing_levels.append('phrase')
        if len(clause_level) < 1:
            missing_levels.append('clause')
        
        if missing_levels:
            anomalies.append({
                'type': 'insufficient_hierarchical_coverage',
                'sequence': seq_num,
                'missing_levels': missing_levels,
                'word_count': len(word_level),
                'phrase_count': len(phrase_level),
                'clause_count': len(clause_level),
                'total_spans': len(seq_anns),
                'severity': 'low'
            })
    
    return anomalies

def analyze_multiple_occurrences(annotations):
    """
    Analyze multiple occurrences of the same text to ensure we're properly handling
    legitimate repeated spans at different positions.
    """
    # Group by sequence
    by_sequence = defaultdict(list)
    for ann in annotations:
        by_sequence[ann['sequence_number']].append(ann)
    
    occurrence_analysis = []
    
    for seq_num, seq_anns in by_sequence.items():
        # Group by text to find multiple occurrences
        text_occurrences = defaultdict(list)
        for ann in seq_anns:
            text_occurrences[ann['text']].append(ann)
        
        # Analyze texts with multiple occurrences
        for text, occurrences in text_occurrences.items():
            if len(occurrences) > 1:
                # Sort by position
                occurrences_sorted = sorted(occurrences, key=lambda x: x['start_pos'])
                
                # Analyze the pattern
                positions = [(ann['start_pos'], ann['end_pos']) for ann in occurrences_sorted]
                labels = [ann['xbar_label'] for ann in occurrences_sorted]
                
                # Check for consistency
                unique_labels = set(labels)
                is_consistent = len(unique_labels) == 1
                
                # Check for appropriate spacing (not overlapping unless expected)
                overlaps = []
                for i in range(len(positions) - 1):
                    curr_start, curr_end = positions[i]
                    next_start, next_end = positions[i + 1]
                    
                    if next_start < curr_end:  # Overlap detected
                        overlap_length = curr_end - next_start
                        overlaps.append({
                            'positions': (positions[i], positions[i + 1]),
                            'overlap_length': overlap_length
                        })
                
                # Calculate spacing between occurrences
                gaps = []
                for i in range(len(positions) - 1):
                    gap = positions[i + 1][0] - positions[i][1]
                    gaps.append(gap)
                
                occurrence_analysis.append({
                    'sequence': seq_num,
                    'text': text,
                    'count': len(occurrences),
                    'positions': positions,
                    'labels': labels,
                    'unique_labels': list(unique_labels),
                    'is_consistent': is_consistent,
                    'overlaps': overlaps,
                    'gaps': gaps,
                    'avg_gap': sum(gaps) / len(gaps) if gaps else 0,
                    'min_gap': min(gaps) if gaps else 0,
                    'max_gap': max(gaps) if gaps else 0
                })
    
    return occurrence_analysis

def analyze_boundary_alignment(annotations):
    """Analyze if boundary predictions align with word boundaries."""
    boundary_issues = []
    
    for ann in annotations:
        text = ann['text']
        raw_text = ann['raw']
        start, end = ann['start_pos'], ann['end_pos']
        
        # Extract the actual text from raw
        if start < len(raw_text) and end <= len(raw_text):
            actual_text = raw_text[start:end]
            if actual_text != text:
                boundary_issues.append({
                    'sequence': ann['sequence_number'],
                    'expected': text,
                    'actual': actual_text,
                    'positions': (start, end)
                })
    
    return boundary_issues

def main():
    annotations_file = Path("data/annotations/annotations.jsonl")
    
    if not annotations_file.exists():
        print(f"Error: {annotations_file} not found")
        sys.exit(1)
    
    print("🔍 SPAN ANNOTATION ANALYSIS")
    print("=" * 50)
    
    # Load annotations
    annotations = load_annotations(annotations_file)
    print(f"📊 Loaded {len(annotations)} annotations for analysis")
    
    # Span quality analysis
    print("\n🎯 SPAN QUALITY ANALYSIS")
    print("-" * 40)
    quality_stats = analyze_span_quality(annotations)
    
    if quality_stats['multiple_occurrences']:
        print(f"📊 Multiple occurrence summary:")
        print(f"  - Texts with multiple occurrences: {len(quality_stats['multiple_occurrences'])}")
        
        # Show examples of high-frequency occurrences
        high_frequency = [mo for mo in quality_stats['multiple_occurrences'] if mo['count'] >= 5]
        if high_frequency:
            print(f"\n� High frequency texts (5+ occurrences):")
            for mo in sorted(high_frequency, key=lambda x: x['count'], reverse=True)[:5]:
                consistent_info = "✅ consistent" if mo['is_consistent'] else f"❌ inconsistent ({len(mo['unique_labels'])} labels)"
                print(f"  '{mo['text']}': {mo['count']} occurrences, {consistent_info}")
                if mo['consecutive_pairs']:
                    print(f"    ⚠️  {len(mo['consecutive_pairs'])} consecutive occurrence pairs")
        
        # Check for punctuation and common words
        punctuation_texts = [mo for mo in quality_stats['multiple_occurrences'] if mo['text'] in [',', '.', ';', ':', '!', '?', '(', ')', '"', "'"]]
        if punctuation_texts:
            print(f"\n📝 Punctuation analysis:")
            for mo in sorted(punctuation_texts, key=lambda x: x['count'], reverse=True):
                print(f"  '{mo['text']}': {mo['count']} occurrences")
                if mo['consecutive_pairs']:
                    print(f"    ⚠️  {len(mo['consecutive_pairs'])} consecutive pairs detected")
    
    # Potential issues
    if quality_stats['potential_issues']:
        print(f"\n⚠️  POTENTIAL ISSUES DETECTED:")
        consecutive_issues = [pi for pi in quality_stats['potential_issues'] if pi['type'] == 'consecutive_duplicates']
        if consecutive_issues:
            print(f"  - Consecutive duplicates: {len(consecutive_issues)} cases")
            for issue in consecutive_issues[:3]:  # Show first 3
                print(f"    '{issue['text']}' has {issue['count']} consecutive pairs in sequence {issue['sequence']}")
    
    # Consistency analysis
    if quality_stats['consistency_analysis']:
        print(f"\n🔍 CONSISTENCY ANALYSIS:")
        print(f"  - Texts with inconsistent labeling: {len(quality_stats['consistency_analysis'])}")
        for ca in quality_stats['consistency_analysis'][:3]:  # Show first 3
            print(f"    '{ca['text']}': {ca['labels']} ({ca['occurrences']} occurrences)")
    
    if not quality_stats['multiple_occurrences']:
        print("✅ No multiple occurrences found - all texts appear only once!")
    
    if not quality_stats['potential_issues']:
        print("✅ No potential issues detected!")
    
    if not quality_stats['consistency_analysis']:
        print("✅ All repeated texts have consistent labeling!")
    
    # Multiple occurrence analysis
    print("\n🔄 MULTIPLE OCCURRENCE ANALYSIS")
    print("-" * 40)
    occurrence_analysis = analyze_multiple_occurrences(annotations)
    
    if occurrence_analysis:
        # Group by text frequency
        high_frequency = [oa for oa in occurrence_analysis if oa['count'] >= 5]
        medium_frequency = [oa for oa in occurrence_analysis if 3 <= oa['count'] < 5]
        low_frequency = [oa for oa in occurrence_analysis if 2 <= oa['count'] < 3]
        
        print(f"📊 Multiple occurrence summary:")
        print(f"  - High frequency (5+ occurrences): {len(high_frequency)} texts")
        print(f"  - Medium frequency (3-4 occurrences): {len(medium_frequency)} texts")
        print(f"  - Low frequency (2 occurrences): {len(low_frequency)} texts")
        print(f"  - Total texts with multiple occurrences: {len(occurrence_analysis)}")
        
        # Show examples of high-frequency occurrences
        if high_frequency:
            print(f"\n🔥 High frequency texts:")
            for oa in sorted(high_frequency, key=lambda x: x['count'], reverse=True)[:5]:
                consistent_info = "✅ consistent" if oa['is_consistent'] else f"❌ inconsistent ({len(oa['unique_labels'])} labels)"
                print(f"  '{oa['text']}': {oa['count']} occurrences, {consistent_info}")
                if oa['overlaps']:
                    print(f"    ⚠️  {len(oa['overlaps'])} overlapping occurrences")
                if oa['gaps']:
                    print(f"    📏 Avg gap: {oa['avg_gap']:.1f} chars (min: {oa['min_gap']}, max: {oa['max_gap']})")
        
        # Check for punctuation and common words
        punctuation_texts = [oa for oa in occurrence_analysis if oa['text'] in [',', '.', ';', ':', '!', '?', '(', ')', '"', "'"]]
        if punctuation_texts:
            print(f"\n📝 Punctuation analysis:")
            for oa in sorted(punctuation_texts, key=lambda x: x['count'], reverse=True):
                print(f"  '{oa['text']}': {oa['count']} occurrences across {len(set(ann[0] for ann in [(oa['sequence'], pos) for pos in oa['positions']]))} sequences")
        
        # Flag potential issues
        inconsistent_texts = [oa for oa in occurrence_analysis if not oa['is_consistent']]
        if inconsistent_texts:
            print(f"\n⚠️  Inconsistent labeling across positions: {len(inconsistent_texts)} texts")
            for oa in inconsistent_texts[:3]:
                print(f"  '{oa['text']}': labels {oa['unique_labels']}")
        
        overlapping_texts = [oa for oa in occurrence_analysis if oa['overlaps']]
        if overlapping_texts:
            print(f"\n🔄 Texts with overlapping occurrences: {len(overlapping_texts)}")
            for oa in overlapping_texts[:3]:
                print(f"  '{oa['text']}': {len(oa['overlaps'])} overlaps")
    else:
        print("✅ No multiple occurrences found - all texts appear only once!")
    
    # Sequence summary
    sequences = set(ann['sequence_number'] for ann in annotations)
    print(f"\n📝 Sequences analyzed: {len(sequences)} sequences ({min(sequences)} to {max(sequences)})")
    print()
    
    # Hierarchical structure analysis
    print("🏗️  HIERARCHICAL STRUCTURE ANALYSIS")
    print("-" * 40)
    hierarchy = analyze_hierarchical_structure(annotations)
    
    # Calculate global statistics
    total_word_spans = sum(h['word_level'] for h in hierarchy.values())
    total_phrase_spans = sum(h['phrase_level'] for h in hierarchy.values())
    total_clause_spans = sum(h['clause_level'] for h in hierarchy.values())
    total_sequences = len(hierarchy)
    
    print(f"📊 GLOBAL SPAN STATISTICS:")
    print(f"  Total Word-level spans: {total_word_spans}")
    print(f"  Total Phrase-level spans: {total_phrase_spans}")
    print(f"  Total Clause-level spans: {total_clause_spans}")
    print(f"  Total spans across all levels: {total_word_spans + total_phrase_spans + total_clause_spans}")
    print()
    
    print(f"📈 AVERAGES PER SEQUENCE:")
    print(f"  Avg word spans/sequence: {total_word_spans / total_sequences:.1f}")
    print(f"  Avg phrase spans/sequence: {total_phrase_spans / total_sequences:.1f}")
    print(f"  Avg clause spans/sequence: {total_clause_spans / total_sequences:.1f}")
    print(f"  Avg total spans/sequence: {(total_word_spans + total_phrase_spans + total_clause_spans) / total_sequences:.1f}")
    print()
    
    print(f"📋 SPAN DISTRIBUTION:")
    total_spans = total_word_spans + total_phrase_spans + total_clause_spans
    print(f"  Word-level: {total_word_spans / total_spans * 100:.1f}% of all spans")
    print(f"  Phrase-level: {total_phrase_spans / total_spans * 100:.1f}% of all spans")
    print(f"  Clause-level: {total_clause_spans / total_spans * 100:.1f}% of all spans")
    print()
    
    print(f"📏 SEQUENCE-BY-SEQUENCE BREAKDOWN:")
    for seq_num in sorted(hierarchy.keys()):
        h = hierarchy[seq_num]
        print(f"Sequence {seq_num}: {h['total']} total spans")
        print(f"  - Word level: {h['word_level']} ({h['word_level']/h['total']*100:.1f}%)")
        print(f"  - Phrase level: {h['phrase_level']} ({h['phrase_level']/h['total']*100:.1f}%)")
        print(f"  - Clause level: {h['clause_level']} ({h['clause_level']/h['total']*100:.1f}%)")
        
        # Check balance
        ratio = h['word_level'] / max(h['phrase_level'], 1)
        if ratio > 10:
            print(f"  ⚠️  High word-to-phrase ratio: {ratio:.1f}")
        elif ratio < 2:
            print(f"  ⚠️  Low word-to-phrase ratio: {ratio:.1f}")
    print()
    
    # Overlapping spans analysis
    print("🔄 OVERLAPPING SPANS ANALYSIS")
    print("-" * 40)
    overlaps, coverage = analyze_overlapping_spans(annotations)
    for seq_num in sorted(overlaps.keys()):
        print(f"Sequence {seq_num}: {len(overlaps[seq_num])} overlapping pairs")
        for ann1, ann2 in overlaps[seq_num][:3]:  # Show first 3
            print(f"  - '{ann1['text']}' ({ann1['xbar_label']}) overlaps '{ann2['text']}' ({ann2['xbar_label']})")
        if len(overlaps[seq_num]) > 3:
            print(f"  ... and {len(overlaps[seq_num]) - 3} more")
    print()
    
    # Anomaly detection
    print("🚨 ANOMALY DETECTION")
    print("-" * 40)
    anomalies = find_anomalies(annotations)
    
    if not anomalies:
        print("✅ No anomalies detected!")
    else:
        # Group anomalies by type and severity
        anomaly_types = Counter(a['type'] for a in anomalies)
        severity_counts = Counter(a.get('severity', 'unknown') for a in anomalies)
        
        print(f"📊 Anomaly summary:")
        print(f"  - Total anomalies: {len(anomalies)}")
        print(f"  - By severity: {dict(severity_counts)}")
        print(f"  - By type: {dict(anomaly_types)}")
        
        # Group by severity for reporting
        high_severity = [a for a in anomalies if a.get('severity') == 'high']
        medium_severity = [a for a in anomalies if a.get('severity') == 'medium']
        low_severity = [a for a in anomalies if a.get('severity') == 'low']
        
        if high_severity:
            print(f"\n🔴 HIGH SEVERITY ANOMALIES ({len(high_severity)}):")
            for anomaly in high_severity[:5]:
                if anomaly['type'] == 'exact_duplicate':
                    print(f"  🔄 Exact duplicate in seq {anomaly['sequence']}: '{anomaly['span']['text']}' ({anomaly['span']['xbar_label']})")
                elif anomaly['type'] == 'overlapping_identical_spans':
                    print(f"  📐 Overlapping identical spans in seq {anomaly['sequence']}: '{anomaly['text']}' ({anomaly['label']}) - overlap: {anomaly['overlap_length']} chars")
                elif anomaly['type'] == 'position_duplicate':
                    print(f"  🔄 Position duplicate in seq {anomaly['sequence']}: '{anomaly['text']}' appears {anomaly['count']} times at same positions {anomaly['duplicate_positions']}")
            if len(high_severity) > 5:
                print(f"  ... and {len(high_severity) - 5} more high severity anomalies")
        
        if medium_severity:
            print(f"\n🟡 MEDIUM SEVERITY ANOMALIES ({len(medium_severity)}):")
            for anomaly in medium_severity[:5]:
                if anomaly['type'] == 'boundary_duplicate':
                    print(f"  🏷️  Boundary duplicate in seq {anomaly['sequence']}: '{anomaly['text']}' @ {anomaly['positions']} has labels {anomaly['labels']}")
                elif anomaly['type'] == 'inconsistent_labeling':
                    print(f"  🏷️  Inconsistent labels in seq {anomaly['sequence']}: '{anomaly['text']}' -> {anomaly['labels']} ({anomaly['occurrences']} occurrences)")
            if len(medium_severity) > 5:
                print(f"  ... and {len(medium_severity) - 5} more medium severity anomalies")
        
        if low_severity:
            print(f"\n🟢 LOW SEVERITY ANOMALIES ({len(low_severity)}):")
            for anomaly in low_severity[:3]:  # Show fewer low severity
                if anomaly['type'] == 'suspicious_short_span':
                    print(f"  ⚠️  Short span in seq {anomaly['sequence']}: '{anomaly['span']['text']}' ({anomaly['span']['xbar_label']})")
                elif anomaly['type'] == 'excessive_repetition':
                    print(f"  � Excessive repetition in seq {anomaly['sequence']}: '{anomaly['text']}' appears {anomaly['count']} times (threshold exceeded)")
                elif anomaly['type'] == 'insufficient_hierarchical_coverage':
                    missing_levels_str = ', '.join(anomaly['missing_levels'])
                    print(f"  🏗️  Insufficient hierarchical coverage in seq {anomaly['sequence']}: missing {missing_levels_str} level spans")
                    print(f"      Word: {anomaly['word_count']}, Phrase: {anomaly['phrase_count']}, Clause: {anomaly['clause_count']} (Total: {anomaly['total_spans']})")
            if len(low_severity) > 3:
                print(f"  ... and {len(low_severity) - 3} more low severity anomalies")
    print()
    
    # Boundary alignment check
    print("🎯 BOUNDARY ALIGNMENT CHECK")
    print("-" * 40)
    boundary_issues = analyze_boundary_alignment(annotations)
    if not boundary_issues:
        print("✅ All boundaries align correctly!")
    else:
        print(f"⚠️  Found {len(boundary_issues)} boundary misalignments:")
        for issue in boundary_issues[:5]:
            print(f"  Seq {issue['sequence']}: expected '{issue['expected']}' but got '{issue['actual']}'")
    print()
    
    # Label distribution
    print("📈 LABEL DISTRIBUTION")
    print("-" * 40)
    label_counts = Counter(ann['xbar_label'] for ann in annotations)
    
    # Categorize labels by hierarchical level
    word_labels = ['noun', 'verb', 'adjective', 'adverb', 'determiner', 
                   'preposition', 'pronoun', 'conjunction', 'keyword', 
                   'identifier', 'operator', 'literal', 'inline_code', 'punctuation']
    phrase_labels = ['noun_phrase', 'verb_phrase', 'expression', 
                     'function_call', 'code_block', 'documentation_comment', 'adverb_phrase']
    clause_labels = ['main_clause', 'subordinate_clause', 'if_statement', 
                     'loop_statement', 'function_definition', 'relative_clause']
    
    word_count = sum(label_counts[label] for label in word_labels if label in label_counts)
    phrase_count = sum(label_counts[label] for label in phrase_labels if label in label_counts)
    clause_count = sum(label_counts[label] for label in clause_labels if label in label_counts)
    other_count = sum(count for label, count in label_counts.items() 
                     if label not in word_labels + phrase_labels + clause_labels)
    
    print(f"📊 BY HIERARCHICAL LEVEL:")
    print(f"  Word-level labels: {word_count} spans")
    print(f"  Phrase-level labels: {phrase_count} spans")  
    print(f"  Clause-level labels: {clause_count} spans")
    if other_count > 0:
        print(f"  Other/Unclassified: {other_count} spans")
    print()
    
    print(f"📋 ALL LABELS (sorted by frequency):")
    for label, count in label_counts.most_common():
        percentage = count / len(annotations) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    print()
    
    # Additional span statistics
    print("📐 SPAN LENGTH STATISTICS")
    print("-" * 40)
    span_lengths = [len(ann['text']) for ann in annotations]
    avg_length = sum(span_lengths) / len(span_lengths)
    min_length = min(span_lengths)
    max_length = max(span_lengths)
    median_length = sorted(span_lengths)[len(span_lengths) // 2]
    
    print(f"  Average span length: {avg_length:.1f} characters")
    print(f"  Minimum span length: {min_length} characters")
    print(f"  Maximum span length: {max_length} characters")
    print(f"  Median span length: {median_length} characters")
    
    # Length distribution
    length_ranges = [(1, 5), (6, 15), (16, 30), (31, 50), (51, float('inf'))]
    print(f"\n  Length distribution:")
    for min_len, max_len in length_ranges:
        if max_len == float('inf'):
            count = sum(1 for length in span_lengths if length >= min_len)
            range_str = f"{min_len}+ chars"
        else:
            count = sum(1 for length in span_lengths if min_len <= length <= max_len)
            range_str = f"{min_len}-{max_len} chars"
        
        percentage = count / len(span_lengths) * 100
        print(f"    {range_str}: {count} spans ({percentage:.1f}%)")
    print()
    
    print("✨ Analysis complete!")

if __name__ == "__main__":
    main()
