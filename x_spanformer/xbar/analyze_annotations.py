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

def deduplicate_spans(annotations):
    """
    Deduplicate spans with identical boundaries but different X-bar labels.
    Strategy: Pick the label with the most occurrences globally. If tied, pick the first one.
    """
    # Count label frequency globally
    label_counts = Counter(ann['xbar_label'] for ann in annotations)
    
    # Group by sequence
    by_sequence = defaultdict(list)
    for ann in annotations:
        by_sequence[ann['sequence_number']].append(ann)
    
    deduplicated = []
    dedup_stats = {'removed': 0, 'kept': 0, 'decisions': []}
    
    for seq_num, seq_anns in by_sequence.items():
        # Group by boundary + text
        boundary_groups = defaultdict(list)
        for ann in seq_anns:
            boundary_key = (ann['start_pos'], ann['end_pos'], ann['text'])
            boundary_groups[boundary_key].append(ann)
        
        # Process each boundary group
        for boundary_key, duplicate_anns in boundary_groups.items():
            if len(duplicate_anns) == 1:
                # No duplicates, keep as is
                deduplicated.extend(duplicate_anns)
                dedup_stats['kept'] += 1
            else:
                # Multiple labels for same boundary - resolve conflict
                start_pos, end_pos, text = boundary_key
                
                # Sort by: 1) global label frequency (desc), 2) original order (asc)
                def sort_key(ann):
                    return (-label_counts[ann['xbar_label']], ann['id'])
                
                sorted_anns = sorted(duplicate_anns, key=sort_key)
                winner = sorted_anns[0]
                losers = sorted_anns[1:]
                
                deduplicated.append(winner)
                dedup_stats['kept'] += 1
                dedup_stats['removed'] += len(losers)
                
                # Log the decision
                decision = {
                    'sequence': seq_num,
                    'text': text,
                    'positions': (start_pos, end_pos),
                    'winner': winner['xbar_label'],
                    'winner_count': label_counts[winner['xbar_label']],
                    'losers': [{'label': ann['xbar_label'], 'count': label_counts[ann['xbar_label']]} 
                              for ann in losers]
                }
                dedup_stats['decisions'].append(decision)
    
    return deduplicated, dedup_stats

def find_anomalies(annotations):
    """Find potential anomalies in span annotations."""
    anomalies = []
    
    # Group by sequence
    by_sequence = defaultdict(list)
    for ann in annotations:
        by_sequence[ann['sequence_number']].append(ann)
    
    for seq_num, seq_anns in by_sequence.items():
        # Check for exact duplicates (boundary + text + label)
        seen_exact = set()
        for ann in seq_anns:
            exact_key = (ann['start_pos'], ann['end_pos'], ann['text'], ann['xbar_label'])
            if exact_key in seen_exact:
                anomalies.append({
                    'type': 'exact_duplicate',
                    'sequence': seq_num,
                    'span': ann
                })
            seen_exact.add(exact_key)
        
        # Check for boundary duplicates (boundary + text, different labels)
        boundary_groups = defaultdict(list)
        for ann in seq_anns:
            boundary_key = (ann['start_pos'], ann['end_pos'], ann['text'])
            boundary_groups[boundary_key].append(ann)
        
        for boundary_key, duplicate_anns in boundary_groups.items():
            if len(duplicate_anns) > 1:
                # Get unique labels
                labels = set(ann['xbar_label'] for ann in duplicate_anns)
                if len(labels) > 1:
                    anomalies.append({
                        'type': 'boundary_duplicate',
                        'sequence': seq_num,
                        'text': boundary_key[2],
                        'positions': (boundary_key[0], boundary_key[1]),
                        'labels': list(labels),
                        'count': len(duplicate_anns)
                    })
        
        # Check for suspiciously short spans
        for ann in seq_anns:
            # Single-character spans are valid for:
            # - Code/mixed domains: identifiers, variables, operators, literals, punctuation
            # - Natural language: punctuation, articles, prepositions
            valid_short_labels = [
                'operator', 'literal', 'identifier', 'punctuation', 
                'conjunction', 'preposition', 'determiner', 'pronoun'
            ]
            
            # For code/mixed domains, single char spans are generally acceptable
            domain_type = ann.get('domain_type', 'mixed')
            
            if (len(ann['text']) <= 1 and 
                ann['xbar_label'] not in valid_short_labels and
                domain_type == 'natural'):  # Only flag in pure natural language
                
                # Additional check: skip common single letters that are valid
                if ann['text'].lower() not in ['a', 'i', 's', 'j', 'c', 'x', 'y', 'z', 'n', 'm', 'k']:
                    anomalies.append({
                        'type': 'suspicious_short_span',
                        'sequence': seq_num,
                        'span': ann
                    })
        
        # Check for repetitive patterns
        text_counts = Counter(ann['text'] for ann in seq_anns)
        for text, count in text_counts.items():
            if count > 3:  # More than 3 occurrences might be suspicious
                anomalies.append({
                    'type': 'repetitive_text',
                    'sequence': seq_num,
                    'text': text,
                    'count': count
                })
        
        # Check for inconsistent labeling of same text
        text_to_labels = defaultdict(set)
        for ann in seq_anns:
            text_to_labels[ann['text']].add(ann['xbar_label'])
        
        for text, labels in text_to_labels.items():
            # Allow reasonable variations (e.g., noun + noun_phrase is fine)
            if len(labels) > 2:
                anomalies.append({
                    'type': 'inconsistent_labeling',
                    'sequence': seq_num,
                    'text': text,
                    'labels': list(labels)
                })
    
    return anomalies

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
    original_annotations = load_annotations(annotations_file)
    print(f"📊 Loaded {len(original_annotations)} original annotations")
    
    # Deduplicate spans
    print("\n🚿 DEDUPLICATING SPANS")
    print("-" * 40)
    annotations, dedup_stats = deduplicate_spans(original_annotations)
    
    if dedup_stats['removed'] > 0:
        print(f"✨ Deduplication completed:")
        print(f"  - Original annotations: {len(original_annotations)}")
        print(f"  - After deduplication: {len(annotations)}")
        print(f"  - Removed duplicates: {dedup_stats['removed']}")
        print(f"  - Deduplication decisions: {len(dedup_stats['decisions'])}")
        
        print(f"\n📋 Deduplication decisions:")
        for decision in dedup_stats['decisions']:
            winner_info = f"{decision['winner']} (appears {decision['winner_count']} times)"
            loser_info = ", ".join([f"{l['label']} ({l['count']})" for l in decision['losers']])
            print(f"  '{decision['text']}' @ {decision['positions']}: kept {winner_info}, removed {loser_info}")
        
        # Save deduplicated annotations
        dedup_file = annotations_file.parent / "annotations_deduplicated.jsonl"
        with open(dedup_file, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + '\n')
        print(f"\n💾 Saved deduplicated annotations to: {dedup_file}")
    else:
        print("✅ No duplicates found - all annotations are unique!")
    
    # Sequence summary
    sequences = set(ann['sequence_number'] for ann in annotations)
    print(f"\n📝 Sequences analyzed: {sorted(sequences)}")
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
        anomaly_types = Counter(a['type'] for a in anomalies)
        for anomaly_type, count in anomaly_types.items():
            print(f"{anomaly_type}: {count} instances")
        
        print("\nDetailed anomalies:")
        for anomaly in anomalies[:10]:  # Show first 10
            if anomaly['type'] == 'exact_duplicate':
                print(f"  🔄 Exact duplicate in seq {anomaly['sequence']}: '{anomaly['span']['text']}' ({anomaly['span']['xbar_label']})")
            elif anomaly['type'] == 'boundary_duplicate':
                print(f"  🏷️  Boundary duplicate in seq {anomaly['sequence']}: '{anomaly['text']}' @ {anomaly['positions']} has labels {anomaly['labels']}")
            elif anomaly['type'] == 'suspicious_short_span':
                print(f"  ⚠️  Short span in seq {anomaly['sequence']}: '{anomaly['span']['text']}' ({anomaly['span']['xbar_label']})")
            elif anomaly['type'] == 'repetitive_text':
                print(f"  🔁 Repetitive in seq {anomaly['sequence']}: '{anomaly['text']}' appears {anomaly['count']} times")
            elif anomaly['type'] == 'inconsistent_labeling':
                print(f"  🏷️  Inconsistent labels in seq {anomaly['sequence']}: '{anomaly['text']}' -> {anomaly['labels']}")
        
        if len(anomalies) > 10:
            print(f"  ... and {len(anomalies) - 10} more anomalies")
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
