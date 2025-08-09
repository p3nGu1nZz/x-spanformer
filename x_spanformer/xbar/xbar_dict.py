#!/usr/bin/env python3
"""
X-Bar Dictionary Management System

Provides unified dictionary management for domain-specific vocabularies
without position information, focusing on unique spans by hierarchical level.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class XBarDictionary:
    """
    Manages domain-specific dictionaries for unique spans.
    
    Organizes spans by domain (natural, code, mixed) and hierarchical level
    (word_level, phrase_level, clause_level) without position information.
    """
    
    def __init__(self):
        """Initialize empty dictionaries."""
        self._reset_dictionaries()
        
    def _reset_dictionaries(self):
        """Reset all dictionaries to empty state."""
        self.dictionaries = {
            "natural": {
                "word_level": set(),
                "phrase_level": set(),
                "clause_level": set()
            },
            "code": {
                "word_level": set(),
                "phrase_level": set(),
                "clause_level": set()
            },
            "mixed": {
                "word_level": set(),
                "phrase_level": set(),
                "clause_level": set()
            }
        }
        
        self.stats = {
            "sequences_processed": 0,
            "total_unique_spans": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def add_spans(self, domain_type: str, hierarchical_level: str, spans: List[str]) -> int:
        """
        Add spans to the specified domain and level.
        
        Args:
            domain_type: Domain type ('natural', 'code', 'mixed')
            hierarchical_level: Hierarchical level ('word_level', 'phrase_level', 'clause_level')
            spans: List of span texts to add
            
        Returns:
            Number of new unique spans added
        """
        if domain_type not in self.dictionaries:
            logger.warning(f"Unknown domain type: {domain_type}")
            return 0
            
        if hierarchical_level not in self.dictionaries[domain_type]:
            logger.warning(f"Unknown hierarchical level: {hierarchical_level}")
            return 0
        
        level_dict = self.dictionaries[domain_type][hierarchical_level]
        initial_count = len(level_dict)
        
        # Add spans (set automatically handles duplicates)
        for span in spans:
            if span and span.strip():  # Only add non-empty spans
                level_dict.add(span.strip())
        
        new_count = len(level_dict) - initial_count
        if new_count > 0:
            logger.debug(f"Added {new_count} new spans to {domain_type}.{hierarchical_level} (total: {len(level_dict)})")
            
        return new_count
    
    def add_sequence_spans(self, domain_type: str, word_spans: List[str], 
                          phrase_spans: List[str], clause_spans: List[str]) -> Dict[str, int]:
        """
        Add spans from a complete sequence across all hierarchical levels.
        
        Args:
            domain_type: Domain type ('natural', 'code', 'mixed')
            word_spans: Word-level spans
            phrase_spans: Phrase-level spans  
            clause_spans: Clause-level spans
            
        Returns:
            Dictionary with counts of new spans added per level
        """
        counts = {
            "word_level": self.add_spans(domain_type, "word_level", word_spans),
            "phrase_level": self.add_spans(domain_type, "phrase_level", phrase_spans),
            "clause_level": self.add_spans(domain_type, "clause_level", clause_spans)
        }
        
        total_new = sum(counts.values())
        if total_new > 0:
            self.stats["sequences_processed"] += 1
            logger.debug(f"Sequence added {total_new} new spans across all levels for domain {domain_type}")
            
        return counts
    
    def get_domain_stats(self, domain_type: str) -> Dict[str, int]:
        """Get statistics for a specific domain."""
        if domain_type not in self.dictionaries:
            return {}
            
        domain_dict = self.dictionaries[domain_type]
        stats = {}
        total = 0
        
        for level, spans in domain_dict.items():
            count = len(spans)
            stats[level] = count
            total += count
            
        stats["total"] = total
        return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all dictionaries."""
        domain_totals = {}
        level_totals = defaultdict(int)
        total_unique_spans = 0
        domains = {}
        
        for domain, levels in self.dictionaries.items():
            domain_stats = self.get_domain_stats(domain)
            domains[domain] = domain_stats
            domain_totals[domain] = domain_stats["total"]
            total_unique_spans += domain_stats["total"]
            
            # Aggregate by level across domains
            for level, count in domain_stats.items():
                if level != "total":
                    level_totals[level] += count
        
        # Update stats
        self.stats["total_unique_spans"] = total_unique_spans
        self.stats["last_updated"] = datetime.now().isoformat()
        
        return {
            "sequences_processed": self.stats["sequences_processed"],
            "total_unique_spans": total_unique_spans,
            "last_updated": self.stats["last_updated"],
            "domain_totals": domain_totals,
            "level_totals": dict(level_totals),
            "domains": domains
        }
    
    def get_spans_by_domain(self, domain_type: str) -> Dict[str, List[str]]:
        """
        Get all spans for a specific domain organized by hierarchical level.
        
        Args:
            domain_type: Domain type ('natural', 'code', 'mixed')
            
        Returns:
            Dictionary with hierarchical levels as keys and sorted span lists as values
        """
        if domain_type not in self.dictionaries:
            return {}
        
        result = {}
        for level, spans in self.dictionaries[domain_type].items():
            result[level] = sorted(list(spans))
        
        return result
    
    def get_spans_by_level(self, hierarchical_level: str) -> Dict[str, List[str]]:
        """
        Get all spans for a specific hierarchical level across all domains.
        
        Args:
            hierarchical_level: Level ('word_level', 'phrase_level', 'clause_level')
            
        Returns:
            Dictionary with domains as keys and sorted span lists as values
        """
        result = {}
        for domain, levels in self.dictionaries.items():
            if hierarchical_level in levels:
                result[domain] = sorted(list(levels[hierarchical_level]))
            else:
                result[domain] = []
        
        return result
    
    def get_spans_filtered(self, domain_type: str, hierarchical_level: str) -> List[str]:
        """
        Get spans filtered by both domain and hierarchical level.
        
        Args:
            domain_type: Domain type ('natural', 'code', 'mixed')
            hierarchical_level: Level ('word_level', 'phrase_level', 'clause_level')
            
        Returns:
            Sorted list of spans
        """
        if (domain_type not in self.dictionaries or 
            hierarchical_level not in self.dictionaries[domain_type]):
            return []
        
        return sorted(list(self.dictionaries[domain_type][hierarchical_level]))
    
    def get_dictionary_summary(self) -> Dict[str, Any]:
        """Get a summary of all dictionary contents."""
        stats = self.get_all_stats()
        summary = {
            "total_unique_spans": stats["total_unique_spans"],
            "sequences_processed": stats["sequences_processed"],
            "domains": {}
        }
        
        for domain in ["natural", "code", "mixed"]:
            domain_spans = self.get_spans_by_domain(domain)
            summary["domains"][domain] = {
                level: len(spans) for level, spans in domain_spans.items()
            }
        
        return summary
    
    def log_statistics(self):
        """Log comprehensive statistics about the dictionaries."""
        stats = self.get_all_stats()
        
        logger.info("=" * 40)
        logger.info("X-BAR DICTIONARY STATISTICS")
        logger.info("=" * 40)
        logger.info(f"Sequences processed: {stats['sequences_processed']}")
        logger.info(f"Total unique spans: {stats['total_unique_spans']}")
        
        logger.info("Domain distribution:")
        for domain, count in stats["domain_totals"].items():
            logger.info(f"  {domain}: {count} unique spans")
        
        logger.info("Level distribution:")
        for level, count in stats["level_totals"].items():
            logger.info(f"  {level}: {count} unique spans")
        
        logger.info("Detailed breakdown:")
        for domain, domain_stats in stats["domains"].items():
            logger.info(f"  {domain}:")
            for level, count in domain_stats.items():
                if level != "total":
                    logger.info(f"    {level}: {count}")
            logger.info(f"    total: {domain_stats['total']}")
        logger.info("=" * 40)
    
    def generate_annotations_from_dictionary(self, corpus_file: Path, output_dir: Path) -> int:
        """
        Generate annotations.jsonl by systematically matching dictionary spans against corpus sequences.
        
        This approach provides better coverage and overlapping spans compared to direct LLM output.
        
        Args:
            corpus_file: Path to corpus.jsonl file
            output_dir: Output directory for annotations.jsonl
            
        Returns:
            Number of annotations generated
        """
        import json
        import re
        from pathlib import Path
        from datetime import datetime
        
        logger.info("Generating annotations from dictionary spans...")
        
        # Load corpus sequences
        sequences = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                sequences.append(json.loads(line))
        
        logger.info(f"Loaded {len(sequences)} sequences from corpus")
        
        # Collect all annotations
        all_annotations = []
        annotation_id = 0
        total_sequences = len(sequences)
        
        # Prepare output file for incremental writing
        annotations_file = output_dir / "annotations.jsonl"
        
        # Clear the file first
        with open(annotations_file, 'w', encoding='utf-8') as f:
            pass  # Just clear the file
        
        for seq_idx, seq in enumerate(sequences, 1):
            sequence_number = seq['meta']['sequence_number']
            raw_text = seq['raw']
            domain_type = seq.get('type', 'mixed')
            text_length = len(raw_text)
            
            sequence_annotations = []
            
            # Process each domain and hierarchical level
            for check_domain in ['natural', 'code', 'mixed']:
                # Focus on the sequence's domain and mixed (which covers cross-domain spans)
                if check_domain != domain_type and check_domain != 'mixed':
                    continue
                    
                for level in ['word_level', 'phrase_level', 'clause_level']:
                    spans_to_find = self.get_spans_filtered(check_domain, level)
                    level_matches = 0
                    
                    for span_text in spans_to_find:
                        # Apply basic filtering similar to xbar_json
                        span_text = str(span_text).strip()
                        if not span_text:
                            continue
                            
                        # Skip obvious artifacts (repeated characters)
                        if len(set(span_text)) == 1 and len(span_text) > 3:
                            continue
                            
                        # Skip repetitive punctuation patterns
                        if len(span_text) > 1 and all(c in '.,;:!?-_()[]{}' for c in span_text):
                            continue
                            
                        # Skip very short non-meaningful text
                        if len(span_text) == 1 and span_text.isspace():
                            continue
                        
                        # Skip placeholder/garbage values
                        if span_text in ['text', 'label', 'xbar_label', 'unknown']:
                            continue
                            
                        # Additional filtering for word-level spans to reduce noise
                        if level == 'word_level':
                            # Skip multi-word spans at word level
                            if len(span_text.split()) > 1:
                                continue
                            # Skip very common single characters and short words
                            if len(span_text) <= 2 and span_text.lower() in ['a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'as', 'is', 'it', 'or', 'and', 'but']:
                                continue
                            # Skip pure numeric strings longer than 1 character (years, page numbers, etc.)
                            if span_text.isdigit() and len(span_text) > 1:
                                continue
                            # Skip single punctuation marks
                            if len(span_text) == 1 and span_text in '.,;:!?()[]{}"\'-_/\\':
                                continue
                        
                        span_len = len(span_text)
                        if span_len == 0:
                            continue
                            
                        # Use regex for boundary-aware matching
                        pattern = re.escape(span_text)
                        
                        # Find all matches with proper word boundaries
                        for match in re.finditer(pattern, raw_text, re.IGNORECASE):
                            start_pos = match.start()
                            end_pos = match.end()
                            
                            # Check if this is a valid linguistic boundary
                            if not self._is_valid_span_boundary(raw_text, start_pos, end_pos, span_text):
                                continue
                            
                            # Create annotation record
                            annotation = {
                                "id": annotation_id,
                                "sequence_number": sequence_number,
                                "raw": raw_text,
                                "domain_type": domain_type,
                                "start_pos": start_pos,
                                "end_pos": end_pos,
                                "xbar_label": self._get_xbar_label_for_level(level),
                                "text": span_text,
                                "source": "dictionary_match",
                                "matched_domain": check_domain,
                                "hierarchical_level": level,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            sequence_annotations.append(annotation)
                            annotation_id += 1
                            level_matches += 1
                    
                    # Log each level processing
                    logger.debug(f"Seq {seq_idx}/{total_sequences} | {check_domain}:{level} | found {level_matches} matches")
            
            # Remove exact duplicates per sequence according to X-Spanformer paper
            # Overlapping spans are allowed, but exact duplicates (same start AND end) are not
            deduplicated_annotations = self._remove_exact_duplicates(sequence_annotations)
            
            # Log deduplication results
            removed_count = len(sequence_annotations) - len(deduplicated_annotations)
            if removed_count > 0:
                logger.debug(f"Seq {seq_idx}: Removed {removed_count} exact duplicates")
            
            # Add deduplicated sequence annotations to overall list
            all_annotations.extend(deduplicated_annotations)
            
            # Write annotations incrementally for debugging
            with open(annotations_file, 'a', encoding='utf-8') as f:
                for annotation in deduplicated_annotations:
                    f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
            
            # Enhanced logging with telemetry
            logger.debug(f"Processing sequence {sequence_number} ({seq_idx}/{total_sequences}) | domain: {domain_type} | text_len: {text_length} | seq_annotations: {len(deduplicated_annotations)} | total_so_far: {len(all_annotations)}")
        
        # Final summary logging
        logger.info(f"Generated {len(all_annotations)} annotations from dictionary matching")
        logger.info(f"Saved annotations to {annotations_file}")
        
        return len(all_annotations)

    def _is_valid_span_boundary(self, text: str, start_pos: int, end_pos: int, span_text: str) -> bool:
        """
        Check if a span has valid linguistic boundaries according to X-Spanformer paper.
        
        Valid spans should:
        1. Start at word boundaries (beginning of text, after whitespace, or after punctuation)
        2. End at word boundaries (end of text, before whitespace, or before punctuation)
        3. Not be substrings within larger words
        
        Args:
            text: The full text containing the span
            start_pos: Start position of the span
            end_pos: End position of the span  
            span_text: The actual span text
            
        Returns:
            True if this is a valid linguistic span boundary
        """
        # Check start boundary
        if start_pos > 0:
            char_before = text[start_pos - 1]
            # Valid start: after whitespace, punctuation, or word boundary characters
            if not (char_before.isspace() or 
                   char_before in '.,;:!?()[]{}"\'-_/\\|`~@#$%^&*+=<>' or
                   char_before.isdigit() != span_text[0].isdigit()):  # Number/letter boundary
                return False
        
        # Check end boundary  
        if end_pos < len(text):
            char_after = text[end_pos]
            # Valid end: before whitespace, punctuation, or word boundary characters
            if not (char_after.isspace() or 
                   char_after in '.,;:!?()[]{}"\'-_/\\|`~@#$%^&*+=<>' or
                   char_after.isdigit() != span_text[-1].isdigit()):  # Number/letter boundary
                return False
        
        # Additional validation for single character spans
        if len(span_text) == 1:
            # Single letters should only be valid if they're standalone words or meaningful punctuation
            if span_text.isalpha():
                # Single letters like "a", "I" are valid if they have word boundaries
                return (start_pos == 0 or text[start_pos - 1].isspace()) and \
                       (end_pos == len(text) or text[end_pos].isspace())
            elif span_text in '.,;:!?()[]{}"\'-':
                # Punctuation is valid if it's at proper boundaries
                return True
                
        # Multi-character spans are valid if they pass boundary checks above
        return True
    
    def _remove_exact_duplicates(self, annotations: List[Dict]) -> List[Dict]:
        """
        Remove exact duplicates and invalid multi-word spans from sequence annotations 
        according to X-Spanformer paper.
        
        From Section 3.3: The factorized pointer network assumes independence between 
        start and end boundary decisions. Exact duplicates (same start AND end positions)
        violate this assumption and should be removed to maintain theoretical consistency.
        
        When multiple hierarchical levels conflict at the same position, we choose the most
        linguistically appropriate level based on the span content and structure.
        
        Args:
            annotations: List of annotation dictionaries for a single sequence
            
        Returns:
            Deduplicated and filtered list of annotations
        """
        if not annotations:
            return annotations
        
        # First filter out invalid multi-word spans at word level
        filtered = []
        for annotation in annotations:
            text = annotation.get('text', '')
            level = annotation.get('hierarchical_level', '')
            
            # Word-level spans should contain only single words (no spaces)
            if level == 'word_level' and len(text.split()) > 1:
                continue  # Skip multi-word spans at word level
                
            filtered.append(annotation)
        
        # Group annotations by exact position to resolve conflicts
        position_groups = defaultdict(list)
        for annotation in filtered:
            position_key = (annotation['start_pos'], annotation['end_pos'])
            position_groups[position_key].append(annotation)
        
        # For each position, choose the most appropriate annotation
        deduplicated = []
        for position_key, conflicting_annotations in position_groups.items():
            if len(conflicting_annotations) == 1:
                # No conflict, keep the single annotation
                deduplicated.append(conflicting_annotations[0])
            else:
                # Multiple annotations at same position - choose the most appropriate
                best_annotation = self._choose_best_hierarchical_level(conflicting_annotations)
                deduplicated.append(best_annotation)
        
        # Sort by position for consistent ordering
        deduplicated.sort(key=lambda x: (x['start_pos'], x['end_pos']))
        
        return deduplicated
    
    def _choose_best_hierarchical_level(self, conflicting_annotations: List[Dict]) -> Dict:
        """
        Choose the most appropriate hierarchical level when multiple levels conflict 
        at the same position using greedy selection.
        
        Greedy selection priority: word_level > phrase_level > clause_level
        Always prefer the lowest/most specific hierarchical level available.
        
        Args:
            conflicting_annotations: List of annotations at the same position
            
        Returns:
            The annotation with the lowest hierarchical level (highest priority)
        """
        if len(conflicting_annotations) == 1:
            return conflicting_annotations[0]
        
        # Define priority order (lower number = higher priority)
        level_priority = {'word_level': 1, 'phrase_level': 2, 'clause_level': 3}
        
        # Find annotation with highest priority (lowest number)
        best_annotation = min(conflicting_annotations, 
                             key=lambda ann: level_priority.get(ann['hierarchical_level'], 999))
        
        return best_annotation
    
    def _get_xbar_label_for_level(self, hierarchical_level: str) -> str:
        """Map hierarchical level to appropriate X-bar label."""
        level_mapping = {
            'word_level': 'noun',  # Default word-level label
            'phrase_level': 'noun_phrase',  # Default phrase-level label  
            'clause_level': 'clause'  # Default clause-level label
        }
        return level_mapping.get(hierarchical_level, 'unknown')
    
    def save_dictionaries(self, output_dir: Path) -> int:
        """
        Save dictionaries to a single dictionary.jsonl file.
        
        Args:
            output_dir: Directory to save the dictionary file
            
        Returns:
            Total number of spans saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dictionary_file = output_dir / "dictionary.jsonl"
        total_spans = 0
        
        with open(dictionary_file, 'w', encoding='utf-8') as f:
            span_id = 0
            for domain_type, levels in self.dictionaries.items():
                for hierarchical_level, spans in levels.items():
                    for span_text in sorted(spans):
                        record = {
                            "id": span_id,
                            "text": span_text,
                            "domain_type": domain_type,
                            "hierarchical_level": hierarchical_level,
                            "source": "dictionary",
                            "created_at": datetime.now().isoformat()
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        span_id += 1
                        total_spans += 1
        
        logger.info(f"Saved {total_spans} unique spans to dictionary.jsonl")
        return total_spans
    
    def load_dictionaries(self, output_dir: Path):
        """
        Load dictionaries from dictionary.jsonl file.
        
        Args:
            output_dir: Directory containing dictionary.jsonl file
        """
        output_dir = Path(output_dir)
        dictionary_file = output_dir / "dictionary.jsonl"
        
        if not dictionary_file.exists():
            logger.debug(f"No existing dictionary file found at {dictionary_file}")
            return
        
        # Reset dictionaries
        self._reset_dictionaries()
        loaded_spans = 0
        
        try:
            with open(dictionary_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        
                        domain_type = record.get("domain_type")
                        hierarchical_level = record.get("hierarchical_level") 
                        text = record.get("text")
                        
                        if domain_type and hierarchical_level and text:
                            self.add_spans(domain_type, hierarchical_level, [text])
                            loaded_spans += 1
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping invalid line {line_num} in dictionary file: {e}")
                        continue
            
            logger.info(f"Loaded {loaded_spans} spans from dictionary.jsonl")
            
        except Exception as e:
            logger.error(f"Error loading dictionary file: {e}")


# Global dictionary instance
_global_dict: Optional[XBarDictionary] = None


def get_global_dict() -> XBarDictionary:
    """Get the global dictionary instance (singleton pattern)."""
    global _global_dict
    if _global_dict is None:
        _global_dict = XBarDictionary()
    return _global_dict


def reset_global_dict():
    """Reset the global dictionary instance."""
    global _global_dict
    _global_dict = None