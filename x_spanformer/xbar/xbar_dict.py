#!/usr/bin/env python3
"""
X-bar Dictionary Manager

Maintains domain-specific dictionaries for word-level, phrase-level, and clause-level spans.
Supports multiple domain types with extensible design for future domains like audio or vision.

The dictionary structure is:
{
    domain_type: {
        "word_level": set(),
        "phrase_level": set(), 
        "clause_level": set()
    }
}

This allows us to track unique spans across all sequences for each domain and hierarchical level,
enabling vocabulary building and span normalization across the entire corpus.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Set, List, Any, Optional
from collections import defaultdict

from .xbar_map import DomainType

# Configure logger
logger = logging.getLogger(__name__)


class XBarDictionary:
    """
    Manages domain-specific dictionaries for hierarchical X-bar spans.
    
    Maintains separate vocabularies for each domain type and hierarchical level,
    supporting extensible domain types for future multimodal expansion.
    """
    
    def __init__(self):
        """Initialize empty dictionaries for all domain types and levels."""
        # Structure: domain_type -> level -> set of unique spans
        self.dictionaries: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: {
                "word_level": set(),
                "phrase_level": set(),
                "clause_level": set()
            }
        )
        
        # Track statistics
        self.stats = {
            "total_unique_spans": 0,
            "spans_by_domain": defaultdict(int),
            "spans_by_level": defaultdict(int),
            "sequences_processed": 0
        }
    
    def add_spans(self, domain_type: str, level: str, spans: List[str]) -> int:
        """
        Add spans to the appropriate domain and level dictionary.
        
        Args:
            domain_type: Domain type (e.g., 'natural', 'code', 'mixed')
            level: Hierarchical level ('word_level', 'phrase_level', 'clause_level')
            spans: List of span text to add
            
        Returns:
            Number of new unique spans added
        """
        if level not in ["word_level", "phrase_level", "clause_level"]:
            logger.warning(f"Unknown level '{level}', skipping spans")
            return 0
        
        # Ensure domain exists
        if domain_type not in self.dictionaries:
            self.dictionaries[domain_type] = {
                "word_level": set(),
                "phrase_level": set(),
                "clause_level": set()
            }
        
        # Get current size before adding
        before_size = len(self.dictionaries[domain_type][level])
        
        # Add spans (set automatically handles uniqueness)
        for span in spans:
            if span and span.strip():  # Only add non-empty spans
                self.dictionaries[domain_type][level].add(span.strip())
        
        # Calculate new spans added
        after_size = len(self.dictionaries[domain_type][level])
        new_spans = after_size - before_size
        
        # Update statistics
        self.stats["spans_by_domain"][domain_type] += new_spans
        self.stats["spans_by_level"][level] += new_spans
        self.stats["total_unique_spans"] += new_spans
        
        if new_spans > 0:
            logger.debug(f"Added {new_spans} new spans to {domain_type}.{level} "
                        f"(total: {after_size})")
        
        return new_spans
    
    def add_sequence_spans(self, domain_type: str, word_spans: List[str], 
                          phrase_spans: List[str], clause_spans: List[str]) -> Dict[str, int]:
        """
        Add all spans from a sequence to the appropriate dictionaries.
        
        Args:
            domain_type: Domain type for the sequence
            word_spans: List of word-level spans
            phrase_spans: List of phrase-level spans  
            clause_spans: List of clause-level spans
            
        Returns:
            Dictionary with counts of new spans added per level
        """
        results = {}
        
        results["word_level"] = self.add_spans(domain_type, "word_level", word_spans)
        results["phrase_level"] = self.add_spans(domain_type, "phrase_level", phrase_spans)
        results["clause_level"] = self.add_spans(domain_type, "clause_level", clause_spans)
        
        self.stats["sequences_processed"] += 1
        
        total_new = sum(results.values())
        if total_new > 0:
            logger.debug(f"Sequence added {total_new} new spans across all levels "
                        f"for domain {domain_type}")
        
        return results
    
    def get_domain_stats(self, domain_type: str) -> Dict[str, Any]:
        """
        Get statistics for a specific domain.
        
        Args:
            domain_type: Domain to get stats for
            
        Returns:
            Dictionary with domain statistics
        """
        if domain_type not in self.dictionaries:
            return {
                "domain": domain_type,
                "word_level": 0,
                "phrase_level": 0,
                "clause_level": 0,
                "total": 0
            }
        
        domain_dict = self.dictionaries[domain_type]
        word_count = len(domain_dict["word_level"])
        phrase_count = len(domain_dict["phrase_level"])
        clause_count = len(domain_dict["clause_level"])
        
        return {
            "domain": domain_type,
            "word_level": word_count,
            "phrase_level": phrase_count,
            "clause_level": clause_count,
            "total": word_count + phrase_count + clause_count
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all domains.
        
        Returns:
            Dictionary with complete statistics
        """
        all_stats = {
            "sequences_processed": self.stats["sequences_processed"],
            "total_unique_spans": self.stats["total_unique_spans"],
            "domains": {},
            "level_totals": {
                "word_level": 0,
                "phrase_level": 0,
                "clause_level": 0
            },
            "domain_totals": {}
        }
        
        # Get stats for each domain
        for domain_type in self.dictionaries.keys():
            domain_stats = self.get_domain_stats(domain_type)
            all_stats["domains"][domain_type] = domain_stats
            all_stats["domain_totals"][domain_type] = domain_stats["total"]
            
            # Add to level totals
            all_stats["level_totals"]["word_level"] += domain_stats["word_level"]
            all_stats["level_totals"]["phrase_level"] += domain_stats["phrase_level"]
            all_stats["level_totals"]["clause_level"] += domain_stats["clause_level"]
        
        return all_stats
    
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
            logger.info(f"    word_level: {domain_stats['word_level']}")
            logger.info(f"    phrase_level: {domain_stats['phrase_level']}")
            logger.info(f"    clause_level: {domain_stats['clause_level']}")
            logger.info(f"    total: {domain_stats['total']}")
        
        logger.info("=" * 40)
    
    def save_dictionaries(self, output_dir: Path):
        """
        Save dictionaries to JSON files for persistence and analysis.
        
        Args:
            output_dir: Directory to save dictionary files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each domain dictionary separately
        for domain_type, domain_dict in self.dictionaries.items():
            domain_file = output_dir / f"xbar_dict_{domain_type}.json"
            
            # Convert sets to sorted lists for JSON serialization
            serializable_dict = {
                level: sorted(list(spans)) 
                for level, spans in domain_dict.items()
            }
            
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {domain_type} dictionary: {domain_file}")
        
        # Save comprehensive statistics
        stats_file = output_dir / "xbar_dict_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_all_stats(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dictionary statistics: {stats_file}")
    
    def load_dictionaries(self, output_dir: Path):
        """
        Load dictionaries from JSON files.
        
        Args:
            output_dir: Directory containing dictionary files
        """
        output_dir = Path(output_dir)
        
        if not output_dir.exists():
            logger.warning(f"Dictionary directory does not exist: {output_dir}")
            return
        
        # Load domain dictionaries
        dict_files = list(output_dir.glob("xbar_dict_*.json"))
        loaded_domains = 0
        
        for dict_file in dict_files:
            # Extract domain name from filename
            domain_name = dict_file.stem.replace("xbar_dict_", "")
            
            try:
                with open(dict_file, 'r', encoding='utf-8') as f:
                    domain_data = json.load(f)
                
                # Convert lists back to sets
                self.dictionaries[domain_name] = {
                    level: set(spans) 
                    for level, spans in domain_data.items()
                }
                
                loaded_domains += 1
                logger.debug(f"Loaded {domain_name} dictionary from {dict_file}")
                
            except Exception as e:
                logger.error(f"Failed to load dictionary {dict_file}: {e}")
        
        if loaded_domains > 0:
            logger.info(f"Loaded {loaded_domains} domain dictionaries")
            # Recalculate statistics
            self._recalculate_stats()
        else:
            logger.warning("No dictionary files found to load")
    
    def _recalculate_stats(self):
        """Recalculate statistics after loading dictionaries."""
        self.stats = {
            "total_unique_spans": 0,
            "spans_by_domain": defaultdict(int),
            "spans_by_level": defaultdict(int),
            "sequences_processed": 0  # This will need to be set externally
        }
        
        for domain_type, domain_dict in self.dictionaries.items():
            domain_total = 0
            for level, spans in domain_dict.items():
                span_count = len(spans)
                domain_total += span_count
                self.stats["spans_by_level"][level] += span_count
            
            self.stats["spans_by_domain"][domain_type] = domain_total
            self.stats["total_unique_spans"] += domain_total
    
    def generate_annotations_jsonl(self, output_dir: Path, sequence_metadata: Optional[Dict] = None):
        """
        Generate annotations.jsonl file from dictionaries with unique IDs.
        
        Args:
            output_dir: Output directory for annotations file
            sequence_metadata: Optional metadata about sequences processed
        """
        output_dir = Path(output_dir)
        annotations_file = output_dir / "annotations.jsonl"
        
        annotation_id = 1
        total_annotations = 0
        
        with open(annotations_file, 'w', encoding='utf-8') as f:
            for domain_type, domain_dict in self.dictionaries.items():
                for level, spans in domain_dict.items():
                    for span_text in sorted(spans):  # Sort for consistent output
                        annotation = {
                            "id": annotation_id,
                            "text": span_text,
                            "xbar_label": self._get_default_label_for_level(level),
                            "domain_type": domain_type,
                            "hierarchical_level": level,
                            "source": "dictionary"
                        }
                        
                        f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
                        annotation_id += 1
                        total_annotations += 1
        
        logger.info(f"Generated {total_annotations} dictionary-based annotations: {annotations_file}")
    
    def _get_default_label_for_level(self, level: str) -> str:
        """Get a default label for dictionary entries based on hierarchical level."""
        level_defaults = {
            "word_level": "word",
            "phrase_level": "phrase", 
            "clause_level": "clause"
        }
        return level_defaults.get(level, "unknown")


# Global instance for use across the pipeline
global_xbar_dict = XBarDictionary()


def get_global_dict() -> XBarDictionary:
    """Get the global X-bar dictionary instance."""
    return global_xbar_dict


def reset_global_dict():
    """Reset the global dictionary (useful for testing)."""
    global global_xbar_dict
    global_xbar_dict = XBarDictionary()
