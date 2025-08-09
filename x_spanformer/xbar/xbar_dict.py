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