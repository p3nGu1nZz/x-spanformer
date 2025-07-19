"""
Vocabulary Statistics Schema

This module defines the schema for vocabulary induction statistics and provides
utilities for formatting and validating vocabulary metrics.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import math


@dataclass
class VocabStats:
    """
    Schema for vocabulary induction statistics.
    
    Attributes:
        total_pieces: Final number of vocabulary pieces after EM + pruning
        baseline_ppl: Initial perplexity before EM iterations
        final_ppl: Final perplexity after EM iterations
        oov_rate: Out-of-vocabulary rate (0.0 = perfect coverage, 1.0 = no coverage)
        coverage: Corpus coverage percentage (0.0 to 1.0)
        em_iterations: Number of EM iterations performed
        pruned_pieces: Number of pieces removed during pruning
        baseline_oov: Initial OOV rate before EM iterations
    """
    total_pieces: int
    baseline_ppl: float
    final_ppl: float
    oov_rate: float
    coverage: float
    em_iterations: int
    pruned_pieces: int
    baseline_oov: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VocabStats':
        """Create VocabStats from dictionary."""
        return cls(
            total_pieces=data['total_pieces'],
            baseline_ppl=data['baseline_ppl'],
            final_ppl=data['final_ppl'],
            oov_rate=data['oov_rate'],
            coverage=data.get('coverage', 0.0),
            em_iterations=data['em_iterations'],
            pruned_pieces=data['pruned_pieces'],
            baseline_oov=data.get('baseline_oov', 0.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_pieces': self.total_pieces,
            'baseline_ppl': self.baseline_ppl,
            'final_ppl': self.final_ppl,
            'oov_rate': self.oov_rate,
            'coverage': self.coverage,
            'em_iterations': self.em_iterations,
            'pruned_pieces': self.pruned_pieces,
            'baseline_oov': self.baseline_oov
        }
    
    def format_perplexity(self, ppl: float) -> str:
        """Format perplexity for display with proper handling of large numbers."""
        if not math.isfinite(ppl):
            return "∞"
        elif ppl >= 1e6:
            return f"{ppl:.2e}"
        else:
            return f"{ppl:.2f}"
    
    def format_percentage(self, rate: float) -> str:
        """Format rate as percentage."""
        return f"{rate * 100:.4f}%"
    
    def get_ppl_improvement(self) -> float:
        """Calculate perplexity improvement (negative = worse, positive = better)."""
        return self.baseline_ppl - self.final_ppl
    
    def get_ppl_improvement_percent(self) -> float:
        """Calculate perplexity improvement as percentage."""
        if self.baseline_ppl == 0:
            return 0.0
        return ((self.baseline_ppl - self.final_ppl) / self.baseline_ppl) * 100
    
    def is_perfect_coverage(self) -> bool:
        """Check if vocabulary achieves perfect coverage (OOV = 0)."""
        return self.oov_rate == 0.0
    
    def summary(self) -> str:
        """Generate a human-readable summary of vocabulary statistics."""
        lines = []
        lines.append("VOCABULARY INDUCTION SUMMARY")
        lines.append("=" * 40)
        lines.append(f"Final vocabulary size: {self.total_pieces:,} pieces")
        lines.append(f"Pieces pruned: {self.pruned_pieces:,}")
        lines.append(f"EM iterations: {self.em_iterations}")
        lines.append("")
        lines.append("PERPLEXITY METRICS:")
        lines.append(f"  Baseline PPL: {self.format_perplexity(self.baseline_ppl)}")
        lines.append(f"  Final PPL: {self.format_perplexity(self.final_ppl)}")
        
        improvement = self.get_ppl_improvement()
        improvement_pct = self.get_ppl_improvement_percent()
        if improvement > 0:
            lines.append(f"  Improvement: -{self.format_perplexity(improvement)} ({improvement_pct:.2f}% better)")
        elif improvement < 0:
            lines.append(f"  Degradation: +{self.format_perplexity(-improvement)} ({-improvement_pct:.2f}% worse)")
        else:
            lines.append("  No change in perplexity")
        
        lines.append("")
        lines.append("COVERAGE METRICS:")
        lines.append(f"  OOV rate: {self.format_percentage(self.oov_rate)}")
        lines.append(f"  Coverage: {self.format_percentage(self.coverage)}")
        
        if self.is_perfect_coverage():
            lines.append("  ✅ Perfect coverage achieved!")
        else:
            lines.append(f"  ⚠️  {self.format_percentage(self.oov_rate)} of positions uncovered")
        
        return "\n".join(lines)


def load_vocab_stats(filepath: str) -> VocabStats:
    """Load vocabulary statistics from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return VocabStats.from_dict(data)


def save_vocab_stats(stats: VocabStats, filepath: str) -> None:
    """Save vocabulary statistics to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)


def validate_vocab_stats(stats: Dict[str, Any]) -> bool:
    """
    Validate vocabulary statistics dictionary.
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = {
        'total_pieces': int,
        'baseline_ppl': (int, float),
        'final_ppl': (int, float),
        'oov_rate': (int, float),
        'em_iterations': int,
        'pruned_pieces': int
    }
    
    for field, expected_type in required_fields.items():
        if field not in stats:
            print(f"Missing required field: {field}")
            return False
        
        if not isinstance(stats[field], expected_type):
            print(f"Invalid type for {field}: expected {expected_type}, got {type(stats[field])}")
            return False
    
    # Validate ranges
    if stats['oov_rate'] < 0 or stats['oov_rate'] > 1:
        print(f"Invalid OOV rate: {stats['oov_rate']} (must be 0.0-1.0)")
        return False
    
    if stats['total_pieces'] < 0:
        print(f"Invalid total_pieces: {stats['total_pieces']} (must be >= 0)")
        return False
    
    if stats['em_iterations'] < 0:
        print(f"Invalid em_iterations: {stats['em_iterations']} (must be >= 0)")
        return False
    
    return True
