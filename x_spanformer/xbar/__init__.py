"""
X-Spanformer XBar Package

This package provides X-bar theory classifier definitions and position mapping
utilities for span annotation and boundary detection.

Components:
- XBarClassifierMap: Comprehensive classifier definitions for all domains
- PositionMapper: Character-to-position mapping for contextual embeddings
"""

from .xbar_map import XBarClassifierMap, DomainType
from .position_mapper import PositionMapper, CharacterSpan, PositionSpan, parse_character_spans_from_agent_response

__all__ = [
    "XBarClassifierMap", 
    "DomainType",
    "PositionMapper", 
    "CharacterSpan", 
    "PositionSpan",
    "parse_character_spans_from_agent_response"
]
