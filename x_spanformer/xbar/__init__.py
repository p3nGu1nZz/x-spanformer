"""
X-Spanformer XBar Package

This package provides X-bar theory label definitions for span annotation.

Components:
- XBarLabelMap: Unified label definitions for all domains
- XBarAnnotator: Main annotation logic
- XBarJsonParser: JSON parsing and repair for LLM responses
"""

from .xbar_map import XBarLabelMap, DomainType
from .xbar_annotator import XBarAnnotator, ModelConfig
from .xbar_json import XBarJsonParser, parse_json_response, attempt_json_repair, filter_valid_annotations

__all__ = [
    "XBarLabelMap", 
    "DomainType",
    "XBarAnnotator",
    "ModelConfig",
    "XBarJsonParser",
    "parse_json_response",
    "attempt_json_repair", 
    "filter_valid_annotations"
]
