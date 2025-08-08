"""
X-Spanformer XBar Package

This package provides X-bar theory label definitions for span annotation.

Components:
- XBarLabelMap: Unified label definitions for all domains
- XBarAnnotator: Main annotation logic
"""

from .xbar_map import XBarLabelMap, DomainType
from .xbar_annotator import XBarAnnotator, ModelConfig

__all__ = [
    "XBarLabelMap", 
    "DomainType",
    "XBarAnnotator",
    "ModelConfig"
]
