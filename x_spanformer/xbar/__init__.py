"""
X-Spanformer XBar Package

This package provides X-bar theory label definitions for span annotation.

Components:
- XBarLabelMap: Unified label definitions for all domains
- XBarAnnotator: Main annotation logic
- AnnotationAnalyzer: Analysis and reporting for annotation data
"""

from .xbar_map import XBarLabelMap, DomainType
from .xbar_annotator import XBarAnnotator, ModelConfig
from .analyze_annotations import AnnotationAnalyzer, analyze_annotations
from .xbar_dict import XBarDictionary, get_global_dict
from .validate_annotations import AnnotationValidator, ValidationReport, ValidationIssue, validate_annotations

__all__ = [
    "XBarLabelMap", 
    "DomainType",
    "XBarAnnotator",
    "ModelConfig",
    "AnnotationAnalyzer",
    "analyze_annotations",
    "XBarDictionary",
    "get_global_dict",
    "AnnotationValidator",
    "ValidationReport", 
    "ValidationIssue",
    "validate_annotations"
]
