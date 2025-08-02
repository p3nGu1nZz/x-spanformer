"""
X-Spanformer Configuration Package

Provides configuration loaders for different components:
- judge_config_loader: Loads judge agent configurations
- span_annotator_config_loader: Loads span annotator pipeline configurations
"""

from .judge_config_loader import load_judge_config
from .span_annotator_config_loader import load_config as load_span_annotator_config

__all__ = [
    "load_judge_config",
    "load_span_annotator_config"
]
