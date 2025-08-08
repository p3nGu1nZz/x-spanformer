"""
X-Spanformer Configuration Package

Provides configuration loaders for different components:
- judge_config_loader: Loads judge agent configurations

Note: Span annotator configuration is now handled directly in the pipeline
with DEFAULT_MODEL constant and parameters rather than external config files.
"""

from .judge_config_loader import load_judge_config

__all__ = [
    "load_judge_config"
]
