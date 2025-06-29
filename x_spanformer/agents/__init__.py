"""
X-Spanformer Agent Components

This module provides agent classes for text evaluation and improvement:
- JudgeSession: Evaluates text segments for training data quality
- ImproveSession: Improves text segments using AI
- CritiqueSession: Combined evaluation and critique functionality (legacy)
"""

from .session import JudgeSession, ImproveSession, CritiqueSession
from .dialogue import DialogueManager
from .ollama_client import chat
from .prompts import render_prompt
from .config_loader import load_selfcrit_config

__all__ = [
    "JudgeSession",
    "ImproveSession", 
    "CritiqueSession",
    "DialogueManager",
    "chat",
    "render_prompt",
    "load_selfcrit_config"
]