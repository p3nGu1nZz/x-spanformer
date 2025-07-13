"""
X-Spanformer Agent Components

This module provides agent classes for text evaluation:
- JudgeSession: Evaluates text segments for training data quality (5-judge consensus)
"""

from .session import JudgeSession
from .dialogue import DialogueManager
from .ollama_client import chat, check_ollama_connection
from .prompts import render_prompt
from .config_loader import load_judge_config

__all__ = [
    "JudgeSession",
    "DialogueManager",
    "chat",
    "check_ollama_connection",
    "render_prompt",
    "load_judge_config"
]