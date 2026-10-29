"""Dependency-free evidence analysis primitives."""
from .core import analyze_hypotheses, find_contradictions, suggest_next_test

__all__ = ["analyze_hypotheses", "find_contradictions", "suggest_next_test"]
