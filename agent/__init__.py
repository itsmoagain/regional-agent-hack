"""Agent-facing helpers for insight and recommendation generation."""

from __future__ import annotations

from .insight_engine import build_insights
from .rag_retriever import generate_recommendations, retrieve_context
from .rule_engine import apply_rules

__all__ = [
    "apply_rules",
    "build_insights",
    "generate_recommendations",
    "retrieve_context",
]
