"""Systematic futures signal generation."""

from .config import MODEL_VERSION, STRATEGIES
from .signals import generate_candidate_orders

__all__ = ["MODEL_VERSION", "STRATEGIES", "generate_candidate_orders"]
