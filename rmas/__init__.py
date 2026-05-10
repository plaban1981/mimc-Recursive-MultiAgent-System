"""
RecursiveMAS — Recursive Multi-Agent System
Based on: "Recursive Multi-Agent Systems" (arXiv:2604.25917v1)

Implements 4 collaboration patterns:
  - Sequential:    Planner → Critic → Solver (chain)
  - Mixture:       Domain specialists run in parallel → Summarizer
  - Distillation:  Expert guides Learner
  - Deliberation:  Reflector ↔ Tool-Caller with external tools
"""
from rmas.orchestrator import RecursiveMAS
from rmas.config import RMASConfig, CollaborationPattern, GroqModel, ROLE_MODELS

__all__ = ["RecursiveMAS", "RMASConfig", "CollaborationPattern", "GroqModel", "ROLE_MODELS"]
