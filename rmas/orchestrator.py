"""
RecursiveMAS Orchestrator

Selects the collaboration pattern graph, initialises the shared RMASState,
runs the recursive loop, and returns structured results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from rmas.config import CollaborationPattern, RMASConfig
from rmas.state import RMASState
from rmas.patterns import (
    build_sequential_graph,
    build_mixture_graph,
    build_distillation_graph,
    build_deliberation_graph,
)


@dataclass
class RMASResult:
    pattern: str
    question: str
    final_answer: str
    rounds_completed: int
    elapsed_seconds: float
    context: dict[str, Any]
    all_messages: list


class RecursiveMAS:
    """
    Main entry point for Recursive Multi-Agent System.

    Usage:
        rmas = RecursiveMAS(RMASConfig(pattern=CollaborationPattern.SEQUENTIAL, recursion_rounds=3))
        result = rmas.run("What is the capital of France?")
        print(result.final_answer)
    """

    def __init__(self, config: RMASConfig | None = None):
        self.config = config or RMASConfig()
        self._graph = self._build_graph()

    def _build_graph(self):
        builders = {
            CollaborationPattern.SEQUENTIAL: build_sequential_graph,
            CollaborationPattern.MIXTURE: build_mixture_graph,
            CollaborationPattern.DISTILLATION: build_distillation_graph,
            CollaborationPattern.DELIBERATION: build_deliberation_graph,
        }
        builder = builders[self.config.pattern]
        return builder(self.config)

    def run(self, question: str) -> RMASResult:
        """Run the RecursiveMAS on a question and return the result."""
        initial_state: RMASState = {
            "messages": [HumanMessage(content=question)],
            "round": 0,
            "max_rounds": self.config.recursion_rounds,
            "context": {},
            "pattern": self.config.pattern.value,
            "final_answer": "",
        }

        start = time.perf_counter()
        final_state = self._graph.invoke(initial_state)
        elapsed = time.perf_counter() - start

        return RMASResult(
            pattern=self.config.pattern.value,
            question=question,
            final_answer=final_state.get("final_answer", ""),
            rounds_completed=final_state.get("round", 0),
            elapsed_seconds=elapsed,
            context=final_state.get("context", {}),
            all_messages=final_state.get("messages", []),
        )

    def stream(self, question: str):
        """Stream intermediate agent outputs as they are generated."""
        initial_state: RMASState = {
            "messages": [HumanMessage(content=question)],
            "round": 0,
            "max_rounds": self.config.recursion_rounds,
            "context": {},
            "pattern": self.config.pattern.value,
            "final_answer": "",
        }
        yield from self._graph.stream(initial_state)
