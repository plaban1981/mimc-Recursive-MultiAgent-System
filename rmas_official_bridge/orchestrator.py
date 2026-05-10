"""
OfficialBridgeMAS — entry point for rmas_official_bridge.

Infers mas_task (math / code / choice) from the question text so the
official prompt builders select the right format automatically.
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from rmas_official_bridge.config import BridgeConfig, CollaborationStyle
from rmas_official_bridge.state import BridgeState
from rmas_official_bridge.patterns import (
    build_sequential_graph,
    build_mixture_graph,
    build_distillation_graph,
    build_deliberation_graph,
)


@dataclass
class BridgeResult:
    style:            str
    question:         str
    final_answer:     str
    rounds_completed: int
    elapsed_seconds:  float
    context:          dict[str, Any]
    all_messages:     list


def _infer_mas_task(question: str) -> str:
    """Match official dataset task types: math | code | choice."""
    q = question.lower()
    code_kw = ("implement", "write a function", "write code", "algorithm",
                "program", "def ", "class ", "pseudocode")
    if any(kw in q for kw in code_kw):
        return "code"
    if re.search(r"^\s*[A-D]\s*[\.\):\-]\s+", question, re.MULTILINE):
        return "choice"
    return "math"


_GRAPH_BUILDERS = {
    CollaborationStyle.SEQUENTIAL_LIGHT:  build_sequential_graph,
    CollaborationStyle.SEQUENTIAL_SCALED: build_sequential_graph,
    CollaborationStyle.MIXTURE:           build_mixture_graph,
    CollaborationStyle.DISTILLATION:      build_distillation_graph,
    CollaborationStyle.DELIBERATION:      build_deliberation_graph,
}


class OfficialBridgeMAS:
    """
    Run RecursiveMAS collaboration styles using the official prompts,
    backed by either the Anthropic Claude API or Groq API.
    """

    def __init__(self, config: BridgeConfig | None = None):
        self.config = config or BridgeConfig()
        self._graph = _GRAPH_BUILDERS[self.config.style](self.config)

    def run(self, question: str) -> BridgeResult:
        mas_task      = _infer_mas_task(question)
        initial_state: BridgeState = {
            "messages":     [HumanMessage(content=question)],
            "round":        0,
            "max_rounds":   self.config.num_recursive_rounds,
            "context":      {},
            "style":        self.config.style.value,
            "mas_task":     mas_task,
            "final_answer": "",
        }
        start = time.perf_counter()
        final = self._graph.invoke(initial_state)
        elapsed = time.perf_counter() - start
        return BridgeResult(
            style=self.config.style.value,
            question=question,
            final_answer=final.get("final_answer", ""),
            rounds_completed=final.get("round", 0),
            elapsed_seconds=elapsed,
            context=final.get("context", {}),
            all_messages=final.get("messages", []),
        )

    def stream(self, question: str):
        mas_task      = _infer_mas_task(question)
        initial_state: BridgeState = {
            "messages":     [HumanMessage(content=question)],
            "round":        0,
            "max_rounds":   self.config.num_recursive_rounds,
            "context":      {},
            "style":        self.config.style.value,
            "mas_task":     mas_task,
            "final_answer": "",
        }
        yield from self._graph.stream(initial_state)
