"""
Shared LangGraph state for rmas_official_bridge.

Identical to rmas/state.py but adds `mas_task` so prompt builders can
select the right task format (math / code / choice) automatically.
"""
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BridgeState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    round:        int           # current recursion round (0-indexed)
    max_rounds:   int
    context:      dict[str, Any]  # role → last text output (outer-link approximation)
    style:        str
    mas_task:     str           # "math" | "code" | "choice"
    final_answer: str
