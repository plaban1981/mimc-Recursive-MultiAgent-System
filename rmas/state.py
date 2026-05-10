from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RMASState(TypedDict):
    """
    Shared state flowing through the RecursiveMAS agent graph.

    - messages:      full conversation (managed by add_messages reducer)
    - round:         current recursion round (0-indexed, incremented after each full pass)
    - max_rounds:    total recursion rounds configured
    - context:       per-agent "latent thoughts" accumulated across rounds
                     (simulates RecursiveLink cross-agent latent state transfer)
    - pattern:       which collaboration pattern is active
    - final_answer:  extracted final answer after all rounds complete
    """
    messages: Annotated[list[BaseMessage], add_messages]
    round: int
    max_rounds: int
    context: dict[str, Any]
    pattern: str
    final_answer: str
