"""
Deliberation-Style RecursiveMAS Bridge
  Reflector ↔ ToolCaller — both agents can use tools.

Official difference from rmas/:
  The Reflector ALSO has tools (system prompt from reflector_tool_notes.py).
  Both agents share the deliberation system prompt.
  Groq uses llama3-groq-70b-8192-tool-use-preview (native tool-use support).

Slot replacement simulates the two outer links:
  DELIBERATION_REFLECTOR_SLOT ← Reflector's output  (outer_rt: reflector→toolcaller)
  DELIBERATION_FEEDBACK_SLOT  ← ToolCaller's output (outer_tr: toolcaller→reflector)
"""
from __future__ import annotations
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from prompts import (
    DELIBERATION_FEEDBACK_SLOT,
    DELIBERATION_REFLECTOR_SLOT,
    build_deliberation_reflector_prompt,
    build_deliberation_reflector_prompt_with_feedback_slot,
    build_deliberation_toolcaller_prompt,
)

from rmas_official_bridge.state import BridgeState
from rmas_official_bridge.agents.factory import create_bridge_agent
from rmas_official_bridge.config import BridgeConfig, get_style_models


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def _question(state: BridgeState) -> str:
    return next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return stdout. Use for calculations and data processing."""
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})  # noqa: S102
        return buf.getvalue() or "Executed successfully (no output)."
    except Exception as e:
        return f"Error: {e}"


def _build_search_tool():
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=4)
    return DuckDuckGoSearchRun()


def build_deliberation_graph(config: BridgeConfig):
    models = get_style_models(config.style, config.provider)
    p  = config.provider
    mt = config.max_tokens
    tools = [_build_search_tool(), python_repl]

    reflector  = create_bridge_agent("deliberation_reflector",  "deliberation",
                                     tools=tools, model=models["reflector"],
                                     temperature=0.5, max_tokens=mt, provider=p)
    toolcaller = create_bridge_agent("deliberation_toolcaller", "deliberation",
                                     tools=tools, model=models["toolcaller"],
                                     temperature=config.temperature, max_tokens=mt, provider=p)

    def reflector_node(state: BridgeState) -> dict:
        ctx      = state.get("context", {})
        round_n  = state.get("round", 0)
        q        = _question(state)
        mas_task = state.get("mas_task", "math")
        prev_tc  = ctx.get("toolcaller", "")

        if round_n == 0 or not prev_tc:
            prompt = build_deliberation_reflector_prompt(q, mas_task=mas_task)
        else:
            prompt = build_deliberation_reflector_prompt_with_feedback_slot(
                q, mas_task=mas_task
            ).replace(DELIBERATION_FEEDBACK_SLOT, prev_tc)

        result  = reflector.invoke({"messages": [HumanMessage(content=prompt)]})
        content = _last_ai_content(result)
        return {
            "messages": [AIMessage(content=f"[Reflector — Round {round_n+1}]\n{content}")],
            "context":  {**ctx, "reflector": content},
        }

    def toolcaller_node(state: BridgeState) -> dict:
        ctx      = state.get("context", {})
        round_n  = state.get("round", 0)
        q        = _question(state)
        mas_task = state.get("mas_task", "math")
        is_final = (round_n + 1) >= state.get("max_rounds", 1)
        prompt   = build_deliberation_toolcaller_prompt(
            q, reflector_signal=ctx.get("reflector", ""), mas_task=mas_task
        )
        if is_final:
            prompt += "\n\nThis is the FINAL round. Produce a complete, well-evidenced final answer."

        result   = toolcaller.invoke({"messages": [HumanMessage(content=prompt)]})
        content  = _last_ai_content(result)
        new_round = round_n + 1
        return {
            "messages": [AIMessage(content=f"[ToolCaller — Round {new_round}]\n{content}")],
            "context":  {**ctx, "toolcaller": content},
            "round":    new_round,
            "final_answer": content,
        }

    def route(state: BridgeState) -> Literal["reflector", "__end__"]:
        return "reflector" if state["round"] < state["max_rounds"] else "__end__"

    g = StateGraph(BridgeState)
    g.add_node("reflector",  reflector_node)
    g.add_node("toolcaller", toolcaller_node)
    g.set_entry_point("reflector")
    g.add_edge("reflector", "toolcaller")
    g.add_conditional_edges("toolcaller", route, {"reflector": "reflector", "__end__": END})
    return g.compile()
