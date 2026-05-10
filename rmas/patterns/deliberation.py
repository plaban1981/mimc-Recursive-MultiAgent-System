"""
Deliberation Style RecursiveMAS: Reflector ↔ Tool-Caller (→ loop)

Based on inner-thinking + tool-integrated deliberation from the paper.
The Reflector reasons deeply without tools; the Tool-Caller retrieves external
information via DuckDuckGo search and Python execution. They exchange, critique,
and refine until reaching consensus, after which Tool-Caller produces final answer.
"""
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from rmas.state import RMASState
from rmas.agents.factory import create_agent
from rmas.agents.prompts import REFLECTOR_PROMPT, TOOL_CALLER_PROMPT
from rmas.config import RMASConfig


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output. Use for calculations and data processing."""
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})  # noqa: S102
        return buf.getvalue() or "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {e}"


def _build_search_tool():
    if os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch
        return TavilySearch(max_results=5)
    return DuckDuckGoSearchRun()


def build_deliberation_graph(config: RMASConfig):
    """Build the Deliberation RecursiveMAS graph (Reflector↔Tool-Caller recursive loop)."""
    tools = [_build_search_tool(), python_repl]

    reflector = create_agent("Reflector", REFLECTOR_PROMPT, model=config.get_model("Reflector"), temperature=0.5)
    tool_caller = create_agent("ToolCaller", TOOL_CALLER_PROMPT, tools=tools, model=config.get_model("ToolCaller"), temperature=config.temperature)

    def reflector_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        round_n = state.get("round", 0)
        tool_findings = ctx.get("ToolCaller", "")

        prompt = ""
        if tool_findings:
            prompt = f"\n\nTool-Caller's retrieved findings:\n{tool_findings}\n\nReflect on these findings. What's still uncertain? What hypothesis do you form?"

        msgs = state["messages"] + ([HumanMessage(content=prompt)] if prompt else [])
        result = reflector.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["Reflector"] = content
        return {
            "messages": [AIMessage(content=f"[Reflector — Round {round_n+1}]\n{content}")],
            "context": new_ctx,
        }

    def tool_caller_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        round_n = state.get("round", 0)
        reflection = ctx.get("Reflector", "")
        prev = ctx.get("ToolCaller", "")

        is_final = (round_n + 1) >= state.get("max_rounds", 1)
        prompt = f"Reflector's analysis:\n{reflection}"
        if prev:
            prompt += f"\n\nYour previous findings:\n{prev}"
        if is_final:
            prompt += "\n\nThis is the FINAL round. Produce a complete, well-evidenced final answer."

        msgs = state["messages"] + [HumanMessage(content=prompt)]
        result = tool_caller.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["ToolCaller"] = content
        new_round = round_n + 1
        return {
            "messages": [AIMessage(content=f"[ToolCaller — Round {new_round}]\n{content}")],
            "context": new_ctx,
            "round": new_round,
            "final_answer": content,
        }

    def route(state: RMASState) -> Literal["reflector", "__end__"]:
        return "reflector" if state.get("round", 0) < state.get("max_rounds", 1) else "__end__"

    graph = StateGraph(RMASState)
    graph.add_node("reflector", reflector_node)
    graph.add_node("tool_caller", tool_caller_node)

    graph.set_entry_point("reflector")
    graph.add_edge("reflector", "tool_caller")
    graph.add_conditional_edges("tool_caller", route, {"reflector": "reflector", "__end__": END})

    return graph.compile()
