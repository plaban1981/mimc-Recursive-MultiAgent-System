"""
Mixture Style RecursiveMAS: Domain Specialists → Summarizer (→ loop)

Based on Mixture-of-Experts collaboration in the paper.
Math, Code, and Science specialists reason over the input in parallel (executed
sequentially here for API compatibility), then a Summarizer synthesizes them.
"""
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from rmas.state import RMASState
from rmas.agents.factory import create_agent
from rmas.agents.prompts import (
    MATH_SPECIALIST_PROMPT,
    CODE_SPECIALIST_PROMPT,
    SCIENCE_SPECIALIST_PROMPT,
    SUMMARIZER_PROMPT,
)
from rmas.config import RMASConfig


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def _specialist_node(agent, role: str):
    def node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        round_n = state.get("round", 0)

        extra = ""
        if round_n > 0 and role in ctx:
            extra = f"\n\nYour previous analysis (refine it):\n{ctx[role]}"

        msgs = state["messages"] + ([HumanMessage(content=extra)] if extra else [])
        result = agent.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx[role] = content
        return {
            "messages": [AIMessage(content=f"[{role} — Round {round_n+1}]\n{content}")],
            "context": new_ctx,
        }
    return node


def build_mixture_graph(config: RMASConfig):
    """Build the Mixture RecursiveMAS graph (specialists → summarizer recursive loop)."""
    math_agent = create_agent("MathSpecialist", MATH_SPECIALIST_PROMPT, model=config.get_model("MathSpecialist"), temperature=config.temperature)
    code_agent = create_agent("CodeSpecialist", CODE_SPECIALIST_PROMPT, model=config.get_model("CodeSpecialist"), temperature=config.temperature)
    science_agent = create_agent("ScienceSpecialist", SCIENCE_SPECIALIST_PROMPT, model=config.get_model("ScienceSpecialist"), temperature=config.temperature)
    summarizer = create_agent("Summarizer", SUMMARIZER_PROMPT, model=config.get_model("Summarizer"), temperature=config.temperature)

    def summarizer_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        round_n = state.get("round", 0)

        specialist_ctx = "\n\n".join(
            f"[{role}]:\n{thought}"
            for role, thought in ctx.items()
            if role in ("MathSpecialist", "CodeSpecialist", "ScienceSpecialist")
        )

        prev = ctx.get("Summarizer", "")
        prompt = f"Specialist analyses:\n\n{specialist_ctx}"
        if prev:
            prompt += f"\n\nYour previous synthesis (improve upon it):\n{prev}"

        msgs = state["messages"] + [HumanMessage(content=prompt)]
        result = summarizer.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["Summarizer"] = content
        new_round = round_n + 1
        return {
            "messages": [AIMessage(content=f"[Summarizer — Round {new_round}]\n{content}")],
            "context": new_ctx,
            "round": new_round,
            "final_answer": content,
        }

    def route(state: RMASState) -> Literal["math", "__end__"]:
        return "math" if state.get("round", 0) < state.get("max_rounds", 1) else "__end__"

    graph = StateGraph(RMASState)
    graph.add_node("math", _specialist_node(math_agent, "MathSpecialist"))
    graph.add_node("code", _specialist_node(code_agent, "CodeSpecialist"))
    graph.add_node("science", _specialist_node(science_agent, "ScienceSpecialist"))
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("math")
    graph.add_edge("math", "code")
    graph.add_edge("code", "science")
    graph.add_edge("science", "summarizer")
    graph.add_conditional_edges("summarizer", route, {"math": "math", "__end__": END})

    return graph.compile()
