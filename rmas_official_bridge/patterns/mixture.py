"""
Mixture-Style RecursiveMAS Bridge
  Math + Code + Science → Summarizer

Groq assignment (closest to official checkpoints):
  Math    → qwen-qwq-32b           (≈ DeepSeek-R1-Distill-Qwen-1.5B reasoning)
  Code    → qwen-2.5-coder-32b-preview (≈ Qwen2.5-Coder-3B)
  Science → llama-3.3-70b-versatile    (≈ BioMistral-7B)
  Summary → llama-3.3-70b-versatile

Slot replacement simulates bidirectional outer links:
  HIE_*_EXPERT_SLOT ← specialist outputs   (outer_1s/2s/3s: specialist→summarizer)
  HIE_FEEDBACK_SLOT ← Summarizer output    (outer_s1/s2/s3: summarizer→specialist)
"""
from __future__ import annotations
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from prompts import (
    HIE_MATH_EXPERT_SLOT,
    HIE_CODE_EXPERT_SLOT,
    HIE_SCIENCE_EXPERT_SLOT,
    HIE_FEEDBACK_SLOT,
    build_hie_expert_prompt,
    build_hie_expert_prompt_with_feedback_slot,
    build_hie_summarizer_prompt_with_slots,
)

from rmas_official_bridge.state import BridgeState
from rmas_official_bridge.agents.factory import create_bridge_agent
from rmas_official_bridge.config import BridgeConfig, get_style_models


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def _question(state: BridgeState) -> str:
    return next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")


def _make_specialist_node(agent, role_key: str, hie_role: str):
    def node(state: BridgeState) -> dict:
        ctx      = state.get("context", {})
        round_n  = state.get("round", 0)
        q        = _question(state)
        mas_task = state.get("mas_task", "math")
        if round_n == 0:
            prompt = build_hie_expert_prompt(q, hie_role=hie_role, mas_task=mas_task)
        else:
            prompt = build_hie_expert_prompt_with_feedback_slot(
                q, hie_role=hie_role, mas_task=mas_task
            ).replace(HIE_FEEDBACK_SLOT, ctx.get("summarizer", ""))
        result  = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        content = _last_ai_content(result)
        return {
            "messages": [AIMessage(content=f"[{hie_role} — Round {round_n+1}]\n{content}")],
            "context":  {**ctx, role_key: content},
        }
    return node


def build_mixture_graph(config: BridgeConfig):
    models = get_style_models(config.style, config.provider)
    p  = config.provider
    t  = config.temperature
    mt = config.max_tokens

    math_agent    = create_bridge_agent("hie_math_expert",    "hie",
                                        model=models["math"],       temperature=t, max_tokens=mt, provider=p)
    code_agent    = create_bridge_agent("hie_code_expert",    "hie",
                                        model=models["code"],       temperature=t, max_tokens=mt, provider=p)
    science_agent = create_bridge_agent("hie_science_expert", "hie",
                                        model=models["science"],    temperature=t, max_tokens=mt, provider=p)
    summarizer    = create_bridge_agent("summarizer",         "hie",
                                        model=models["summarizer"], temperature=t, max_tokens=mt, provider=p)

    def summarizer_node(state: BridgeState) -> dict:
        ctx      = state.get("context", {})
        round_n  = state.get("round", 0)
        q        = _question(state)
        mas_task = state.get("mas_task", "math")
        prompt = (
            build_hie_summarizer_prompt_with_slots(q, mas_task=mas_task)
            .replace(HIE_MATH_EXPERT_SLOT,    ctx.get("math",    ""))
            .replace(HIE_CODE_EXPERT_SLOT,    ctx.get("code",    ""))
            .replace(HIE_SCIENCE_EXPERT_SLOT, ctx.get("science", ""))
        )
        result   = summarizer.invoke({"messages": [HumanMessage(content=prompt)]})
        content  = _last_ai_content(result)
        new_round = round_n + 1
        return {
            "messages": [AIMessage(content=f"[Summarizer — Round {new_round}]\n{content}")],
            "context":  {**ctx, "summarizer": content},
            "round":    new_round,
            "final_answer": content,
        }

    def route(state: BridgeState) -> Literal["math", "__end__"]:
        return "math" if state["round"] < state["max_rounds"] else "__end__"

    g = StateGraph(BridgeState)
    g.add_node("math",       _make_specialist_node(math_agent,    "math",    "hie_math_expert"))
    g.add_node("code",       _make_specialist_node(code_agent,    "code",    "hie_code_expert"))
    g.add_node("science",    _make_specialist_node(science_agent, "science", "hie_science_expert"))
    g.add_node("summarizer", summarizer_node)
    g.set_entry_point("math")
    g.add_edge("math", "code"); g.add_edge("code", "science"); g.add_edge("science", "summarizer")
    g.add_conditional_edges("summarizer", route, {"math": "math", "__end__": END})
    return g.compile()
