"""
Distillation-Style RecursiveMAS Bridge
  Expert(scaled) → Learner(light on Groq / scaled on Anthropic)

Official key constraint: Expert outputs a PLAN ONLY — "Do not provide the
final answer." The Learner executes the plan and writes the final answer.

Slots simulate the two outer links:
  DISTILL_EXPERT_SLOT    ← Expert's plan      (outer_el: expert→learner)
  DISTILL_FEEDBACK_SLOT  ← Learner's answer   (outer_le: learner→expert)
"""
from __future__ import annotations
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from prompts import (
    DISTILL_EXPERT_SLOT,
    DISTILL_FEEDBACK_SLOT,
    build_distill_expert_prompt,
    build_distill_expert_prompt_with_feedback_slot,
    build_distill_learner_prompt_with_slot,
)

from rmas_official_bridge.state import BridgeState
from rmas_official_bridge.agents.factory import create_bridge_agent
from rmas_official_bridge.config import BridgeConfig, get_style_models


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def _question(state: BridgeState) -> str:
    return next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")


def build_distillation_graph(config: BridgeConfig):
    models = get_style_models(config.style, config.provider)
    p  = config.provider
    mt = config.max_tokens

    expert  = create_bridge_agent("expert",  "distill",
                                  model=models["expert"],  temperature=0.5,  max_tokens=mt, provider=p)
    learner = create_bridge_agent("learner", "distill",
                                  model=models["learner"], temperature=config.temperature,
                                  max_tokens=mt, provider=p)

    def expert_node(state: BridgeState) -> dict:
        ctx      = state.get("context", {})
        round_n  = state.get("round", 0)
        q        = _question(state)
        mas_task = state.get("mas_task", "math")
        prev_learner = ctx.get("learner", "")

        if round_n == 0 or not prev_learner:
            prompt = build_distill_expert_prompt(q, mas_task=mas_task)
        else:
            prompt = build_distill_expert_prompt_with_feedback_slot(
                q, mas_task=mas_task
            ).replace(DISTILL_FEEDBACK_SLOT, prev_learner)

        result  = expert.invoke({"messages": [HumanMessage(content=prompt)]})
        content = _last_ai_content(result)
        return {
            "messages": [AIMessage(content=f"[Expert Plan — Round {round_n+1}]\n{content}")],
            "context":  {**ctx, "expert": content},
        }

    def learner_node(state: BridgeState) -> dict:
        ctx      = state.get("context", {})
        round_n  = state.get("round", 0)
        q        = _question(state)
        mas_task = state.get("mas_task", "math")
        prompt   = build_distill_learner_prompt_with_slot(
            q, mas_task=mas_task
        ).replace(DISTILL_EXPERT_SLOT, ctx.get("expert", ""))

        result   = learner.invoke({"messages": [HumanMessage(content=prompt)]})
        content  = _last_ai_content(result)
        new_round = round_n + 1
        return {
            "messages": [AIMessage(content=f"[Learner Answer — Round {new_round}]\n{content}")],
            "context":  {**ctx, "learner": content},
            "round":    new_round,
            "final_answer": content,
        }

    def route(state: BridgeState) -> Literal["expert", "__end__"]:
        return "expert" if state["round"] < state["max_rounds"] else "__end__"

    g = StateGraph(BridgeState)
    g.add_node("expert",  expert_node)
    g.add_node("learner", learner_node)
    g.set_entry_point("expert")
    g.add_edge("expert", "learner")
    g.add_conditional_edges("learner", route, {"expert": "expert", "__end__": END})
    return g.compile()
