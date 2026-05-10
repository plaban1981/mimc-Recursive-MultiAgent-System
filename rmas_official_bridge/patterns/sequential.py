"""
Sequential-Style RecursiveMAS Bridge
  Light  → Planner(light) → Refiner(light) → Solver(scaled)
  Scaled → Planner(scaled) → Refiner(scaled) → Solver(scaled)

Uses official prompt builders from RecursiveMAS/prompts.py.
Slot replacement simulates the outer RecursiveLink tensor injection:
  FEEDBACK_SLOT  ← Solver's previous output   (outer_31: solver→planner)
  PLANNER_SLOT   ← Planner's output           (outer_12: planner→refiner)
  REFINED_SLOT   is handled inside build_math_solver_prompt directly
"""
from __future__ import annotations
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from prompts import (
    FEEDBACK_SLOT,
    PLANNER_SLOT,
    build_math_planner_prompt,
    build_math_planner_prompt_with_feedback_slot,
    build_math_refiner_prompt_with_slot,
    build_math_solver_prompt,
)

from rmas_official_bridge.state import BridgeState
from rmas_official_bridge.agents.factory import create_bridge_agent
from rmas_official_bridge.config import BridgeConfig, get_style_models


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def _question(state: BridgeState) -> str:
    return next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")


def build_sequential_graph(config: BridgeConfig):
    models = get_style_models(config.style, config.provider)
    p = config.provider
    t = config.temperature
    mt = config.max_tokens

    planner = create_bridge_agent("planner", "chain", model=models["planner"],
                                  temperature=t, max_tokens=mt, provider=p)
    refiner = create_bridge_agent("refiner", "chain", model=models["refiner"],
                                  temperature=t, max_tokens=mt, provider=p)
    solver  = create_bridge_agent("solver",  "chain", model=models["solver"],
                                  temperature=t, max_tokens=mt, provider=p)

    def planner_node(state: BridgeState) -> dict:
        ctx     = state.get("context", {})
        round_n = state.get("round", 0)
        q       = _question(state)
        if round_n == 0:
            prompt = build_math_planner_prompt(q)
        else:
            prompt = build_math_planner_prompt_with_feedback_slot(q).replace(
                FEEDBACK_SLOT, ctx.get("solver", "")
            )
        result  = planner.invoke({"messages": [HumanMessage(content=prompt)]})
        content = _last_ai_content(result)
        ctx2    = {**ctx, "planner": content}
        return {"messages": [AIMessage(content=f"[Planner — Round {round_n+1}]\n{content}")],
                "context": ctx2}

    def refiner_node(state: BridgeState) -> dict:
        ctx     = state.get("context", {})
        round_n = state.get("round", 0)
        q       = _question(state)
        prompt  = build_math_refiner_prompt_with_slot(q).replace(
            PLANNER_SLOT, ctx.get("planner", "")
        )
        result  = refiner.invoke({"messages": [HumanMessage(content=prompt)]})
        content = _last_ai_content(result)
        ctx2    = {**ctx, "refiner": content}
        return {"messages": [AIMessage(content=f"[Refiner — Round {round_n+1}]\n{content}")],
                "context": ctx2}

    def solver_node(state: BridgeState) -> dict:
        ctx     = state.get("context", {})
        round_n = state.get("round", 0)
        q       = _question(state)
        prompt  = build_math_solver_prompt(q, refined_plan=ctx.get("refiner", ""))
        result  = solver.invoke({"messages": [HumanMessage(content=prompt)]})
        content = _last_ai_content(result)
        ctx2    = {**ctx, "solver": content}
        new_round = round_n + 1
        return {"messages": [AIMessage(content=f"[Solver — Round {new_round}]\n{content}")],
                "context": ctx2, "round": new_round, "final_answer": content}

    def route(state: BridgeState) -> Literal["planner", "__end__"]:
        return "planner" if state["round"] < state["max_rounds"] else "__end__"

    g = StateGraph(BridgeState)
    g.add_node("planner", planner_node)
    g.add_node("refiner", refiner_node)
    g.add_node("solver",  solver_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "refiner")
    g.add_edge("refiner", "solver")
    g.add_conditional_edges("solver", route, {"planner": "planner", "__end__": END})
    return g.compile()
