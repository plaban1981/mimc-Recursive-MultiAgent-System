"""
Sequential Style RecursiveMAS: Planner → Critic → Solver → (loop)

Based on the "chain-of-agents" pattern from the paper (Table 1, Sequential Style).
Three agents with complementary roles progressively decompose, judge, and solve.
Each recursion round feeds the previous round's outputs back as context (outer RecursiveLink).
"""
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from rmas.state import RMASState
from rmas.agents.factory import create_agent
from rmas.agents.prompts import PLANNER_PROMPT, CRITIC_PROMPT, SOLVER_PROMPT
from rmas.config import RMASConfig


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def _context_block(ctx: dict, exclude: str = "") -> str:
    parts = [f"[{role}]: {thought}" for role, thought in ctx.items() if role != exclude]
    return "\n\n".join(parts) if parts else ""


def build_sequential_graph(config: RMASConfig):
    """Build the Sequential RecursiveMAS graph (Planner→Critic→Solver recursive loop)."""
    planner = create_agent("Planner", PLANNER_PROMPT, model=config.get_model("Planner"), temperature=config.temperature)
    critic = create_agent("Critic", CRITIC_PROMPT, model=config.get_model("Critic"), temperature=config.temperature)
    solver = create_agent("Solver", SOLVER_PROMPT, model=config.get_model("Solver"), temperature=config.temperature)

    def planner_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        round_n = state.get("round", 0)

        extra = ""
        if round_n > 0 and ctx:
            extra = (
                f"\n\n[Round {round_n} context — refine your plan based on this]\n"
                + _context_block(ctx, exclude="Planner")
            )

        msgs = state["messages"] + ([HumanMessage(content=extra)] if extra else [])
        result = planner.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["Planner"] = content
        return {"messages": [AIMessage(content=f"[Planner — Round {round_n+1}]\n{content}")], "context": new_ctx}

    def critic_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        plan = ctx.get("Planner", "")
        prev_solve = ctx.get("Solver", "")

        critique_request = f"Planner's current plan:\n{plan}"
        if prev_solve:
            critique_request += f"\n\nPrevious Solver answer to critique:\n{prev_solve}"

        msgs = state["messages"] + [HumanMessage(content=critique_request)]
        result = critic.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["Critic"] = content
        return {"messages": [AIMessage(content=f"[Critic — Round {state.get('round',0)+1}]\n{content}")], "context": new_ctx}

    def solver_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        plan = ctx.get("Planner", "")
        critique = ctx.get("Critic", "")
        prev = ctx.get("Solver", "")

        solve_prompt = (
            f"Use the plan and critique below to produce your best answer.\n\n"
            f"Plan:\n{plan}\n\nCritique:\n{critique}"
        )
        if prev:
            solve_prompt += f"\n\nYour previous answer (improve upon it):\n{prev}"

        msgs = state["messages"] + [HumanMessage(content=solve_prompt)]
        result = solver.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["Solver"] = content
        new_round = state.get("round", 0) + 1
        return {
            "messages": [AIMessage(content=f"[Solver — Round {new_round}]\n{content}")],
            "context": new_ctx,
            "round": new_round,
            "final_answer": content,
        }

    def route(state: RMASState) -> Literal["planner", "__end__"]:
        return "planner" if state.get("round", 0) < state.get("max_rounds", 1) else "__end__"

    graph = StateGraph(RMASState)
    graph.add_node("planner", planner_node)
    graph.add_node("critic", critic_node)
    graph.add_node("solver", solver_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "critic")
    graph.add_edge("critic", "solver")
    graph.add_conditional_edges("solver", route, {"planner": "planner", "__end__": END})

    return graph.compile()
