"""
Distillation Style RecursiveMAS: Expert → Learner (→ loop)

Based on knowledge distillation in the paper (Table 1, Distillation Style).
A more capable Expert agent guides a smaller Learner agent.
Each round the Learner internalises Expert corrections and becomes more autonomous.
"""
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from rmas.state import RMASState
from rmas.agents.factory import create_agent
from rmas.agents.prompts import EXPERT_PROMPT, LEARNER_PROMPT
from rmas.config import RMASConfig


def _last_ai_content(result: dict) -> str:
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_msgs[-1].content if ai_msgs else ""


def build_distillation_graph(config: RMASConfig):
    """Build the Distillation RecursiveMAS graph (Expert→Learner recursive loop)."""
    expert = create_agent("Expert", EXPERT_PROMPT, model=config.get_model("Expert"), temperature=0.5)
    learner = create_agent("Learner", LEARNER_PROMPT, model=config.get_model("Learner"), temperature=config.temperature)

    def expert_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        round_n = state.get("round", 0)
        learner_attempt = ctx.get("Learner", "")

        prompt_extra = ""
        if learner_attempt:
            prompt_extra = (
                f"\n\nThe Learner's latest attempt (correct and elaborate where needed):\n{learner_attempt}"
            )

        msgs = state["messages"] + ([HumanMessage(content=prompt_extra)] if prompt_extra else [])
        result = expert.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["Expert"] = content
        return {
            "messages": [AIMessage(content=f"[Expert — Round {round_n+1}]\n{content}")],
            "context": new_ctx,
        }

    def learner_node(state: RMASState) -> dict:
        ctx = state.get("context", {})
        round_n = state.get("round", 0)
        expert_guidance = ctx.get("Expert", "")
        prev_attempt = ctx.get("Learner", "")

        prompt = f"Expert's guidance for this round:\n{expert_guidance}"
        if prev_attempt:
            prompt += f"\n\nYour previous attempt (improve upon it using the Expert's feedback):\n{prev_attempt}"

        msgs = state["messages"] + [HumanMessage(content=prompt)]
        result = learner.invoke({"messages": msgs})
        content = _last_ai_content(result)

        new_ctx = dict(ctx)
        new_ctx["Learner"] = content
        new_round = round_n + 1
        return {
            "messages": [AIMessage(content=f"[Learner — Round {new_round}]\n{content}")],
            "context": new_ctx,
            "round": new_round,
            "final_answer": content,
        }

    def route(state: RMASState) -> Literal["expert", "__end__"]:
        return "expert" if state.get("round", 0) < state.get("max_rounds", 1) else "__end__"

    graph = StateGraph(RMASState)
    graph.add_node("expert", expert_node)
    graph.add_node("learner", learner_node)

    graph.set_entry_point("expert")
    graph.add_edge("expert", "learner")
    graph.add_conditional_edges("learner", route, {"expert": "expert", "__end__": END})

    return graph.compile()
