"""
Agent factory — create_agent() builds a LangGraph ReAct deep agent.

Supports two inference backends selected by the model ID prefix:
  - "groq/<model>"  → ChatGroq  (Groq LPU — ultra-fast, for intermediate roles)
  - anything else   → ChatAnthropic (Claude — highest quality, for synthesis roles)

The "deep" aspect maps to RecursiveMAS's recursive latent refinement:
- Each agent is wrapped in an outer recursive loop (controlled by orchestrator)
- Agents receive accumulated cross-agent context (simulated RecursiveLink outer transfer)
- Inner refinement is handled by the model's chain-of-thought reasoning
"""
from typing import Optional
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

_GROQ_PREFIX = "groq/"


def _build_llm(model: str, temperature: float, max_tokens: int):
    """Instantiate ChatGroq or ChatAnthropic based on the 'groq/' prefix."""
    if model.startswith(_GROQ_PREFIX):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model[len(_GROQ_PREFIX):],
            # Groq converts temperature=0 to 1e-8 internally; clamp to avoid warnings
            temperature=max(temperature, 1e-8),
            max_tokens=max_tokens,
        )
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_agent(
    role: str,
    description: str,
    tools: Optional[list[BaseTool]] = None,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 1024,
):
    """
    Create a RecursiveMAS deep agent using LangGraph's create_react_agent.

    Args:
        role:        Agent role name (e.g. "Planner", "Expert")
        description: Role-specific system prompt describing responsibilities
        tools:       Optional tools (e.g. search, python REPL) for the agent
        model:       Model ID — Anthropic ("claude-*") or Groq ("groq/<model-id>")
        temperature: Sampling temperature
        max_tokens:  Max tokens per response

    Returns:
        A compiled LangGraph ReAct agent (CompiledStateGraph)
    """
    llm = _build_llm(model, temperature, max_tokens)
    tools = tools or []

    system_prompt = (
        f"You are the **{role}** agent in a Recursive Multi-Agent System (RecursiveMAS).\n\n"
        f"{description}\n\n"
        "--- Recursive Collaboration Protocol ---\n"
        "You operate in iterative recursion rounds. Each round you receive:\n"
        "  • The original problem\n"
        "  • Accumulated context (thoughts) from peer agents in previous rounds\n"
        "  • Your own previous output (if any)\n"
        "Use this recursive context to progressively deepen and refine your response.\n"
        "Be concise, focused, and improve upon previous rounds rather than repeating them."
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
