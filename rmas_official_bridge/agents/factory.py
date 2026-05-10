"""
Agent factory for rmas_official_bridge.

Supports two providers:
  - anthropic : ChatAnthropic (requires ANTHROPIC_API_KEY)
  - groq      : ChatGroq      (requires GROQ_API_KEY)

The system prompt is taken verbatim from the official RecursiveMAS
prompts.get_system_prompt() — returns the deliberation tool-use prompt
for deliberation roles, and "You are a helpful assistant." for all others.
"""
from __future__ import annotations
from typing import Optional

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

# Official repo on sys.path via __init__.py
from prompts import get_system_prompt

from rmas_official_bridge.config import Provider


def _make_llm(provider: Provider, model: str, temperature: float, max_tokens: int):
    if provider == Provider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)
    # Default: Anthropic
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)


def create_bridge_agent(
    role:        str,
    mas_design:  str                    = "chain",
    tools:       Optional[list[BaseTool]] = None,
    model:       str                    = "claude-haiku-4-5-20251001",
    temperature: float                  = 0.6,
    max_tokens:  int                    = 1024,
    provider:    Provider               = Provider.ANTHROPIC,
):
    """
    Create an agent using the official RecursiveMAS system prompt.

    role       : matches official role naming convention
    mas_design : "chain" | "hie" | "distill" | "deliberation"
    provider   : anthropic (Claude) | groq (open-source via Groq API)
    """
    system_prompt = get_system_prompt(mas_design=mas_design, mas_role=role)
    llm = _make_llm(provider, model, temperature, max_tokens)
    return create_react_agent(model=llm, tools=tools or [], prompt=system_prompt)
