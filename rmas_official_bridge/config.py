"""
Configuration for rmas_official_bridge.

Supports two inference providers:
  --provider anthropic  (default) — uses Claude Haiku / Sonnet
  --provider groq       — uses open-source models via Groq API
                          (closest to the actual official RecursiveMAS models)

Groq model map mirrors the official HuggingFace checkpoints:
  Light  agents → llama-3.1-8b-instant        (≈ 1–3 B models: Qwen3-1.7B, Llama3.2-1B)
  Scaled agents → llama-3.3-70b-versatile     (≈ 4–9 B models: Gemma3-4B, Qwen3.5-9B)
  Math   agent  → qwen-qwq-32b                (≈ DeepSeek-R1-Distill-Qwen-1.5B reasoning)
  Code   agent  → qwen-2.5-coder-32b-preview  (≈ Qwen2.5-Coder-3B)
  Science agent → llama-3.3-70b-versatile     (≈ BioMistral-7B)
  Tool   agents → llama3-groq-70b-8192-tool-use-preview (native tool-use support)
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class CollaborationStyle(str, Enum):
    SEQUENTIAL_LIGHT  = "sequential_light"
    SEQUENTIAL_SCALED = "sequential_scaled"
    MIXTURE           = "mixture"
    DISTILLATION      = "distillation"
    DELIBERATION      = "deliberation"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    GROQ      = "groq"


@dataclass
class BridgeConfig:
    style:               CollaborationStyle = CollaborationStyle.SEQUENTIAL_LIGHT
    provider:            Provider           = Provider.ANTHROPIC
    num_recursive_rounds: int               = 3
    temperature:         float              = 0.6
    max_tokens:          int                = 1024
    verbose:             bool               = True


# ---------------------------------------------------------------------------
# Anthropic model tiers (Claude API)
# ---------------------------------------------------------------------------
ANT_LIGHT  = "claude-haiku-4-5-20251001"  # fast / cheap intermediate agents
ANT_SCALED = "claude-sonnet-4-6"          # synthesis and strong-reasoning agents

# ---------------------------------------------------------------------------
# Groq model tiers — open-source, closest to the official HF checkpoints
# Deprecation notes (Groq, July 2025):
#   qwen-qwq-32b              → replaced by qwen/qwen3-32b
#   qwen-2.5-coder-32b-preview → replaced by qwen/qwen3-32b
#   llama3-groq-70b-8192-tool-use-preview → replaced by llama-3.3-70b-versatile
# ---------------------------------------------------------------------------
GRQ_LIGHT    = "llama-3.1-8b-instant"     # ≈ Llama3.2-1B, Qwen3-1.7B
GRQ_SCALED   = "llama-3.3-70b-versatile"  # ≈ Qwen3.5-9B, Gemma3-4B
GRQ_MATH     = "qwen/qwen3-32b"           # replaces qwen-qwq-32b (decommissioned Jul 2025)
GRQ_CODE     = "qwen/qwen3-32b"           # replaces qwen-2.5-coder-32b-preview (decommissioned)
GRQ_TOOL     = "llama-3.3-70b-versatile"  # replaces llama3-groq-70b-8192-tool-use-preview


def get_style_models(style: CollaborationStyle, provider: Provider) -> dict[str, str]:
    """Return the model string for each agent role given style + provider."""
    if provider == Provider.ANTHROPIC:
        return _ANTHROPIC_STYLE_MAP[style.value]
    return _GROQ_STYLE_MAP[style.value]


_ANTHROPIC_STYLE_MAP: dict[str, dict[str, str]] = {
    "sequential_light": {
        "planner": ANT_LIGHT,   # official: Qwen3-1.7B
        "refiner": ANT_LIGHT,   # official: Llama3.2-1B
        "solver":  ANT_SCALED,  # official: Qwen2.5-Math-1.5B
    },
    "sequential_scaled": {
        "planner": ANT_SCALED,  # official: Gemma3-4B
        "refiner": ANT_SCALED,  # official: Llama3.2-3B
        "solver":  ANT_SCALED,  # official: Qwen3.5-4B
    },
    "mixture": {
        "math":       ANT_LIGHT,   # official: DeepSeek-R1-Distill-Qwen-1.5B
        "code":       ANT_LIGHT,   # official: Qwen2.5-Coder-3B
        "science":    ANT_SCALED,  # official: BioMistral-7B
        "summarizer": ANT_SCALED,  # official: Qwen3.5-2B
    },
    "distillation": {
        "expert":  ANT_SCALED,  # official: Qwen3.5-9B
        "learner": ANT_SCALED,  # official: Qwen3.5-4B
    },
    "deliberation": {
        "reflector":  ANT_SCALED,  # official: Qwen3.5-4B
        "toolcaller": ANT_SCALED,  # official: Qwen3.5-4B
    },
}

_GROQ_STYLE_MAP: dict[str, dict[str, str]] = {
    "sequential_light": {
        "planner": GRQ_LIGHT,    # Qwen3-1.7B analogue
        "refiner": GRQ_LIGHT,    # Llama3.2-1B analogue
        "solver":  GRQ_SCALED,   # Math solver needs more power
    },
    "sequential_scaled": {
        "planner": GRQ_SCALED,
        "refiner": GRQ_SCALED,
        "solver":  GRQ_SCALED,
    },
    "mixture": {
        "math":       GRQ_MATH,    # QwQ-32B — best math reasoning on Groq
        "code":       GRQ_CODE,    # Qwen-Coder — best code on Groq
        "science":    GRQ_SCALED,  # BioMistral analogue
        "summarizer": GRQ_SCALED,
    },
    "distillation": {
        "expert":  GRQ_SCALED,
        "learner": GRQ_LIGHT,    # Learner intentionally smaller
    },
    "deliberation": {
        "reflector":  GRQ_TOOL,   # Groq native tool-use model
        "toolcaller": GRQ_TOOL,   # Groq native tool-use model
    },
}
