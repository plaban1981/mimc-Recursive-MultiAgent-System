from enum import Enum
from dataclasses import dataclass
from typing import Optional


class CollaborationPattern(str, Enum):
    """The 4 collaboration patterns from the RecursiveMAS paper."""
    SEQUENTIAL = "sequential"      # Planner → Critic → Solver chain
    MIXTURE = "mixture"            # Domain specialists + Summarizer
    DISTILLATION = "distillation"  # Expert + Learner
    DELIBERATION = "deliberation"  # Reflector + Tool-Caller


class GroqModel(str, Enum):
    """Groq-hosted models — ultra-fast LPU inference.
    Values carry the 'groq/' prefix so the agent factory routes them to ChatGroq.
    """
    LLAMA_8B    = "groq/llama-3.1-8b-instant"      # ~560 t/s  — cheapest, fast intermediate steps
    LLAMA_70B   = "groq/llama-3.3-70b-versatile"   # ~280 t/s  — capable, supports tool/function calling
    GPT_OSS_20B = "groq/gpt-oss-20b-0709"          # ~1000 t/s — fastest capable option


# Per-role model assignments.
# Strategy: Groq for fast/cheap intermediate agents; Claude Sonnet for synthesis and final-answer roles.
#
# Intermediate roles (Planner, Critic, Specialists, Learner):
#   → Groq — low-latency iteration; quality is amplified across recursive rounds.
# Synthesis / final-answer roles (Solver, Summarizer, Expert, Reflector, ToolCaller):
#   → Claude Sonnet — best reasoning depth, output quality, and tool-call reliability.
ROLE_MODELS: dict[str, str] = {
    # Sequential
    "Planner":           GroqModel.LLAMA_8B,     # fast decomposition
    "Critic":            GroqModel.LLAMA_70B,    # nuanced evaluation needs more capacity
    "Solver":            "claude-sonnet-4-6",    # final answer → Claude for best quality

    # Mixture
    "MathSpecialist":    GroqModel.LLAMA_8B,     # direct calculation / formula application
    "CodeSpecialist":    GroqModel.LLAMA_70B,    # code reasoning benefits from 70B
    "ScienceSpecialist": GroqModel.LLAMA_8B,     # factual domain retrieval
    "Summarizer":        "claude-sonnet-4-6",    # multi-specialist synthesis → Claude

    # Distillation
    "Expert":            "claude-sonnet-4-6",    # high-quality expert guidance
    "Learner":           GroqModel.LLAMA_70B,    # needs capacity to internalize & generalize

    # Deliberation
    "Reflector":         "claude-sonnet-4-6",    # deep inner reasoning / hypothesis formation
    "ToolCaller":        GroqModel.LLAMA_70B,    # Llama 3.3 70B supports function calling on Groq
}


@dataclass
class RMASConfig:
    """Configuration for a RecursiveMAS run."""
    pattern: CollaborationPattern = CollaborationPattern.SEQUENTIAL
    recursion_rounds: int = 3
    model: Optional[str] = None       # None = use per-role defaults from ROLE_MODELS; set to override all roles
    temperature: float = 0.7
    verbose: bool = True
    max_tokens_per_agent: int = 1024

    def get_model(self, role: str) -> str:
        """Return the model for a role.
        Resolution order: explicit config.model override → ROLE_MODELS → claude-sonnet-4-6 fallback.
        """
        if self.model:
            return self.model
        return ROLE_MODELS.get(role, "claude-sonnet-4-6")
