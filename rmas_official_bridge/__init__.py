"""
rmas_official_bridge
====================
Runs the five RecursiveMAS collaboration styles using the **exact prompts
and slot-based context structure** from the official RecursiveMAS repository
(github.com/RecursiveMAS/RecursiveMAS), with two inference backends:

  Provider.ANTHROPIC  → Claude Haiku / Sonnet  (requires ANTHROPIC_API_KEY)
  Provider.GROQ       → Open-source models via Groq API (requires GROQ_API_KEY)
                        Closest analogue to the actual HF checkpoints:
                        QwQ-32B (math), Qwen-Coder-32B (code), Llama-70B (reasoning)

The official prompts are imported directly from the cloned repo so this
package stays in sync with the official release automatically.
"""
import sys
from pathlib import Path

_OFFICIAL_REPO = Path(__file__).resolve().parent.parent / "RecursiveMAS"
if str(_OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_REPO))

from .orchestrator import OfficialBridgeMAS, BridgeResult
from .config import BridgeConfig, CollaborationStyle, Provider

__all__ = [
    "OfficialBridgeMAS", "BridgeResult",
    "BridgeConfig", "CollaborationStyle", "Provider",
]
