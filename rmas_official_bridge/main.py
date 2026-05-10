#!/usr/bin/env python3
"""
rmas_official_bridge — CLI
Based on arXiv:2604.25917v1  (official RecursiveMAS repository prompts)

Uses the exact prompt builders from RecursiveMAS/prompts.py and supports
two inference providers:

  --provider anthropic  Claude Haiku / Sonnet  (ANTHROPIC_API_KEY)
  --provider groq       Open-source via Groq   (GROQ_API_KEY)
                        Closest to the actual RecursiveMAS HF checkpoints

Usage:
  uv run rmas_official_bridge/main.py --style sequential_light
  uv run rmas_official_bridge/main.py --style mixture --provider groq
  uv run rmas_official_bridge/main.py --style distillation --rounds 2
  uv run rmas_official_bridge/main.py --style deliberation --provider groq --stream
  uv run rmas_official_bridge/main.py --all --provider groq
"""
import os
import sys
import argparse
from pathlib import Path

# Ensure the project root (one level up) is on sys.path so the package works
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text
from rich import box
from langchain_core.messages import AIMessage

load_dotenv()

console = Console()

STYLE_COLORS = {
    "sequential_light":  "cyan",
    "sequential_scaled": "bright_cyan",
    "mixture":           "magenta",
    "distillation":      "yellow",
    "deliberation":      "green",
}

STYLE_DESCRIPTIONS = {
    "sequential_light":  "Planner(Haiku/Llama-8B) → Refiner → Solver  [light agents]",
    "sequential_scaled": "Planner(Sonnet/Llama-70B) → Refiner → Solver [scaled agents]",
    "mixture":           "Math(QwQ) + Code(Coder) + Science → Summarizer",
    "distillation":      "Expert(plan-only) → Learner(executes plan)",
    "deliberation":      "Reflector ↔ ToolCaller  [both have search + Python tools]",
}

PROVIDER_MODELS = {
    "anthropic": {
        "sequential_light":  "Planner=Haiku · Refiner=Haiku · Solver=Sonnet",
        "sequential_scaled": "Planner=Sonnet · Refiner=Sonnet · Solver=Sonnet",
        "mixture":           "Math/Code=Haiku · Science/Summarizer=Sonnet",
        "distillation":      "Expert=Sonnet · Learner=Sonnet",
        "deliberation":      "Reflector=Sonnet · ToolCaller=Sonnet",
    },
    "groq": {
        "sequential_light":  "Planner=Llama-3.1-8B · Refiner=Llama-3.1-8B · Solver=Llama-3.3-70B",
        "sequential_scaled": "All=Llama-3.3-70B-Versatile",
        "mixture":           "Math=QwQ-32B · Code=Qwen-Coder-32B · Science/Sum=Llama-70B",
        "distillation":      "Expert=Llama-70B · Learner=Llama-8B",
        "deliberation":      "Reflector=Llama3-Groq-70B-Tool · ToolCaller=Llama3-Groq-70B-Tool",
    },
}

DEMO_QUESTIONS = {
    "sequential_light":  "Explain how backpropagation works and derive the gradient update rule for a two-layer neural network.",
    "sequential_scaled": "Prove that the sum of the first n odd numbers equals n². Show all reasoning steps.",
    "mixture":           "Design an efficient algorithm to find all prime pairs (p, p+2) up to 10 million. Discuss the math, code, and scientific applications.",
    "distillation":      "What is the intuition behind the attention mechanism in transformers, and how does self-attention differ from cross-attention?",
    "deliberation":      "What are the most recent breakthroughs in fusion energy research, and what technical barriers remain?",
}


def print_header(provider: str):
    console.print()
    console.print(Panel(
        "[bold white]RecursiveMAS Official Bridge[/bold white]\n"
        "[dim]Official prompts from RecursiveMAS/prompts.py  •  arXiv:2604.25917v1[/dim]\n\n"
        f"[italic]Provider: [bold]{provider.upper()}[/bold]  — "
        "Anthropic Claude API  or  Groq open-source models[/italic]",
        box=box.DOUBLE_EDGE, border_style="bright_blue", expand=False, padding=(1, 4),
    ))
    console.print()


def print_config(style: str, provider: str, rounds: int, question: str):
    color = STYLE_COLORS[style]
    table = Table(box=box.ROUNDED, border_style="dim", show_header=False, padding=(0, 1))
    table.add_column("Key",   style="dim")
    table.add_column("Value", style="bright_white")
    table.add_row("Style",    f"[{color}]{style.upper()}[/]  {STYLE_DESCRIPTIONS[style]}")
    table.add_row("Provider", provider.upper())
    table.add_row("Models",   PROVIDER_MODELS[provider][style])
    table.add_row("Rounds",   str(rounds))
    table.add_row("Question", Text(question[:120] + ("..." if len(question) > 120 else ""),
                                   style="italic"))
    console.print(table)
    console.print()


def run_and_display(style: str, provider: str, rounds: int, question: str):
    from rmas_official_bridge import OfficialBridgeMAS, BridgeConfig, CollaborationStyle
    from rmas_official_bridge.config import Provider

    color  = STYLE_COLORS[style]
    config = BridgeConfig(
        style=CollaborationStyle(style),
        provider=Provider(provider),
        num_recursive_rounds=rounds,
    )
    mas = OfficialBridgeMAS(config)

    console.print(Rule(f"[{color}]Running {style.upper()}[/]", style=color))
    console.print()

    with console.status(f"[{color}]Agents collaborating ({rounds} round{'s' if rounds!=1 else ''})…[/]"):
        result = mas.run(question)

    stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    stats.add_column("k", style="dim")
    stats.add_column("v", style="bright_white")
    stats.add_row("Rounds completed", str(result.rounds_completed))
    stats.add_row("Time elapsed",     f"{result.elapsed_seconds:.2f}s")
    stats.add_row("Agents used",      str(len(result.context)))
    console.print(stats)
    console.print()

    if result.context:
        console.print(f"[{color}]Agent Contributions:[/]")
        for role, thought in result.context.items():
            preview = thought[:350].replace("\n", " ") + ("…" if len(thought) > 350 else "")
            console.print(Panel(preview, title=f"[dim]{role}[/]",
                                border_style="dim", padding=(0, 1)))

    console.print()
    console.print(Panel(
        result.final_answer or "[dim]No answer generated[/dim]",
        title=f"[bold {color}]Final Answer ({style.upper()})[/bold {color}]",
        border_style=color, padding=(1, 2),
    ))
    console.print()
    return result


def stream_and_display(style: str, provider: str, rounds: int, question: str):
    from rmas_official_bridge import OfficialBridgeMAS, BridgeConfig, CollaborationStyle
    from rmas_official_bridge.config import Provider

    color  = STYLE_COLORS[style]
    config = BridgeConfig(
        style=CollaborationStyle(style),
        provider=Provider(provider),
        num_recursive_rounds=rounds,
    )
    mas = OfficialBridgeMAS(config)

    console.print(Rule(f"[{color}]Streaming {style.upper()}[/]", style=color))
    console.print()

    seen = set()
    for chunk in mas.stream(question):
        for _node, node_state in chunk.items():
            for msg in node_state.get("messages", []):
                if isinstance(msg, AIMessage) and msg.content not in seen:
                    seen.add(msg.content)
                    first, *rest = msg.content.split("\n")
                    body = "\n".join(rest).strip()
                    console.print(Panel(
                        body or first,
                        title=f"[{color}]{first}[/]" if body else f"[{color}]Agent Output[/]",
                        border_style=color, padding=(0, 1),
                    ))
                    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="RecursiveMAS Official Bridge — official prompts, Claude or Groq inference"
    )
    parser.add_argument(
        "--style",
        choices=["sequential_light", "sequential_scaled", "mixture", "distillation", "deliberation"],
        default="sequential_light",
        help="Collaboration style (matches official run.py --style)",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "groq"], default="anthropic",
        help="Inference provider (anthropic=Claude API, groq=open-source via Groq)",
    )
    parser.add_argument("--rounds",   type=int, default=2, help="Recursion rounds (default: 2)")
    parser.add_argument("--question", type=str, default="",  help="Question to ask")
    parser.add_argument("--all",      action="store_true",   help="Run all 5 styles sequentially")
    parser.add_argument("--stream",   action="store_true",   help="Stream agent outputs")
    args = parser.parse_args()

    # Check required API key
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not set in .env")
        raise SystemExit(1)
    if args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]Error:[/bold red] GROQ_API_KEY not set in .env")
        raise SystemExit(1)

    print_header(args.provider)

    styles = (
        ["sequential_light", "sequential_scaled", "mixture", "distillation", "deliberation"]
        if args.all else [args.style]
    )

    for style in styles:
        question = args.question or DEMO_QUESTIONS[style]
        print_config(style, args.provider, args.rounds, question)
        if args.stream:
            stream_and_display(style, args.provider, args.rounds, question)
        else:
            run_and_display(style, args.provider, args.rounds, question)
        if args.all and style != styles[-1]:
            console.print(Rule(style="dim"))
            console.print()


if __name__ == "__main__":
    main()
