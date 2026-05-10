##  Setup - what packages to install

● All dependencies are declared in pyproject.toml and installed with a single command:

  - uv sync

#### The packages it installs:
```
  ┌─────────────────────────────────────────────────────┬──────────────────────────────────────────────┐
  │                       Package                       │                   Purpose                    │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ langchain, langchain-anthropic, langchain-community │ LangChain core + Anthropic + community tools │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ langgraph                                           │ Agent graph execution engine                 │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ anthropic (transitive)                              │ Claude API client                            │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ duckduckgo-search, ddgs                             │ Web search for the deliberation pattern      │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ rich                                                │ Terminal UI (panels, tables, progress)       │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ python-dotenv                                       │ Loads .env for API keys                      │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ playwright                                          │ Browser automation (deliberation pattern)    │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ pdfplumber, pypdf                                   │ PDF reading utilities                        │
  ├─────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ reportlab                                           │ PDF generation                               │
  └─────────────────────────────────────────────────────┴──────────────────────────────────────────────┘
```
  After uv sync, also run:
  uv run playwright install
  That downloads the browser binaries Playwright needs (used in the deliberation pattern).


  1. Setup (one-time)
  uv sync
  cp .env.example .env   # then add your ANTHROPIC_API_KEY

  2. Run a pattern
  uv run main.py                                        # sequential (default)
  uv run main.py --pattern mixture                      # parallel specialists → summarizer
  uv run main.py --pattern distillation                 # expert → learner
  uv run main.py --pattern deliberation                 # reflector + web search + python REPL

  3. Customize
  uv run main.py --rounds 3 --question "Your question here"
  uv run main.py --stream          # stream agent outputs as they arrive
  uv run main.py --all             # run all 4 patterns back-to-back

  What happens at runtime:

  Each pattern runs a LangGraph graph where agents pass outputs through state["context"] across recursion rounds (default: 2). The final answer comes from
  the last agent in the chain — Solver (sequential), Summarizer (mixture), Learner (distillation), or ToolCaller (deliberation).

  The deliberation pattern makes live DuckDuckGo searches, so it's the slowest and most variable. The others are pure LLM calls.


  ## what does the rounds parameter signify

● --rounds controls how many times the full agent chain loops recursively before producing the final answer.

#### For the sequential pattern with --rounds 3:
```
  * Round 1:  Planner → Critic → Solver
  * Round 2:  Planner → Critic → Solver  (each agent sees Round 1 outputs in context)
  * Round 3:  Planner → Critic → Solver  (each agent sees Round 2 outputs in context)
              └── final_answer extracted here
```
  Each round, agents receive the previous round's outputs via state["context"] and are instructed to refine rather than repeat. This simulates the paper's
  "RecursiveLink" — cross-agent latent state transfer.

  #### Practical effect:
  - --rounds 1 — single pass, fastest, cheapest
  - --rounds 2 — default, one refinement cycle
  - --rounds 3+ — more polish, but diminishing returns after ~3, and cost scales linearly (each round = N more API calls where N = number of agents in the
  pattern)

  For a simple factual question like "What is Claude Cowork?", --rounds 1 is sufficient. Higher rounds matter more for complex reasoning tasks like math
  derivations or multi-step code problems where iterative critique genuinely improves the answer.

## groq
```
 rmas/config.py
  - Added GroqModel str-enum with groq/ prefix values (LLAMA_8B ~560 t/s, LLAMA_70B ~280 t/s, GPT_OSS_20B ~1000 t/s)
  - Rewrote ROLE_MODELS with a hybrid strategy — the groq/ prefix is the routing signal:
    - Groq (fast/cheap): Planner → llama-3.1-8b-instant, Critic/CodeSpecialist/Learner/ToolCaller → llama-3.3-70b-versatile,
  MathSpecialist/ScienceSpecialist → llama-3.1-8b-instant
    - Claude Sonnet (synthesis): Solver, Summarizer, Expert, Reflector — all final-answer/synthesis roles stay on Claude
  - Changed RMASConfig.model from str = "claude-sonnet-4-6" to Optional[str] = None and added get_model(role) for per-role resolution

  rmas/agents/factory.py
  - Added _build_llm() that routes "groq/*" to ChatGroq (from langchain-groq, already in deps) and everything else to ChatAnthropic

  All 4 pattern builders (sequential, mixture, distillation, deliberation)
  - Changed model=config.model → model=config.get_model("RoleName") so each agent picks up its specific model

  main.py
  - Added --model CLI flag to override all roles with a single model (e.g. --model groq/llama-3.3-70b-versatile for all-Groq)
  - Added a warning if GROQ_API_KEY is missing
  - Config table now shows each role and its assigned provider/model before running

  Usage:
  uv run main.py                                        # hybrid: Groq intermediate + Claude synthesis
  uv run main.py --model claude-sonnet-4-6              # all-Claude override
  uv run main.py --model groq/llama-3.3-70b-versatile  # all-Groq override

```
  ## rmas_official_bridge/ — what was created

  A new package that imports the exact prompt builders from the cloned RecursiveMAS/prompts.py and runs the same five collaboration styles using either
  Claude (Anthropic) or Groq.

  File layout
```
  rmas_official_bridge/
  ├── __init__.py          # adds RecursiveMAS/ to sys.path so prompts.py is importable
  ├── config.py            # Provider enum, model mapping for both Anthropic & Groq
  ├── state.py             # BridgeState (adds mas_task for official prompt-type detection)
  ├── orchestrator.py      # OfficialBridgeMAS.run() / .stream()
  ├── agents/
  │   └── factory.py       # calls prompts.get_system_prompt() — uses official system prompt verbatim
  └── patterns/
      ├── sequential.py    # FEEDBACK_SLOT / PLANNER_SLOT replacement (outer_31, outer_12)
      ├── mixture.py       # HIE_*_EXPERT_SLOT + HIE_FEEDBACK_SLOT (all 6 outer links)
      ├── distillation.py  # DISTILL_EXPERT_SLOT + DISTILL_FEEDBACK_SLOT (outer_el, outer_le)
      └── deliberation.py  # DELIBERATION_REFLECTOR_SLOT + DELIBERATION_FEEDBACK_SLOT
```
  CLI usage

  # Add your Groq key to .env first
  # GROQ_API_KEY=gsk_...

  # Groq provider — open-source models, closest to official HF checkpoints
  uv run rmas_official_bridge/main.py --style sequential_light --provider groq
  uv run rmas_official_bridge/main.py --style mixture          --provider groq --rounds 2
  uv run rmas_official_bridge/main.py --style distillation     --provider groq
  uv run rmas_official_bridge/main.py --style deliberation     --provider groq --stream
  uv run rmas_official_bridge/main.py --all                    --provider groq

  # Anthropic provider (same as before)
  uv run rmas_official_bridge/main.py --style sequential_scaled --provider anthropic

  #### Groq model mapping (mirrors official checkpoints)
```
  ┌───────────────────────────────────┬───────────────────────────────────────┬──────────────────────────┐
  │           Style / Role            │              Groq model               │  Official HF checkpoint  │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────────┤
  │ Sequential light: Planner/Refiner │ llama-3.1-8b-instant                  │ Qwen3-1.7B / Llama3.2-1B │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────────┤
  │ Sequential: Solver                │ llama-3.3-70b-versatile               │ Qwen2.5-Math             │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────────┤
  │ Mixture: Math                     │ qwen-qwq-32b                          │ DeepSeek-R1-Distill-Qwen │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────────┤
  │ Mixture: Code                     │ qwen-2.5-coder-32b-preview            │ Qwen2.5-Coder-3B         │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────────┤
  │ Distillation: Expert              │ llama-3.3-70b-versatile               │ Qwen3.5-9B               │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────────┤
  │ Distillation: Learner             │ llama-3.1-8b-instant                  │ Qwen3.5-4B               │
  ├───────────────────────────────────┼───────────────────────────────────────┼──────────────────────────┤
  │ Deliberation: both                │ llama3-groq-70b-8192-tool-use-preview │ Qwen3.5-4B               │
  └───────────────────────────────────┴───────────────────────────────────────┴──────────────────────────┘

  ```
