# Arc

**Micro-agents you can teach, share, and compose.**

Arc is a Python framework for building personal AI agents that run entirely on your machine. It uses local LLMs via Ollama, costs nothing to run, and remembers things about you across sessions — without sending data anywhere.

---

## What it does

- **Talks to local LLMs** — Ollama only, zero API cost, fully offline
- **Uses tools** — reads/writes files, runs terminal commands, searches the web, fetches URLs
- **Remembers you** — three-tier memory that persists across sessions and gets smarter over time
- **Stays safe** — every destructive tool call asks for your approval first
- **Learns new tricks** — drop a `.py` skill file in `~/.arc/skills/` and it's auto-loaded on restart

---

## Installation

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) running locally

```bash
# 1. Clone
git clone https://github.com/your-username/arc.git
cd arc

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install
pip install -e ".[dev]"

# 4. Pull a model in Ollama (if you haven't already)
ollama pull llama3.2

# 5. First-time setup
arc init

# 6. Start chatting
arc chat
```

> The first run downloads the embedding model (~25 MB, BAAI/bge-small-en-v1.5) for long-term memory. Subsequent starts are instant.

---

## Architecture

Arc is built around a **micro-kernel** — a small coordinator that wires together independent subsystems via an event bus. No subsystem imports another directly; everything goes through the kernel.

```
┌─────────────────────────────────────────────────────────┐
│                        CLI Platform                      │
│           (streaming output, approval prompts)           │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                       Agent Loop                         │
│         COMPOSE → THINK → ACT → OBSERVE → repeat        │
└──┬──────────────┬──────────────┬──────────────┬─────────┘
   │              │              │              │
┌──▼───┐   ┌──────▼──────┐  ┌───▼────┐  ┌──────▼──────┐
│ LLM  │   │   Memory    │  │ Skills │  │  Security   │
│Ollama│   │  (3 tiers)  │  │Manager │  │   Engine    │
└──────┘   └──────┬──────┘  └───┬────┘  └─────────────┘
                  │              │
           ┌──────▼──────┐  ┌───▼──────────────────────┐
           │  Session    │  │  Built-in Skills           │
           │  Episodic   │  │  filesystem / terminal /   │
           │  Core Facts │  │  web_search / web_read     │
           └─────────────┘  └──────────────────────────-┘
```

### Core components

| Component | Location | Role |
|---|---|---|
| **Kernel** | `arc/core/kernel.py` | Event bus + provider registry + lifecycle |
| **Agent Loop** | `arc/agent/loop.py` | think → act → observe cycle, max-iterations synthesis |
| **Context Composer** | `arc/memory/context.py` | Assembles messages across all 3 memory tiers within token budget |
| **LLM Provider** | `arc/llm/ollama.py` | Streaming Ollama client with tool-call support |
| **Skill Manager** | `arc/skills/manager.py` | Registers skills, dispatches tool calls |
| **Security Engine** | `arc/security/engine.py` | Capability policy + interactive approval flow |
| **Identity / Soul** | `arc/identity/` | Personality profiles, first-run wizard |
| **Middleware** | `arc/middleware/` | Cost tracking, structured event logging |

---

## Memory System

Arc has a **three-tier memory** architecture. All storage is local SQLite — no cloud, no embeddings API.

```
Tier 3 — Core Memory      (SQLite)
  Stable facts about you: name, projects, preferences.
  Always injected into every system prompt.
  Updated by LLM-driven distillation every 5 turns.

Tier 2 — Episodic Memory  (SQLite + sqlite-vec)
  Semantic chunks from past conversations.
  Retrieved by vector similarity for each new message.
  Ranked by: 0.7 × similarity + 0.2 × recency + 0.1 × frequency.

Tier 1 — Session Memory   (RAM)
  Current conversation turns.
  Token-budget capped — oldest turns dropped first.
```

- Embeddings: `BAAI/bge-small-en-v1.5` via fastembed (384-dim, ONNX, ~25 MB, offline after first download)
- All memory operations are **async fire-and-forget** — they never block a response
- Single DB at `~/.arc/memory/memory.db`, shared across all platforms (future: Telegram, WhatsApp, etc.)

### Memory commands (inside `arc chat`)

```
/memory              — show all core facts
/memory episodic     — show recent episodic memories
/memory forget <id>  — delete a core fact by its id
```

---

## Skills

Skills are collections of tools the agent can call. Arc auto-discovers them at startup.

### Built-in skills

| Skill | Tools |
|---|---|
| **filesystem** | `read_file`, `write_file`, `list_directory` |
| **terminal** | `run_command` (bash / PowerShell) |
| **browsing** | `web_search`, `web_read`, `http_get` |

### Adding a custom skill

Drop a `.py` file in `~/.arc/skills/` that defines a class inheriting from `Skill`. It's auto-loaded on restart — no config changes needed.

```python
# ~/.arc/skills/my_skill.py
from arc.skills.base import Skill, tool
from arc.core.types import SkillManifest, ToolResult

class MySkill(Skill):
    def manifest(self) -> SkillManifest:
        return SkillManifest(name="my_skill", version="1.0.0",
                             description="Does my custom thing")

    @tool
    async def my_tool(self, message: str) -> ToolResult:
        return ToolResult(success=True, output=f"Got: {message}")
```

### Soft skills (no Python)

Drop a `.md` file in `~/.arc/skills/`. Its content is injected into every system prompt — useful for domain knowledge, style rules, or persona tweaks.

---

## Security

Every tool has a declared set of **capabilities** (e.g. `FILE_WRITE`, `SHELL_EXEC`, `NETWORK`). The security engine checks each call against a policy before execution:

- **auto_allow** — low-risk reads run silently
- **always_ask** — destructive operations prompt you every time
- **never_allow** — blocked entirely (configurable)
- **remembered decisions** — "allow always" / "deny always" are persisted per tool

---

## CLI commands

```
arc init          First-time setup wizard (model, personality, name)
arc chat          Start an interactive chat session
arc chat -m <model>   Override the Ollama model for this session
arc logs          Show today's log
arc config        Show current configuration
arc version       Show version
```

Inside `arc chat`:

```
/help             List all commands
/skills           Show loaded skills and their tools
/memory           Show long-term core facts
/memory episodic  Show recent episodic memories
/cost             Show token usage for this session
/clear            Clear conversation history
/exit             Exit
```

---

## Configuration

Config lives at `~/.arc/config.toml` (created by `arc init`):

```toml
[llm]
default_model = "llama3.2"
base_url = "http://localhost:11434"

[agent]
max_iterations = 25
temperature = 0.7

[security]
default_policy = "ask"
```

Identity and personality are in `~/.arc/identity.md`.

---

## Development

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arc --cov-report=term-missing

