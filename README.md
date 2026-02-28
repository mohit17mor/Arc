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
- **Delegates work** — spawns background worker agents for long-running tasks so you keep chatting
- **Runs on a schedule** — set recurring or one-time jobs that fire automatically and notify you
- **Notifies you** — results delivered in-chat, to a log file, or via Telegram

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
┌─────────────────────────────────────────────────────────────┐
│                        CLI Platform                          │
│         (streaming output, approval prompts, /commands)      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                       Agent Loop  (main)                     │
│         COMPOSE → THINK → ACT → OBSERVE → repeat            │
└──┬──────────────┬──────────────┬──────────────┬─────────────┘
   │              │              │              │
┌──▼───┐   ┌──────▼──────┐  ┌───▼────┐  ┌──────▼──────┐
│ LLM  │   │   Memory    │  │ Skills │  │  Security   │
│Ollama│   │  (3 tiers)  │  │Manager │  │   Engine    │
└──────┘   └──────┬──────┘  └───┬────┘  └─────────────┘
                  │              │
           ┌──────▼──────┐  ┌───▼──────────────────────────┐
           │  Session    │  │  Built-in Skills               │
           │  Episodic   │  │  filesystem / terminal /       │
           │  Core Facts │  │  web_search / web_read /       │
           └─────────────┘  │  worker / scheduler            │
                            └──────────────────────────────-─┘
                                        │
              ┌─────────────────────────▼──────────────────────┐
              │            Multi-Agent Layer                    │
              │                                                 │
              │  ┌─────────────────┐   ┌────────────────────┐  │
              │  │  Worker Agents  │   │  Scheduler Engine  │  │
              │  │ (VirtualPlatform│   │  (cron / interval  │  │
              │  │  fire-and-forget│   │   / oneshot jobs)  │  │
              │  └────────┬────────┘   └─────────┬──────────┘  │
              │           └──────────┬────────────┘            │
              │              ┌───────▼────────┐                │
              │              │ Notification   │                │
              │              │    Router      │                │
              │              │ CLI / File /   │                │
              │              │   Telegram     │                │
              └──────────────└────────────────┘────────────────┘
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
| **Worker Skill** | `arc/skills/builtin/worker.py` | Spawns background agents, delivers results via notifications |
| **Scheduler Engine** | `arc/scheduler/engine.py` | Fires cron / interval / oneshot jobs in the background |
| **Notification Router** | `arc/notifications/router.py` | Delivers job results to CLI, file, Telegram |
| **Agent Registry** | `arc/agent/registry.py` | Tracks running workers for status and clean shutdown |
| **Virtual Platform** | `arc/platforms/virtual/app.py` | Silent in-process platform for background agents |

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

## Multi-Agent Workers

The main agent can **delegate sub-tasks to background worker agents** and continue the conversation immediately. Workers run silently on a `VirtualPlatform`, use the same tools as the main agent, and deliver results back as notifications when done.

```
User: "Research the latest AI news and summarise it"
Arc:  "I've started a background worker for that. I'll share the results when it's done."
          ↓  (worker runs in background — web search, reads URLs)
Arc:  "Here's what the worker found: ..."   ← delivered automatically when ready
```

### How delegation works

The main agent decides whether to do work inline or delegate based on clear rules in its system prompt:

**Do it inline** when: single web search + quick answer, simple lookup or calculation.

**Delegate** when: many tool calls, parallel research across multiple topics, explicitly long-running tasks (analysing a whole codebase, monitoring something), or when the user wants to keep chatting while it runs.

### Watching worker activity

Open a second terminal while `arc chat` is running:

```bash
arc workers              # show last 40 lines of activity then exit
arc workers --follow     # live-tail updates as workers run (Ctrl-C to stop)
arc workers -n 100       # show last 100 lines
```

Example output:
```
14:30:00 | research_ai   | SPAWNED    | research_ai_news
14:30:01 | research_ai   | THINKING   | iter=1
14:30:02 | research_ai   | TOOL CALL  | web_search(query="AI news today")
14:30:04 | research_ai   | TOOL DONE  | ✓ Found 8 results about...
14:30:07 | research_ai   | COMPLETE   | ✓
```

All worker activity is written to `~/.arc/worker_activity.log`. The main chat window stays clean.

---

## Scheduler

Arc can run tasks automatically on a schedule — recurring or one-time. Scheduled jobs use the same notification pipeline as workers, so results appear in the chat window when you're next active.

### Setting up a scheduled job

Just ask in natural language:

```
"Remind me every weekday at 9am to check my downloads"
"Fetch the latest AI news every morning at 8am"
"Remind me in 2 hours to take a break"
"Check my project's GitHub stars every day at noon"
```

Arc uses the `schedule_job` tool internally with three trigger types:

| Trigger | When to use | Example |
|---|---|---|
| `oneshot` | Single future event | "remind me in 1 hour", "alert me at 6pm" |
| `cron` | Recurring on a schedule | "every weekday at 9am" → `0 9 * * 1-5` |
| `interval` | Repeat every N seconds | "check every 30 minutes" |

**`use_tools`**: set to `true` when the job needs live data (web search, file read). Leave `false` for reminders or anything the LLM can answer from its own knowledge.

### Managing jobs (inside `arc chat`)

```
/jobs                    — list all scheduled jobs with next run time
/jobs cancel <name>      — cancel a scheduled job by name
```

Or ask the agent: *"cancel my morning_news job"*, *"what jobs do I have scheduled?"*

Completed one-time (`oneshot`) jobs are automatically deleted. Recurring jobs advance their `next_run` after each execution.

### Scheduler activity in `arc workers --follow`

Scheduler jobs with `use_tools=true` also appear in the worker activity log:
```
08:00:00 | morning_news  | SPAWNED    | morning_news
08:00:01 | morning_news  | THINKING   | iter=1
08:00:02 | morning_news  | TOOL CALL  | web_search(query="AI news today")
08:00:08 | morning_news  | COMPLETE   | ✓
```

---

## Notifications

When a worker or scheduled job finishes, the result is delivered through the **notification router**:

1. **In-chat** (always) — the main agent presents the result naturally when you're next active, or immediately if you're idle
2. **File** — appended to `~/.arc/notifications.log`
3. **Telegram** — sent to your Telegram chat (optional, configure below)

---

## Skills

Skills are collections of tools the agent can call. Arc auto-discovers them at startup.

### Built-in skills

| Skill | Tools | Description |
|---|---|---|
| **filesystem** | `read_file`, `write_file`, `list_directory` | Local file operations |
| **terminal** | `run_command` | Run shell commands (bash / PowerShell) |
| **browsing** | `web_search`, `web_read`, `http_get` | Web research |
| **worker** | `delegate_task`, `list_workers` | Spawn and track background agents |
| **scheduler** | `schedule_job`, `list_jobs`, `cancel_job` | Manage scheduled tasks |

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

Worker agents run with a **permissive security policy** by default (no approval prompts) since they run in the background. The scheduler's sub-agents also use permissive mode.

---

## CLI commands

```
arc init               First-time setup wizard (model, personality, name)
arc chat               Start an interactive chat session
arc chat -m <model>    Override the Ollama model for this session
arc workers            Show recent worker and scheduler activity
arc workers --follow   Live-tail worker activity in a second terminal
arc workers -n <N>     Show last N lines of activity
arc logs               Show today's log
arc config             Show current configuration
arc version            Show version
```

Inside `arc chat`:

```
/help                  List all commands
/skills                Show loaded skills and their tools
/memory                Show long-term core facts
/memory episodic       Show recent episodic memories
/memory forget <id>    Delete a core fact by its id
/jobs                  List all scheduled jobs with next run time
/jobs cancel <name>    Cancel a scheduled job
/cost                  Show token usage for this session
/perms                 Show remembered security permissions
/clear                 Clear conversation history
/exit                  Exit
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

[scheduler]
enabled = true
db_path = "~/.arc/scheduler.db"

[telegram]
# Optional — fill in to receive notifications via Telegram
token = ""
chat_id = ""
```

Identity and personality are in `~/.arc/identity.md`.

---

## Development

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arc --cov-report=term-missing
```

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

