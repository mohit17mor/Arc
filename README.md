<h1 align="center">Arc</h1>
<h3 align="center">Autonomous AI agents that work while you sleep.</h3>

<p align="center">
  <img src="public/assets/arc.png" width="320" alt="Arc" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Python-blue?style=flat-square" alt="Python" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT" />
  <img src="https://img.shields.io/badge/tests-930%2B%20passing-brightgreen?style=flat-square" alt="Tests" />
  <img src="https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square" alt="Platforms" />
</p>

---

Arc is an AI agent framework where you create specialized agents, queue tasks, and wake up to results. Named agents with their own roles, LLMs, and system prompts pick up work from a persistent task board, collaborate through review loops, and deliver results via Telegram — all while you're away.

```bash
# Create agents with different LLMs
arc agent create researcher --role "Deep web research" --model ollama/llama3.2
arc agent create writer --role "Content creation" --model openai/gpt-4o

# Queue 5 tasks at midnight
arc task add "Find top AI startups funded in 2026" --assign researcher
arc task add "Compare Rust vs Go for backend services" --assign researcher
arc task add "Write a blog post about AI agents" --assign writer

# Start the daemon and go to sleep
arc gateway

# Wake up to Telegram notifications with results
```

---

## What Makes Arc Different

### 🤖 Taskforce — Named Autonomous Agents
Not one agent doing everything. A **team** of specialized agents, each with its own role, LLM, and personality. Define them once, assign tasks forever.

```toml
# ~/.arc/agents/researcher.toml
name = "researcher"
role = "AI industry analyst"
max_concurrent = 1

[llm]
provider = "ollama"
model = "llama3.2"

system_prompt = """
You are a senior research analyst. Always cite sources.
Search at least 3 sources for every claim.
Structure output as: Executive Summary → Key Findings → Sources.
"""
```

Each agent can use a **different LLM** — cheap local models for research, powerful cloud models for creative work. Supports Ollama, OpenAI, Groq, OpenRouter, Together, LM Studio, and any OpenAI-compatible endpoint.

### 📋 Persistent Task Queue
Tasks survive across restarts. Queue 10 tasks at night, close your laptop, results arrive in the morning.

- **Priority ordering** — urgent tasks run first
- **Task dependencies** — "write the blog post AFTER the research is done"
- **Human-in-the-loop** — tasks pause and wait for your input when needed
- **Full audit trail** — every agent action is logged as comments on the task

### 🔄 Multi-Step Workflows with Review
Chain agents together. The output of one step feeds into the next. Add reviewers at any step — AI or human.

```bash
# Researcher gathers info → Writer drafts → You review before publishing
arc task add "Create a blog post about AI trends" \
  --step researcher \
  --step writer --review-by human
```

When the reviewer bounces work back, it goes to the agent who submitted it — with specific feedback. Automatic bounce limits prevent infinite loops.

### 🧠 Code Intelligence
AST-aware code navigation using tree-sitter. Agents understand codebases structurally, not just as text files.

- **`repo_map`** — condensed project overview with all classes/functions/methods
- **`find_symbol`** — locate any definition and get the full source body
- **`search_code`** — grep with AST context (shows the enclosing function/class scope)

Supports Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, and 7 more languages.

### 📊 Web Dashboard
Full single-page dashboard at `http://localhost:18789`:

- **Dashboard** — task counts, agent status, uptime
- **Chat** — real-time WebChat with streaming responses
- **Task Board** — Kanban view, create tasks, approve/revise, view comments
- **Agents** — create/manage agents with LLM picker and system prompt editor
- **Scheduler** — view and cancel scheduled jobs
- **Skills & MCP** — all loaded skills with tools, MCP server status
- **Logs** — real-time system event stream with source/type filtering

One WebSocket connection stays alive across all tabs. Notifications appear everywhere.

---

## Quick Start

```bash
git clone https://github.com/mohit17mor/Arc.git && cd Arc
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
playwright install chromium
arc init                                          # setup wizard
arc chat                                          # interactive chat
```

To run the full daemon with task processing + dashboard:
```bash
arc gateway                                       # http://localhost:18789
```

---

## 📖 Documentation

**[Full Documentation →](https://mohit17mor.github.io/Arc)**

- [Getting Started](https://mohit17mor.github.io/Arc/getting-started/)
- [Taskforce — Agents, Tasks, Workflows](https://mohit17mor.github.io/Arc/taskforce/overview/)
- [All Features](https://mohit17mor.github.io/Arc/features/browser/)
- [Architecture](https://mohit17mor.github.io/Arc/architecture/)
- [CLI Reference](https://mohit17mor.github.io/Arc/cli/)
- [Configuration](https://mohit17mor.github.io/Arc/configuration/)

---

## Development

```bash
pip install -e ".[dev]"
pytest                    # 930+ tests
pytest --cov=arc          # with coverage
```

---

## License

MIT
