# Arc

**Autonomous AI agents that work while you sleep.**

Arc is an AI agent framework where you create specialized agents, queue tasks, and wake up to results. Named agents with their own roles, LLMs, and system prompts pick up work from a persistent task board, collaborate through review loops, and deliver results via Telegram — all while you're away.

```bash
# Create agents with different LLMs
arc agent create researcher --role "Deep web research" --model ollama/llama3.2
arc agent create writer --role "Content creation" --model openai/gpt-4o

# Queue tasks at midnight
arc task add "Find top AI startups funded in 2026" --assign researcher
arc task add "Write a blog post about AI agents" --assign writer

# Start the daemon and go to sleep
arc gateway
```

## Key Features

- **[Taskforce](taskforce/overview.md)** — Named autonomous agents with per-agent LLMs, persistent task queue, multi-step workflows with review loops
- **[Browser Automation](features/browser.md)** — Real Chromium browser control via accessibility tree (not screenshots)
- **[Code Intelligence](features/code-intel.md)** — AST-aware code navigation using tree-sitter
- **[Web Dashboard](features/dashboard.md)** — Single-page app with task board, agent management, real-time logs
- **[Three-Tier Memory](features/memory.md)** — Core facts + episodic recall + session context
- **[Voice Input](features/voice.md)** — Wake word → speech-to-text → agent
- **[Skills & MCP](features/skills-mcp.md)** — Drop-in Python skills + any MCP server
- **[Scheduler](features/scheduler.md)** — Cron, interval, and one-shot scheduled tasks
- **[Workflows](features/workflows-yaml.md)** — Deterministic YAML-defined automation

## Quick Links

- [Getting Started](getting-started.md)
- [CLI Reference](cli.md)
- [Configuration](configuration.md)
- [Architecture](architecture.md)
- [GitHub Repository](https://github.com/mohit17mor/Arc)
