# Arc

Arc is an AI workspace with two modes:

- a **main agent** you talk to directly in chat
- optional **task agents** that pick up queued background work

You can start simple with one chat session, then add multi-agent automation only when you need it.

## Fastest Setup

```bash
git clone https://github.com/mohit17mor/Arc.git
cd Arc
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
playwright install chromium
arc init
arc chat
```

When you want the dashboard, background task processing, and channels, start:

```bash
arc gateway
```

The dashboard runs at `http://localhost:18789`.

## Start Here

- **[Getting Started](getting-started.md)**: fastest path from install to first chat
- **[What Arc Is](core/what-is-arc.md)**: the mental model in plain language
- **[Main Agent vs Task Agents](core/main-agent-vs-task-agents.md)**: understand the two agent types
- **[Capabilities Overview](capabilities/overview.md)**: browser, memory, code tools, skills, MCP, and more
- **[Multi-Agent Overview](multi-agent/overview.md)**: task agents, queues, chains, and reviews
- **[Providers Overview](providers/overview.md)**: global provider setup and per-agent overrides

## What Arc Can Do

- Chat with a local or cloud model
- Run a web dashboard for chat, task management, logs, and agent management
- Queue tasks for named background agents
- Chain multiple agents in sequence with optional reviewers
- Use built-in tools like browser automation, code intelligence, memory, scheduler, and voice input
- Extend the system with custom skills or external MCP servers

## Choose Based On What You Want

### I just want a good local AI assistant

Start with [Getting Started](getting-started.md), run `arc init`, then use `arc chat`.

### I want multiple specialized agents working in the background

Go to [Multi-Agent Overview](multi-agent/overview.md).

### I want to connect more tools

Read [Skills](capabilities/skills.md) and [MCP](capabilities/mcp.md).

### I want to tune providers and models

Read [Providers Overview](providers/overview.md) and [Per-Agent Provider Config](providers/per-agent-config.md).
