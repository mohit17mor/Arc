# Taskforce Overview

The Taskforce is Arc's system for autonomous agent teams. Instead of one agent doing everything, you create **specialized agents** with their own roles, LLMs, and system prompts. Tasks are queued, agents pick them up, and results are delivered — even while you sleep.

## How It Works

```
┌─────────────────────────────────────────────────┐
│                 Task Queue (SQLite)              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │Task 1│ │Task 2│ │Task 3│ │Task 4│ │Task 5│  │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘  │
└─────┼────────┼────────┼────────┼────────┼───────┘
      │        │        │        │        │
      ▼        ▼        ▼        ▼        ▼
 researcher  researcher  writer   writer  reviewer
 (ollama)    (ollama)    (gpt-4o) (gpt-4o) (ollama)
```

## Key Concepts

### Named Agents
Each agent is a TOML file in `~/.arc/agents/`. It defines the agent's identity, LLM, skills, and behavior. See [Agents](agents.md).

### Persistent Task Queue
Tasks are stored in SQLite and survive across restarts. Queue 10 tasks at night, close your laptop, results arrive in the morning. See [Task Queue](tasks.md).

### Multi-Step Workflows
Chain agents together with optional review steps. The output of one agent feeds into the next. See [Workflows](workflows.md).

### Task Processor
A background daemon (part of `arc gateway`) that polls the queue every 5 seconds, dispatches tasks to the right agent, handles review loops, and delivers results via notifications.

## Quick Example

```bash
# 1. Create agents
arc agent create researcher --role "AI analyst" --model ollama/llama3.2
arc agent create writer --role "Content writer" --model openai/gpt-4o

# 2. Queue tasks
arc task add "Research quantum computing trends" --assign researcher
arc task add "Write a blog post about AI" --assign writer

# 3. Start the daemon
arc gateway

# 4. Monitor
arc task list
arc task show t-a1b2c3d4
```
