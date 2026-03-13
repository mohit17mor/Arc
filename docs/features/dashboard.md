# Web Dashboard

The dashboard is the browser UI served by `arc gateway` at `http://localhost:18789`.

Use it when you want a visual way to work with Arc instead of staying in the terminal.

## What The Dashboard Is Good For

- chatting with the main agent in the browser
- monitoring queued and running tasks
- creating and managing task agents
- creating single-step or multi-step tasks
- reviewing work and replying to blocked tasks
- checking loaded skills, MCP status, and system logs

## Main Areas

### Dashboard

System overview: task counts, agent counts, uptime, and high-level status.

### Chat

Browser-based chat with the main agent. This is the same main-agent layer as `arc chat`, just in a web UI.

### Task Board

Kanban-style task view for:

- queued work
- active work
- review states
- completed work

The task form supports both:

- simple single-agent tasks
- multi-step chains with one agent per step and an optional reviewer per step

### Agents

Create and manage named task agents.

The form supports the full per-agent model block, including:

- provider
- model
- base URL
- API key
- system prompt

So if a task agent needs its own endpoint, users can configure it from the dashboard.

### Scheduler

View scheduled jobs and cancel them.

### Skills And MCP

Inspect the currently loaded capabilities and MCP server status.

### Logs

Watch real-time activity from the main agent, task agents, workers, and other background systems.

## Architecture Notes

The dashboard is a lightweight single-page app served directly by Arc.

- no separate frontend build is required
- one WebSocket connection powers live updates
- REST endpoints provide overview, tasks, agents, scheduler, skills, MCP, and logs
