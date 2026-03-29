<h1 align="center">Arc</h1>
<h3 align="center">An AI workspace with a main agent for chat and optional task agents for background work.</h3>

<p align="center">
  <img src="public/assets/arc.png" width="320" alt="Arc" />
</p>

## Demo

**Watch Arc perform multiple tasks in parallel while recording itself**

[Watch the demo video](https://github.com/user-attachments/assets/43c69201-c885-40d4-8928-77ba39438685)

This demo shows Arc coordinating multiple agents at once. It is:

1. recording the session
2. searching for flights
3. checking the weather
4. finding tourist spots
5. scheduling a water reminder
6. working on a simple coding task

While those tasks are running in the background, the main agent remains free to keep talking to you.

Arc gives you two layers in one system:

- a **main agent** you talk to directly in chat
- optional **task agents** that pick up queued work in the background

That means you can start simple with `arc chat`, then add multi-agent chains, reviewers, dashboard workflows, and custom tools only when you actually need them.

## Quick Start

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/mohit17mor/Arc/main/scripts/install/install.sh | sh
```

```powershell
# Windows PowerShell
irm https://raw.githubusercontent.com/mohit17mor/Arc/main/scripts/install/install.ps1 | iex
```

```bash
arc init
arc chat
```

To run the dashboard and background services:

```bash
arc gateway
```

The dashboard runs at `http://localhost:18789`.

If installation succeeds but something optional failed, run:

```bash
arc doctor
```

## Developer Setup

If you want to work on Arc itself:

```bash
git clone https://github.com/mohit17mor/Arc.git
cd Arc
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
playwright install chromium
```

## What Arc Covers

### Main Agent

Use the main agent for direct chat, planning, research, coding help, and everyday assistant workflows.

### Multi-Agent Background Work

Create named task agents when you want:

- specialized roles such as `researcher`, `writer`, and `reviewer`
- queued tasks that survive restarts
- multi-step chains
- AI or human review loops
- different models per worker

### Built-In Capabilities

Arc includes:

- browser automation
- web dashboard
- memory
- code intelligence
- scheduler
- voice input
- skills
- MCP support

## Mental Model

- `arc init` configures the **main agent** and your default provider.
- `arc chat` talks to the **main agent**.
- `arc gateway` runs the dashboard and background systems.
- **Task agents** are optional named workers you create only for queued jobs.

## Documentation

Full docs: https://mohit17mor.github.io/Arc

Recommended path:

- Getting Started
- Core Concepts
- Capabilities
- Multi-Agent
- Providers

## Development

```bash
pytest
```

## License

MIT
