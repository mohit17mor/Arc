# Getting Started

This page is the shortest path to a working Arc setup.

You do **not** need to create task agents to get started. The main agent works immediately after `arc init`.

## 1. Install Arc

Use the one-command installer. It attempts the full Arc stack on the first pass, including browser support when possible.

=== "macOS / Linux"

    ```bash
    curl -fsSL https://raw.githubusercontent.com/mohit17mor/Arc/main/scripts/install/install.sh | sh
    ```

=== "Windows PowerShell"

    ```powershell
    irm https://raw.githubusercontent.com/mohit17mor/Arc/main/scripts/install/install.ps1 | iex
    ```

The installer may prompt if it needs permission to install required system dependencies such as Python.

## 2. Run The Setup Wizard

```bash
arc init
```

The wizard sets up your main agent identity and your default provider.

It walks through:

1. your name and the agent name
2. the main agent personality or custom system prompt
3. the default LLM provider and model
4. optional extras like Telegram, Tavily, and ngrok

Arc stores this in `~/.arc/config.toml` and `~/.arc/identity.md`.

If the installer reports optional failures, run:

```bash
arc doctor
```

This shows whether the Arc runtime is healthy and whether browser support is ready.

## 3. Start Your First Chat

```bash
arc chat
```

This is the easiest way to understand Arc. At this point you already have:

- the main agent
- your default provider
- built-in capabilities
- any local skills and MCP config you have added

## 4. Run The Gateway When You Need More Than Chat

```bash
arc gateway
```

Use `arc gateway` when you want:

- the web dashboard
- task processing for named task agents
- Telegram or other connected channels
- real-time logs and background services

Open `http://localhost:18789` in your browser.

## 5. Optional Next Steps

### Add named task agents

If you want background workers for specialized jobs, go to [Create Agents](multi-agent/create-agents.md).

### Queue tasks and chains

If you want work to continue without staying in chat, go to [Queue Tasks](multi-agent/tasks.md) and [Multi-Step Chains and Reviews](multi-agent/chains-and-reviews.md).

### Add more tools

If you want to extend Arc, go to [Skills](capabilities/skills.md) and [MCP](capabilities/mcp.md).

## Developer Install

If you are contributing to Arc itself, use the repo-based setup instead:

```bash
git clone https://github.com/mohit17mor/Arc.git
cd Arc
python -m venv .venv
```

=== "Windows"

    ```bash
    .venv\Scripts\activate
    ```

=== "macOS / Linux"

    ```bash
    source .venv/bin/activate
    ```

```bash
pip install -e ".[dev]"
playwright install chromium
```

## First-Day Mental Model

- `arc init` configures the **main agent** and the default LLM settings.
- `arc chat` talks to the **main agent**.
- `arc gateway` turns on the dashboard and background systems.
- **Task agents** are optional. You only create them when you want queued multi-agent work.
