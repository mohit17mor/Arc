# CLI Reference

This page is grouped by what users usually want to do, not just by command family.

## Setup And Core Use

| Command | What it does |
|---|---|
| `arc init` | First-time setup wizard for the main agent and default provider |
| `arc chat` | Start an interactive chat with the main agent |
| `arc gateway` | Run the dashboard and background systems |
| `arc listen` | Start voice input |
| `arc telegram` | Run Arc as a Telegram bot |
| `arc config` | Show current configuration |
| `arc version` | Show Arc version |

## Task Agents

| Command | What it does |
|---|---|
| `arc agent create <name>` | Create a named task agent |
| `arc agent list` | List task agents |
| `arc agent remove <name>` | Delete a task agent |

Common flags for `arc agent create`:

| Flag | Meaning |
|---|---|
| `--role`, `-r` | Role description |
| `--personality`, `-p` | Personality text |
| `--model`, `-m` | Model in `provider/model` format |
| `--max-concurrent` | Maximum parallel tasks for that agent |

## Tasks

| Command | What it does |
|---|---|
| `arc task add "<title>"` | Queue a task |
| `arc task list` | List tasks |
| `arc task show <id>` | Show task detail and comments |
| `arc task cancel <id>` | Cancel a task |
| `arc task reply <id> "<text>"` | Approve, revise, or answer a task |

Common flags for `arc task add`:

| Flag | Meaning |
|---|---|
| `--assign`, `-a` | Assign to a task agent |
| `--priority`, `-p` | Lower number means higher priority |
| `--max-bounces` | Review loop limit |
| `--after` | Make this task wait for another task |

Common flags for `arc task reply`:

| Flag | Meaning |
|---|---|
| `--action`, `-a` | `approve` or `revise` |

## Monitoring

| Command | What it does |
|---|---|
| `arc workers` | Show worker activity |
| `arc workers --follow` | Live-tail worker activity |
| `arc logs` | Show today's logs |

## Slash Commands In Chat

| Command | What it does |
|---|---|
| `/help` | Show available commands |
| `/status` | Show gateway status |
| `/skills` | List loaded skills and tools |
| `/mcp` | Show MCP server status |
| `/memory` | Show core memory |
| `/memory episodic` | Show episodic memory |
| `/memory forget <id>` | Delete a core memory fact |
| `/jobs` | List scheduled jobs |
| `/jobs cancel <name>` | Cancel a scheduled job |
| `/workflow` | List workflows |
| `/workflow <name>` | Run a workflow |
| `/cost` | Show token usage for the current session |
| `/clear` | Reset the current conversation |

## Most Common First Commands

```bash
arc init
arc chat
arc gateway
```

If you are exploring multi-agent features after that:

```bash
arc agent create researcher --role "Deep web research" --model ollama/llama3.2
arc task add "Research AI coding tools" --assign researcher
```
