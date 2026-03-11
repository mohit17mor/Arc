# CLI Reference

## Core Commands

| Command | Description |
|---------|-------------|
| `arc init` | First-time setup wizard |
| `arc chat` | Interactive chat session |
| `arc chat -m <model>` | Chat with a specific model |
| `arc gateway` | Run daemon (dashboard + tasks + Telegram) |
| `arc listen` | Voice input (requires `arc gateway`) |
| `arc telegram` | Run as a Telegram bot |
| `arc config` | Show current configuration |
| `arc version` | Show Arc version |

## Agent Commands

| Command | Description |
|---------|-------------|
| `arc agent create <name>` | Create a named agent |
| `arc agent list` | List all agents |
| `arc agent remove <name>` | Delete an agent |

Options for `arc agent create`:

| Flag | Description |
|------|-------------|
| `--role`, `-r` | Agent's role description |
| `--personality`, `-p` | Personality traits |
| `--model`, `-m` | LLM model (`provider/model` format) |
| `--max-concurrent` | Max parallel tasks (default: 1) |

## Task Commands

| Command | Description |
|---------|-------------|
| `arc task add "<title>"` | Queue a task |
| `arc task list` | List all tasks |
| `arc task show <id>` | Full detail + comments |
| `arc task cancel <id>` | Cancel a task |
| `arc task reply <id> "<text>"` | Answer a blocked task |

Options for `arc task add`:

| Flag | Description |
|------|-------------|
| `--assign`, `-a` | Agent name to assign to |
| `--priority`, `-p` | Priority (1=highest, default: 1) |
| `--max-bounces` | Max review iterations (default: 3) |
| `--after` | Task ID dependency |

Options for `arc task reply`:

| Flag | Description |
|------|-------------|
| `--action`, `-a` | `approve` or `revise` (default: approve) |

## Monitoring Commands

| Command | Description |
|---------|-------------|
| `arc workers` | Show recent worker activity |
| `arc workers --follow` | Live-tail worker activity |
| `arc logs` | View today's logs |

## Chat Slash Commands

| Command | Description |
|---------|-------------|
| `/skills` | List loaded skills and tools |
| `/mcp` | MCP server status |
| `/memory` | Core facts (long-term memory) |
| `/memory episodic` | Recent episodic memories |
| `/memory forget <id>` | Delete a core fact |
| `/jobs` | List scheduled jobs |
| `/jobs cancel <name>` | Cancel a scheduled job |
| `/workflow` | List available workflows |
| `/workflow <name>` | Run a workflow |
| `/cost` | Token usage this session |
| `/clear` | Reset conversation |
| `/status` | Gateway connection status |
| `/help` | Show available commands |
