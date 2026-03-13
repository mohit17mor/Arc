# Configuration

Arc keeps most persistent user configuration under `~/.arc/`.

This page is the reference for where things live and how config is resolved.

## The Most Important Files

| Path | Purpose |
|---|---|
| `~/.arc/config.toml` | Main config, including the default provider |
| `~/.arc/identity.md` | Main agent identity and prompt material |
| `~/.arc/agents/*.toml` | Named task agent definitions |
| `~/.arc/skills/*.py` | Custom Python skills |
| `~/.arc/skills/*.md` | Soft skills |
| `~/.arc/mcp.json` | MCP server config |
| `~/.arc/workflows/*.yaml` | YAML workflows |
| `~/.arc/tasks.db` | Task queue database |
| `~/.arc/scheduler.db` | Scheduler database |
| `~/.arc/logs/` | Log files |

## Global LLM Config

The default provider configured by `arc init` is stored in `~/.arc/config.toml`.

Example:

```toml
[llm]
default_provider = "ollama"
default_model = "llama3.2"
base_url = "http://localhost:11434"
api_key = ""

worker_provider = ""
worker_model = ""
worker_base_url = ""
worker_api_key = ""
```

## Identity And Personality

The main agent identity is stored separately from named task agents.

Relevant config includes:

```toml
[identity]
path = "~/.arc/identity.md"
personality = "helpful"
user_name = ""
agent_name = "Arc"
```

## Environment Variables

Common environment overrides:

| Variable | Maps to |
|---|---|
| `ARC_LLM_PROVIDER` | `llm.default_provider` |
| `ARC_LLM_MODEL` | `llm.default_model` |
| `ARC_LLM_BASE_URL` | `llm.base_url` |
| `ARC_LLM_API_KEY` | `llm.api_key` |

## Config Precedence

Highest to lowest:

1. explicit runtime overrides
2. environment variables
3. project config such as `./arc.toml` if used
4. user config in `~/.arc/config.toml`
5. built-in defaults

## Other Config Sections

Depending on what you enable, `config.toml` may also contain:

- `[memory]`
- `[scheduler]`
- `[telegram]`
- `[tavily]`
- `[ngrok]`
- `[voice]`
- `[gateway]`

Use `arc init` for the common first-time setup, then edit the file directly when you need precise control.
