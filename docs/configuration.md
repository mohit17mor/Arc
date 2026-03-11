# Configuration

All configuration lives in `~/.arc/config.toml`, created by `arc init`.

## Full Reference

```toml
[llm]
default_provider = "ollama"
default_model = "llama3.2"
base_url = "http://localhost:11434"
api_key = ""

# Optional: separate model for background workers
worker_provider = ""
worker_model = ""
worker_base_url = ""
worker_api_key = ""

[agent]
name = "arc"
max_iterations = 25
tool_timeout = 120
temperature = 0.7

[security]
auto_allow = ["file:read"]
always_ask = ["file:write", "file:delete", "shell:exec"]
never_allow = []

[memory]
backend = "sqlite"
path = "~/.arc/memory"
enable_long_term = true
enable_episodic = true

[scheduler]
enabled = true
db_path = "~/.arc/scheduler.db"
poll_interval = 30

[identity]
path = "~/.arc/identity.md"
personality = "helpful"
user_name = ""
agent_name = "Arc"

[telegram]
token = ""
chat_id = ""
allowed_users = []

[tavily]
api_key = ""

[ngrok]
auth_token = ""

[voice]
wake_model = "hey_jarvis"
wake_threshold = 0.5
whisper_model = "base.en"
silence_duration = 1.5
listen_timeout = 30.0

[gateway]
host = "127.0.0.1"
port = 18789
```

## Environment Variables

Configuration can also be set via environment variables:

| Variable | Maps to |
|----------|---------|
| `ARC_LLM_PROVIDER` | `llm.default_provider` |
| `ARC_LLM_MODEL` | `llm.default_model` |
| `ARC_LLM_BASE_URL` | `llm.base_url` |
| `ARC_LLM_API_KEY` | `llm.api_key` |

## Precedence

Configuration is merged from multiple sources (highest to lowest):

1. Explicit overrides (passed in code)
2. Environment variables (`ARC_*`)
3. Project config (`./arc.toml`)
4. User config (`~/.arc/config.toml`)
5. Defaults (hardcoded)

## File Locations

| Path | Purpose |
|------|---------|
| `~/.arc/config.toml` | Main configuration |
| `~/.arc/identity.md` | Agent personality |
| `~/.arc/agents/*.toml` | Named agent definitions |
| `~/.arc/skills/*.py` | Custom Python skills |
| `~/.arc/skills/*.md` | Soft skills (prompt injection) |
| `~/.arc/workflows/*.yaml` | YAML workflow definitions |
| `~/.arc/mcp.json` | MCP server configuration |
| `~/.arc/memory/` | Memory databases |
| `~/.arc/tasks.db` | Task queue database |
| `~/.arc/scheduler.db` | Scheduler database |
| `~/.arc/logs/` | Log files |
| `~/.arc/notifications.log` | Notification history |
