# Create Agents

Task agents are named background workers.

Each agent can have a different role, prompt, and model setup.

## Ways To Create An Agent

## Dashboard

Open the dashboard and go to the **Agents** tab.

The agent form supports:

- name
- role
- personality
- full system prompt
- provider
- model
- base URL
- API key
- max concurrency

That means users do **not** need to edit config files just to give a task agent its own endpoint or API key.

## CLI

```bash
arc agent create researcher \
  --role "Deep web research" \
  --personality "thorough, cites sources" \
  --model openrouter/anthropic/claude-sonnet-4-20250514
```

The CLI is convenient for quick creation, but the dashboard gives fuller visibility for agent setup.

## Agent File

Agents are stored in:

- `~/.arc/agents/<name>.toml`

Example:

```toml
name = "researcher"
role = "AI industry analyst"
personality = "thorough, evidence-driven"
max_concurrent = 1

[llm]
provider = "codex"
model = "codex-mini-latest"
base_url = "https://example.internal/v1"
api_key = "sk-secret"

system_prompt = """
You are a research agent.
Always cite sources and separate findings from assumptions.
"""
```

## How Provider Inheritance Works

- If an agent has its own `[llm]` block, it uses that.
- If it does not, it falls back to the global default provider from `~/.arc/config.toml`.

## When To Give An Agent Its Own Full Prompt

Use `system_prompt` when you want tight control.

Use `role` and `personality` when a lighter setup is enough.

If `system_prompt` is present, it overrides the auto-generated prompt for that agent.
