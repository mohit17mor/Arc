# Agents

Agents are specialized autonomous workers defined as TOML files in `~/.arc/agents/`.

## Creating an Agent

### Via CLI

```bash
arc agent create researcher \
  --role "Deep web research and analysis" \
  --model ollama/llama3.2 \
  --personality "thorough, cites sources"
```

### Via Dashboard

Open [http://localhost:18789](http://localhost:18789), go to the **Agents** tab, click **+ New Agent**.

### Manual TOML

Create `~/.arc/agents/researcher.toml`:

```toml
name = "researcher"
role = "AI industry analyst"
personality = "thorough, cites sources, structured output"
max_concurrent = 1

[llm]
provider = "ollama"
model = "llama3.2"

system_prompt = """
You are a senior research analyst specializing in technology and AI.

## Research Protocol
1. Search at least 3 different sources for every claim
2. Cross-reference findings between sources
3. Always cite URLs for every fact

## Output Format
- Start with a 2-sentence executive summary
- Key findings as bullet points
- End with a "Sources" section with all URLs used
"""
```

## Configuration Reference

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier (alphanumeric + underscores) |
| `role` | Yes | Short description of the agent's purpose |
| `personality` | No | Personality traits that shape behavior |
| `system_prompt` | No | Full system prompt (overrides auto-generated one) |
| `max_concurrent` | No | Max parallel tasks (default: 1) |
| `skills` | No | Whitelist of allowed skills (omit for all) |
| `exclude_skills` | No | Blacklist of skills to exclude |
| `[llm].provider` | No | LLM provider name |
| `[llm].model` | No | Model name |
| `[llm].base_url` | No | Custom API endpoint |
| `[llm].api_key` | No | API key (falls back to global config) |

## Per-Agent LLM Selection

Each agent can use a different LLM. Use cheap models for simple tasks, powerful models for creative work.

| Provider | `--model` format | Example |
|----------|-----------------|---------|
| Ollama (local) | `ollama/<model>` | `ollama/llama3.2` |
| OpenAI | `openai/<model>` | `openai/gpt-4o` |
| OpenRouter | `openrouter/<model>` | `openrouter/anthropic/claude-sonnet-4-20250514` |
| Groq | `groq/<model>` | `groq/llama-3.3-70b-versatile` |
| Together | `together/<model>` | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| LM Studio | `lmstudio/<model>` | `lmstudio/default` |
| Custom | Edit TOML directly | Set `base_url` in `[llm]` section |

If no LLM is specified, the agent uses the default from `~/.arc/config.toml`.

## System Prompt

The `system_prompt` field is the full prompt sent to the LLM. Use TOML triple-quoted strings for multi-line prompts.

If omitted, an auto-generated prompt is created from `role` + `personality`.

!!! tip
    A good system prompt includes: research methodology, output format expectations, quality standards, and specific instructions for the domain.

## Managing Agents

```bash
arc agent list            # show all agents
arc agent remove <name>   # delete an agent
```

Or use the dashboard at the **Agents** tab.
