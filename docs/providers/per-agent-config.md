# Per-Agent Provider Config

Per-agent provider config is what lets Arc run different task agents on different models.

## What You Can Override Per Agent

A task agent can override:

- provider
- model
- base URL
- API key

If those are not set, the agent uses the global defaults.

## Dashboard Support

The dashboard agent creation form already supports the full provider block.

Users can set:

- LLM provider
- model
- base URL
- API key

So for task agents, editing files is optional, not required.

## Example Agent Config

```toml
name = "writer"
role = "Turn research into polished writing"

[llm]
provider = "openai"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
```

## Example: Internal Codex-Compatible Endpoint

```toml
name = "coder"
role = "Implementation agent"

[llm]
provider = "codex"
model = "codex-mini-latest"
base_url = "https://internal.example.com/20250206/app/litellm"
api_key = "sk-..."
```

## When To Override

Override per agent when:

- one role needs a different model quality level
- one role should use a cheaper or faster endpoint
- one role must point at an internal gateway
- one role needs different credentials

## When Not To Override

Do not override unless you need to. Inheriting the global provider keeps setup simpler.
