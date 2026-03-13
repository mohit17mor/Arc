# Providers Overview

Providers are how Arc talks to models.

Arc supports both local and remote setups.

## Global Default Provider

The provider you choose in `arc init` becomes the default provider for the main agent and for any task agent that does not define its own override.

This global config lives in:

- `~/.arc/config.toml`

Common fields are:

- `default_provider`
- `default_model`
- `base_url`
- `api_key`

## Supported Provider Styles

Arc currently supports presets for:

- Ollama
- OpenAI
- OpenRouter
- Groq
- Together
- LM Studio
- Custom OpenAI-compatible endpoints
- Codex / Responses-style endpoints

Some providers come with a built-in default base URL. Others, such as custom endpoints and many Codex/Responses-style setups, need you to provide the base URL explicitly.

## A Good Mental Model

- Use the global provider for the **main agent** and as the default fallback.
- Override the provider per task agent when different workers need different models or endpoints.

## Typical Setups

### Simple local setup

- global provider: Ollama
- main agent uses local model
- task agents also inherit local model unless overridden

### Mixed setup

- global provider: Ollama or OpenAI
- researcher uses a cheap fast model
- writer uses a stronger writing model
- reviewer uses a strict model or a different provider

### Internal endpoint setup

- global or per-agent provider points to a custom base URL
- API key stored globally or per agent as needed

## Where To Configure Things

- `arc init`: main agent and default provider
- dashboard agent form: per-agent provider/model/base URL/API key
- `~/.arc/config.toml`: global config
- `~/.arc/agents/*.toml`: task agent overrides
