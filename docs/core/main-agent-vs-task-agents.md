# Main Agent vs Task Agents

This is the most important distinction in Arc.

## Main Agent

The main agent is the assistant you talk to directly.

It is used by:

- `arc chat`
- the WebChat inside the dashboard
- connected channels like Telegram

The main agent gets its identity from the files created by `arc init`.

That includes:

- your selected personality preset or custom system prompt
- your chosen default provider and model
- global configuration like memory, tools, and channels

## Task Agents

Task agents are optional named workers stored as files in `~/.arc/agents/*.toml`.

Each one can have its own:

- name
- role
- personality
- full system prompt
- provider and model override
- base URL and API key override
- skill allowlist or denylist

They only matter when you queue background tasks.

## What Gets Applied Where

### When you use a preset or custom prompt in `arc init`

That affects the **main agent only**.

It does **not** automatically rewrite the prompts for named task agents.

### When you create a task agent

That agent uses its own configuration from the agent TOML file or the dashboard agent form.

If a task agent does not define its own LLM override, it falls back to the global default provider from `~/.arc/config.toml`.

## Practical Example

### Main agent

You might configure the main agent to be your general-purpose assistant with a strong custom system prompt.

### Task agents

You might also create:

- `researcher` on a cheap fast model
- `writer` on a better writing model
- `reviewer` on a stricter review prompt

That split is normal in Arc.

## Rule Of Thumb

- Change `arc init` when you want to change how **Arc talks to you**.
- Create or edit task agents when you want to change how **background workers do jobs**.
