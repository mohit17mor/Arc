# What Arc Is

Arc is an AI system that can work in two layers.

## Layer 1: Your Main Agent

This is the assistant you talk to directly in `arc chat`, WebChat, or connected channels.

Use it for:

- normal conversation
- ad hoc research or coding help
- planning and execution in a single chat
- creating tasks for background agents when needed

If all you want is a strong AI assistant with tools, this layer may be enough.

## Layer 2: Optional Task Agents

These are named background workers such as `researcher`, `writer`, or `reviewer`.

Use them when you want:

- specialized roles
- different models for different jobs
- persistent queued tasks
- multi-step chains like A -> B -> C
- review loops without staying in chat

## Why Arc Feels Different

Most tools stop at one conversation.

Arc can also:

- remember global configuration and identity
- queue work in a persistent task board
- move work across multiple agents
- expose capabilities through a dashboard
- grow with local skills and MCP servers

## The Simplest Way To Think About It

- The **main agent** is your front door.
- **Task agents** are your background workers.
- The **gateway** runs the dashboard and background processing.
- **Skills** and **MCP** are how Arc gets more tools.

## When To Stay Simple

Use only the main agent if:

- you mostly chat interactively
- you do not need background queues
- one model and one identity are enough

## When To Add Multi-Agent

Add task agents if:

- you want different roles with different prompts
- you want work to continue after you leave chat
- you want reviewer loops or chains
- you want different providers or models per worker
