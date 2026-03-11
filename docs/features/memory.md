# Three-Tier Memory

Arc remembers you across sessions using three tiers of local SQLite storage.

## Tiers

### Tier 1 — Session Memory (RAM)
Current conversation messages. Token-budget managed — older messages are truncated to fit the context window.

### Tier 2 — Episodic Memory (sqlite-vec)
Past conversation chunks stored with vector embeddings. Retrieved by semantic similarity when relevant to the current query. Uses FastEmbed (BAAI/bge-small-en-v1.5, 384-dim, fully local).

### Tier 3 — Core Memory (SQLite)
High-confidence facts about you — name, preferences, projects. Always injected into the system prompt. Distilled automatically from conversations by an LLM background task.

## How It Flows

1. **Context composition** — before each LLM call, the ContextComposer assembles:
   - System prompt + core facts (Tier 3, always present)
   - Relevant episodic memories (Tier 2, retrieved by similarity to current query)
   - Recent session messages (Tier 1, as many as fit in the token budget)

2. **After each turn** — background tasks:
   - Store the turn as episodic memory (Tier 2)
   - Every 5 turns, distill stable facts to core memory (Tier 3)

## Chat Commands

```
/memory              # show core facts
/memory episodic     # show recent episodic memories
/memory forget <id>  # delete a core fact
```

## Configuration

```toml
[memory]
backend = "sqlite"
path = "~/.arc/memory"
enable_long_term = true
enable_episodic = true
```
