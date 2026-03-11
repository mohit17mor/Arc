# Architecture

## Micro-Kernel Design

```
                         ┌──────────────┐
                         │  CLI / Chat  │
                         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │      Agent Loop       │
                    │  think → act → observe│
                    └──┬─────┬─────┬─────┬──┘
                       │     │     │     │
                 ┌─────▼┐ ┌──▼──┐ ▼   ┌─▼────────┐
                 │ LLM  │ │ Mem │ │   │ Security  │
                 └──────┘ └─────┘ │   │  Engine   │
                              ┌───▼───┴───────────┘
                              │   Skill Router
                              │  (two-tier selection)
                              ├────────────────────┐
                              │ Tier 1: Always On  │
                              │ files · terminal   │
                              │ worker · task_board│
                              ├────────────────────┤
                              │ Tier 2: On Demand  │
                              │ browser · scheduler│
                              │ code_intel · MCP   │
                              └────────────────────┘

    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │   CLI    │  │ WebChat  │  │ Telegram │  │  Voice   │
    │ arc chat │  │ Gateway  │  │  arc tg  │  │arc listen│
    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
         └──────────────┴─────────────┴──────────────┘
                       All share the same agent
```

Every subsystem is independent, wired through an event bus. The kernel composes:

- **EventBus** — pub/sub with wildcard matching + middleware chain
- **Registry** — typed service locator (DI by category + name)
- **Config** — Pydantic models with 5-level precedence

## Two-Tier Skill Router

Sending every tool spec to the LLM on every call wastes tokens (~2,500). The router keeps it lean (~800):

**Tier 1** — always sent: filesystem, terminal, worker, task_board

**Tier 2** — activated on demand via `use_skill("browsing")`: browser, scheduler, code_intel, liquid_web, MCP, workflows

New skills auto-appear in the `use_skill` menu.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Kernel | `arc/core/kernel.py` | Central coordinator (~130 lines) |
| EventBus | `arc/core/bus.py` | Pub/sub + middleware |
| AgentLoop | `arc/agent/loop.py` | think → act → observe cycle |
| SkillManager | `arc/skills/manager.py` | Skill registration + tool routing |
| SkillRouter | `arc/skills/router.py` | Two-tier tool selection |
| MemoryManager | `arc/memory/manager.py` | Three-tier memory orchestrator |
| SecurityEngine | `arc/security/engine.py` | Capability checking + approval |
| TaskProcessor | `arc/tasks/processor.py` | Background task queue processing |
| TaskStore | `arc/tasks/store.py` | SQLite persistence for tasks |
| BrowserEngine | `arc/browser/engine.py` | Playwright browser management |
| LLMProvider | `arc/llm/base.py` | Abstract LLM interface |
| GatewayServer | `arc/gateway/server.py` | WebSocket + REST API + dashboard |

## LLM Providers

All LLM access goes through the `LLMProvider` interface. Implementations:

- **OllamaProvider** — local models
- **OpenAICompatProvider** — OpenAI, Groq, OpenRouter, Together, DeepSeek, LM Studio, etc.
- **ResponsesProvider** — OpenAI Responses API
- **MockProvider** — for testing

Swap providers by changing one config line. Agent code never calls LLM APIs directly.
