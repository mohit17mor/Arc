# Arc

**Your AI agent that actually does things.**

Arc is a Python framework for building a personal AI agent that runs on your machine. It browses the web, fills forms, manages files, runs commands, and remembers you across sessions — all locally, all free.

```
You:  "Find me a one-way flight from Delhi to Mumbai on April 10"
Arc:  *opens Google Flights, fills the form, picks dates from the calendar,
       selects suggestions, clicks search — hands you the results*
```

---

## Why Arc?

| | Cloud agents | Arc |
|---|---|---|
| **Cost** | Per-token billing | Free (local LLM via Ollama) |
| **Privacy** | Your data on someone's server | Everything stays on your machine |
| **Browser** | Screenshot → vision (slow, expensive) | Accessibility tree → text (fast, free) |
| **Memory** | Forgets you every session | Three-tier memory that persists forever |
| **Extensibility** | Closed | Drop a `.py` or `.md` file → new skill |

---

## Quick Start

```bash
git clone https://github.com/ArcAI-xyz/Arc.git && cd Arc
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
playwright install chromium                       # for browser control
ollama pull llama3.2                              # or any model you prefer
arc init                                          # first-time setup
arc chat                                          # start talking
```

---

## What It Can Do

### 🌐 Browse the Web — For Real
Not just fetching URLs. Arc opens a **real Chromium browser** and interacts with pages like a human:

- Fills forms (text, dropdowns, comboboxes, date pickers)
- Handles autocomplete suggestions (Google Flights, Amazon, etc.)
- Navigates calendars, picks dates, closes overlays
- Deals with CAPTCHAs by asking you to solve them, then continues
- Works across Google Flights, Amazon, Wikipedia, and most sites

Three browser tools:
- **`browser_go`** — navigate to a URL, get a structured page snapshot
- **`browser_look`** — re-examine the current page
- **`browser_act`** — click, fill, scroll, submit — all in one call

### 🧠 Remember You
Three tiers of memory, all local SQLite:

- **Core facts** — your name, preferences, projects. Always in the system prompt.
- **Episodic** — past conversation chunks, retrieved by semantic similarity.
- **Session** — current conversation, token-budget managed.

### 🔧 Use Tools
Built-in skills: file read/write, terminal commands, web search, web scraping, browser control.

### 👷 Delegate Work
Spawn background worker agents for long tasks. Keep chatting while research happens in the background.

### ⏰ Schedule Jobs
"Remind me every morning at 9am to check my email" — cron, interval, or one-shot triggers.

### 🔔 Get Notified
Results delivered in-chat, to a log file, or via Telegram.

### 🎤 Voice Input — Talk Hands-Free
Say a wake word and Arc listens. No terminal or browser tab needs to be focused.

```
You:  "Hey Jarvis"
🔔    *chime*
You:  "Search for flights from Delhi to Mumbai next week"
Arc:  *runs the query, sends you a desktop notification with results*
You:  "Sort by cheapest"    ← no wake word needed, conversation is still active
Arc:  *responds*
      ... 30 seconds of silence ...
Arc:  *goes back to sleep*
```

**How it works:**
- **Wake word** — `openwakeword` detects "Hey Jarvis" (configurable) at ~2% CPU idle
- **Speech-to-text** — `faster-whisper` (offline, local, ~150MB model) transcribes after wake word
- **Gateway client** — sends transcribed text to `arc gateway` via WebSocket (same protocol as WebChat)
- **Conversation flow** — 4-state machine: SLEEPING → ACTIVE → PROCESSING (mic off) → LISTENING (30s follow-up window)
- **Cross-platform** — conversations appear live in WebChat and Telegram
- **Mic goes deaf** during processing — your meetings, music, and background chatter are never captured

**Setup:**

```bash
pip install arc-agent[voice]     # or: pip install sounddevice faster-whisper openwakeword

# Terminal 1:
arc gateway

# Terminal 2:
arc listen
```

**Configuration** (`~/.arc/config.toml`):

```toml
[voice]
wake_model = "hey_jarvis"     # also: alexa, hey_mycroft, hey_rhasspy
whisper_model = "base.en"     # tiny.en (faster) or small.en (more accurate)
silence_duration = 1.5        # seconds of silence = end of speech
listen_timeout = 30.0         # seconds before going back to sleep
```

### 📋 Workflows — Deterministic Step-by-Step Automation
Define repeatable multi-step tasks in YAML. Unlike ad-hoc tool calls, workflows execute in a **fixed order** with retry, failure handling, human-in-the-loop pauses, and progress events.

Drop `.yaml` files in `~/.arc/workflows/` — they're discovered automatically on next activation.

**Quick example:**
```yaml
# ~/.arc/workflows/jira-rca.yaml
name: jira-rca
description: Root cause analysis for a Jira ticket
trigger: "rca|root cause"
steps:
  - do: Fetch the Jira ticket details
    tool: mcp_call
    args: {server: jira, tool: get_issue, arguments: {key: "${ticket}"}}
  - do: Search the codebase for the relevant error
    retry: 2
  - do: Analyze the root cause and write a summary
    on_fail: continue
  - do: Post the RCA summary as a comment on the ticket
```

**Step formats:**

```yaml
# Simple — just plain English
steps:
  - search the web for NVIDIA news
  - summarize the results

# Extended — with control options
steps:
  - do: Search for relevant data
    retry: 2                    # retry up to 2 times on failure
    on_fail: continue           # continue|stop (default: stop)
    ask_if_unclear: true        # agent asks user instead of guessing (default: true)
    wait_for_input: true        # pause workflow and wait for user response

# Explicit — bypass agent, call tool directly
steps:
  - do: Get the Jira ticket
    tool: mcp_call
    args: {server: jira, tool: get_issue, arguments: {key: "PROJ-123"}}

# Shell command
steps:
  - do: Check running pods
    shell: kubectl get pods -n payments
```

**Human-in-the-loop — workflows that wait for you:**

Workflows can pause and wait for user input in two ways:

1. **Explicit** — mark a step with `wait_for_input: true`:
```yaml
steps:
  - do: Ask the user which environment to deploy to
    wait_for_input: true
  - do: Deploy to the selected environment
```

2. **Implicit** — the agent decides on its own. If a step has `ask_if_unclear: true` (the default) and the agent's response is a question ("Which environment should I deploy to?"), the workflow **automatically pauses** and waits for your answer. No YAML change needed — the agent asks naturally and the workflow holds until you respond.

In both cases:
- The workflow waits **indefinitely** — no timeout. Take hours if you need to.
- Your answer is injected as context for the next step.
- Works across all platforms — answer from CLI, WebChat, or voice.

**Features:**
- **Context passing** — each step sees results from previous steps
- **Progress events** — `workflow:start → step_start → step_complete → waiting_input → complete` via the kernel event bus
- **10-minute timeout** per step to catch stuck calls (does not apply to user input waits)

**Usage:**
```
/workflow list              # see available workflows
/workflow jira-rca          # run a workflow
```
Or the agent can trigger them: *"run the jira-rca workflow for PROJ-1234"*

### 🛍️ Liquid Web — Product Search & Comparison
Ask Arc to find products and it renders a **live 3D carousel** you can browse in your browser:

```
You:  "find me the best wireless earbuds under ₹5000"
Arc:  *searches Tavily → scrapes Amazon, Flipkart, etc. in parallel
       → extracts product data → renders a 3D carousel UI
       → opens it in your browser*
      "Found 12 products from 4 sites. Comparison page is live at: http://localhost:63350"
```

**How it works:**

1. **Search** — queries [Tavily API](https://tavily.com) (free tier: 1,000 searches/month) to find relevant product pages
2. **Scrape** — launches parallel headless Chromium contexts via `BrowserPool` to extract product data
3. **Extract** — pulls structured data using JSON-LD, OpenGraph, and DOM heuristics, plus site-specific extractors for Amazon and Flipkart
4. **Filter** — scores products on quality (price, name, image, rating) and drops blog/review page noise
5. **Deduplicate** — removes near-duplicate products by name similarity
6. **Render** — generates a responsive 3D carousel with convex arc layout, neon gradient borders, and glassmorphism effects
7. **Serve** — starts a local HTTP server (optionally tunneled via ngrok for mobile/Telegram access)

**Server lifecycle:**
- The server runs **independently** in the background — you can keep chatting with Arc
- Closing the browser tab doesn't kill the server — revisit the URL anytime
- Auto-shuts down after **10 minutes** of inactivity
- A new search **automatically replaces** the old one (only one result page is live at a time)
- Everything cleans up when you exit `arc chat`

**Setup:**

```bash
arc init   # enable Liquid Web when prompted, paste your Tavily API key
```

Or add manually to `~/.arc/config.toml`:
```toml
[tavily]
api_key = "tvly-..."
```

Optionally, for public URLs accessible from mobile or Telegram:
```toml
[ngrok]
auth_token = "..."
```

---

## Architecture

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
                 │Ollama│ │3-tier│ │   │  Engine   │
                 └──────┘ └─────┘ │   └───────────┘
                              ┌───▼──────────────────────┐
                              │      Skill Router        │
                              │   (two-tier selection)   │
                              ├──────────────────────────┤
                              │ Tier 1 — Always Active   │
                              │ files · terminal · worker│
                              ├──────────────────────────┤
                              │ Tier 2 — On Demand       │
                              │ browser · scheduler ·    │
                              │ liquid web · MCP gateway │
                              │ workflows                │
                              └──────────────────────────┘

    ┌───────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐
    │    CLI    │  │  WebChat │  │  Telegram  │  │   Voice    │
    │ arc chat  │  │ Gateway  │  │  arc tg    │  │ arc listen │
    └─────┬─────┘  └────┬─────┘  └─────┬──────┘  └─────┬──────┘
          └──────────────┴─────────────┴────────────────┘
                          All share the same agent
```

**Micro-kernel design** — every subsystem is independent, wired through an event bus. Swap Ollama for Claude or GPT by changing one config line.

### Two-Tier Skill Router

Real agents have many skills, but sending every tool spec to the LLM on every call wastes tokens and confuses the model. Arc uses a **two-tier routing system** that keeps the LLM context lean:

**Tier 1 — Always Active**
Core tools the agent needs constantly:
- **filesystem** — read/write files
- **terminal** — run shell commands
- **worker** — delegate tasks to background agents

These tools are sent with every LLM call.

**Tier 2 — On Demand**
Specialized tools activated only when needed:
- **browsing** — browser_go, browser_act, browser_look
- **scheduler** — create/manage scheduled jobs
- **liquid_web** — product search and comparison
- **MCP servers** — any external tool servers

The LLM activates these by calling `use_skill("browsing")`, which injects that skill's tools into the next call. Skills reset at the start of each new user turn.

**Why this matters:**
- Without routing: ~2,500 tokens of tool specs on every call
- With routing: ~800 tokens (Tier 1 + compact `use_skill` menu)
- The LLM naturally calls `use_skill` when it recognizes it needs browsing, scheduling, etc.
- Zero manual wiring — new skills auto-appear in the `use_skill` menu

---

## Browser Control — Under the Hood

The browser skill uses **accessibility-tree snapshots**, not screenshots. Each page is converted to a numbered list of interactive elements:

```
[3] textbox "Where from?" value="Delhi"
[5] combobox "Where to?"
[7] textbox "Departure" value=""
[9] button "Search"
```

The LLM sees this, decides what to do, and sends actions:
```json
{"actions": [
  {"type": "fill", "target": "[5]", "value": "Mumbai"},
  {"type": "fill", "target": "[7]", "value": "2026-04-10"},
  {"type": "click", "target": "[9]"}
]}
```

The engine handles the hard parts mechanically:
- **Autocomplete**: types char-by-char, waits for dropdown, picks best match using word-boundary-aware scoring
- **Calendars**: detects calendar type (data-iso, aria-label, gridcell), navigates months, clicks the right day
- **Overlays**: escalating click fallbacks (normal → force → JS → mouse coordinates)
- **CAPTCHAs**: detected and escalated to human, then continues where it left off

---

## Configuration

`~/.arc/config.toml` (created by `arc init`):

```toml
[llm]
default_model = "llama3.2"
base_url = "http://localhost:11434"

[agent]
max_iterations = 25
temperature = 0.7

[security]
default_policy = "ask"

[scheduler]
enabled = true

[tavily]
api_key = ""    # Liquid Web — get a free key at tavily.com

[telegram]
token = ""      # optional
chat_id = ""    # optional
```

---

## CLI Reference

```bash
arc init                  # first-time setup
arc chat                  # start chatting
arc chat -m <model>       # use a specific model
arc gateway               # run the WebSocket + WebChat server
arc listen                # voice input (requires arc gateway)
arc telegram              # run as a Telegram bot
arc workers --follow      # live-tail background activity
arc logs                  # view today's logs
```

Inside chat:
```
/skills          loaded skills and tools
/mcp             MCP server status
/memory          core facts · /memory episodic · /memory forget <id>
/jobs            scheduled jobs · /jobs cancel <name>
/workflow        list or run workflows · /workflow <name>
/cost            token usage this session
/clear           reset conversation
```

---

## Custom Skills

**Python skill** — drop in `~/.arc/skills/my_skill.py`:
```python
from arc.skills.base import Skill
from arc.core.types import SkillManifest, ToolResult, ToolSpec, Capability

class MySkill(Skill):
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_skill", version="1.0.0",
            description="Does something cool",
            tools=(ToolSpec(name="my_tool", description="...", parameters={...}),),
        )

    async def execute_tool(self, tool_name, arguments):
        return ToolResult(tool_call_id="", success=True, output="Done!")
```

**Soft skill** — drop a `.md` file in `~/.arc/skills/`. Its content is injected into every system prompt. Great for domain knowledge or personality tweaks.

---

## MCP (Model Context Protocol) Support

Arc can connect to external **MCP servers** as a client — giving it access to tools from any MCP-compatible service (GitHub, Jira, databases, etc.) without writing a single line of code.

### Setup

Create `~/.arc/mcp.json` (same format as Claude Desktop / Cursor):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_xxx" }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

SSE (remote) servers are also supported:
```json
{
  "mcpServers": {
    "remote": { "url": "http://localhost:8080/sse" }
  }
}
```

### How It Works

Arc uses a **gateway pattern** — no matter how many MCP servers you configure, only **2 tools** are added to the LLM context:

- **`mcp_list_tools`** — discover available servers and their tools
- **`mcp_call`** — invoke a tool on a specific server

Servers connect **lazily** on first use. If you have 10 servers configured but only use one, only that one starts up.

### Chat Commands

```
/mcp              show configured servers and connection status
/skills           MCP gateway appears alongside built-in skills
```

### Example

```
You:  "list files in my sandbox"
Arc:  → mcp_list_tools({})           — sees: filesystem, github, memory
      → mcp_call(server="filesystem", tool="list_directory",
                 arguments={"path": "/path/to/allowed/dir"})
      "Your sandbox contains: readme.txt, data.csv"
```

---

## Development

```bash
pip install -e ".[dev]"
pytest                    # 777 tests
pytest --cov=arc          # with coverage
```

---

## License

MIT

