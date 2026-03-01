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
                              │        Skills            │
                              │ files · terminal · web   │
                              │ browser · workers · jobs │
                              └──────────────────────────┘
```

**Micro-kernel design** — every subsystem is independent, wired through an event bus. Swap Ollama for Claude or GPT by changing one config line.

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
arc workers --follow      # live-tail background activity
```

Inside chat:
```
/skills          loaded skills and tools
/mcp             MCP server status
/memory          core facts · /memory episodic · /memory forget <id>
/jobs            scheduled jobs · /jobs cancel <name>
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
pytest                    # 481 tests
pytest --cov=arc          # with coverage
```

---

## License

MIT

