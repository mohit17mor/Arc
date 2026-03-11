# Skills & MCP

## Built-in Skills

Arc comes with skills that are auto-discovered at startup:

| Skill | Type | Tools |
|-------|------|-------|
| filesystem | always-on | `read_file`, `write_file`, `list_directory` |
| terminal | always-on | `execute` (shell commands) |
| worker | always-on | `delegate_task`, `list_workers` |
| task_board | always-on | `queue_task`, `list_tasks`, `task_detail`, `cancel_task`, `reply_to_task`, `list_agents` |
| browsing | on-demand | `web_search`, `web_read` |
| browser_control | on-demand | `browser_go`, `browser_act`, `browser_look` |
| code_intel | on-demand | `repo_map`, `find_symbol`, `search_code` |
| scheduler | on-demand | `schedule_job`, `list_jobs`, `cancel_job` |
| liquid_web | on-demand | `liquid_search` |
| mcp_gateway | on-demand | `mcp_list_tools`, `mcp_call` |
| workflow | on-demand | `run_workflow`, `list_workflows` |

## Custom Python Skills

Drop a `.py` file in `~/.arc/skills/`:

```python
from arc.skills.base import Skill
from arc.core.types import SkillManifest, ToolResult, ToolSpec, Capability

class MySkill(Skill):
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_skill", version="1.0.0",
            description="Does something cool",
            tools=(ToolSpec(
                name="my_tool",
                description="...",
                parameters={"type": "object", "properties": {}, "required": []},
            ),),
        )

    async def execute_tool(self, tool_name, arguments):
        return ToolResult(tool_call_id="", success=True, output="Done!")
```

Restart Arc → skill is automatically discovered and registered.

## Soft Skills

Drop a `.md` file in `~/.arc/skills/`. Its content is injected into every agent's system prompt. Great for domain knowledge or personality tweaks.

```markdown
<!-- ~/.arc/skills/finance_expert.md -->
Always use precise financial terminology.
When discussing stocks, include the ticker symbol.
Cite data sources with dates.
```

## MCP (Model Context Protocol)

Arc connects to external MCP servers, giving agents access to tools from any MCP-compatible service.

### Setup

Create `~/.arc/mcp.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_xxx" }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
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

### Gateway Pattern

No matter how many MCP servers you configure, only **2 tools** are added to the LLM context:

- `mcp_list_tools` — discover available servers and their tools
- `mcp_call` — invoke a tool on a specific server

Servers connect **lazily** on first use — no startup cost for unused servers.

### Commands

```
/mcp              # show configured servers and status
/skills           # MCP gateway appears alongside built-in skills
```
