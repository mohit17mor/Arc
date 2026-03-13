# Build Your Own Skill

If Arc almost does what you want but needs one more local capability, a custom skill is usually the right answer.

## Choose The Easiest Path First

### Soft skill

Use a markdown skill if you only need instruction text.

Good examples:

- output format rules
- team conventions
- domain expertise reminders
- tone and citation standards

Create:

- `~/.arc/skills/my_rules.md`

### Python skill

Use a Python skill if you need an actual tool call.

Good examples:

- a wrapper around an internal script
- a custom API call
- a local file processing tool
- a small automation that should feel native in Arc

Create:

- `~/.arc/skills/my_skill.py`

## Recommended Workflow

1. Start with the smallest useful tool.
2. Keep parameters simple and explicit.
3. Return clear success or failure text.
4. Restart Arc and verify it appears in `/skills`.

## Minimal Python Example

```python
from arc.skills.base import Skill
from arc.core.types import SkillManifest, ToolResult, ToolSpec

class PingSkill(Skill):
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="ping_skill",
            version="1.0.0",
            description="Simple example skill",
            tools=(
                ToolSpec(
                    name="ping",
                    description="Return pong",
                    parameters={"type": "object", "properties": {}, "required": []},
                ),
            ),
        )

    async def execute_tool(self, tool_name, arguments):
        return ToolResult(tool_call_id="", success=True, output="pong")
```

## Where To Learn More

- [Skills](skills.md) for how Arc discovers and loads them
- [MCP](mcp.md) if you might prefer an external server instead of a local Arc extension
