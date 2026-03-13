# Skills

Skills are one of Arc's main extension systems.

They let you add either:

- executable tools written in Python
- plain-text instruction files that shape agent behavior

## What Skills Are For

Use skills when you want Arc to gain:

- new local tools
- reusable team instructions
- domain-specific behavior
- custom workflows without editing the core app

## Built-In Skills

Arc already loads built-in skills at startup.

Examples include:

- filesystem access
- terminal execution
- task delegation
- browsing
- browser control
- code intelligence
- scheduler tools
- workflow tools
- MCP gateway tools

You can inspect loaded skills from chat with:

```text
/skills
```

## Two Kinds Of User Skills

## 1. Python Skills

Put a `.py` file in `~/.arc/skills/`.

Use this for custom tools with executable logic.

Example skeleton:

```python
from arc.skills.base import Skill
from arc.core.types import SkillManifest, ToolResult, ToolSpec

class MySkill(Skill):
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_skill",
            version="1.0.0",
            description="Does something useful",
            tools=(
                ToolSpec(
                    name="my_tool",
                    description="Run my custom action",
                    parameters={"type": "object", "properties": {}, "required": []},
                ),
            ),
        )

    async def execute_tool(self, tool_name, arguments):
        return ToolResult(tool_call_id="", success=True, output="Done")
```

Restart Arc and the skill is discovered automatically.

## 2. Soft Skills

Put a `.md` file in `~/.arc/skills/`.

Use this for instruction-only additions such as:

- writing rules
- company conventions
- domain language
- persistent reminders about how the agent should behave

Example:

```markdown
Always cite the source and date for factual claims.
Use concise bullet summaries before long explanations.
Prefer internal naming conventions used by the team.
```

## Where Skills Apply

Bundled strategy text and user soft skills are added into the prompting layer used by Arc.

In practice, skills are a good fit when you want Arc to feel smarter everywhere without building a full external integration.

## When To Use A Skill Instead Of MCP

Prefer a skill when:

- the logic should live locally in Arc
- you want a very small custom extension
- you want direct Python control
- you want instruction files, not just tools

If the tool already exists as an MCP server, MCP may be easier.
