"""Skill for inspecting and updating Arc's shared MCP config."""

from __future__ import annotations

from typing import Any

from arc.core.types import SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill


class MCPConfigSkill(Skill):
    """Expose MCP config editing through the common config service."""

    def __init__(self) -> None:
        self._config_service: Any = None

    def set_dependencies(self, *, config_service: Any) -> None:
        self._config_service = config_service

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="mcp_config",
            version="1.0.0",
            description=(
                "Inspect or update Arc's shared MCP server configuration. "
                "Use only when the user explicitly wants to view or change configured MCP servers."
            ),
            tools=(
                ToolSpec(
                    name="mcp_get_config",
                    description=(
                        "Return the current MCP JSON config file so it can be reviewed or edited."
                    ),
                    parameters={"type": "object", "properties": {}, "required": []},
                ),
                ToolSpec(
                    name="mcp_set_config",
                    description=(
                        "Replace Arc's MCP JSON config with the provided full document. "
                        "The config is validated before it is saved and hot-reloaded."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Complete JSON text for the MCP config file.",
                            },
                        },
                        "required": ["text"],
                    },
                ),
            ),
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if self._config_service is None:
            return ToolResult(
                success=False,
                output="",
                error="MCP config service is not available in this runtime.",
            )

        if tool_name == "mcp_get_config":
            state = self._config_service.inspect()
            text = state["text"] if isinstance(state, dict) else state.text
            return ToolResult(success=True, output=text)

        if tool_name == "mcp_set_config":
            text = arguments.get("text", "")
            result = await self._config_service.save_and_reload(text)
            valid = result["valid"] if isinstance(result, dict) else result.valid
            applied = result["applied"] if isinstance(result, dict) else result.applied
            errors = result["errors"] if isinstance(result, dict) else result.errors
            active = (
                result["active_server_names"]
                if isinstance(result, dict)
                else result.active_server_names
            )
            if valid and applied:
                summary = ", ".join(active) if active else "none"
                return ToolResult(
                    success=True,
                    output=f"MCP config updated and reloaded. Active servers: {summary}",
                )
            error_text = "\n".join(errors) if errors else "MCP config update failed."
            return ToolResult(success=False, output="", error=error_text)

        return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")
