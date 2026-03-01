"""
MCPGatewaySkill — single Arc skill that proxies ALL MCP servers.

Instead of registering N tools per MCP server (bloating the LLM
context), the gateway exposes exactly **2 tools** regardless of how
many servers or server-tools exist:

    mcp_list_tools  — discover what's available on a server
    mcp_call        — invoke a tool on a server

Servers are connected **lazily** on first use — no startup cost,
no wasted connections for servers the user never touches.

This is the skill that gets registered with the SkillManager.
The individual ``MCPServerSkill`` instances are still used internally
for per-server lifecycle management.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.mcp.manager import MCPManager
from arc.skills.base import Skill

logger = logging.getLogger(__name__)

# ── Tool specs (constant — always exactly 2) ──────────────────────

_LIST_TOOLS_SPEC = ToolSpec(
    name="mcp_list_tools",
    description=(
        "List available MCP servers and their tools. "
        "Call with no arguments to see all configured servers. "
        "Call with server=<name> to connect to that server and list its tools."
    ),
    parameters={
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": (
                    "Name of the MCP server to query. "
                    "Omit to list all configured servers."
                ),
            },
        },
    },
    required_capabilities=frozenset([Capability.MCP]),
)

_CALL_TOOL_SPEC = ToolSpec(
    name="mcp_call",
    description=(
        "Call a tool on an MCP server. The server will be connected "
        "automatically on first use. Use mcp_list_tools first to "
        "discover available tools and their parameters."
    ),
    parameters={
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": "Name of the MCP server (e.g. 'github', 'filesystem').",
            },
            "tool": {
                "type": "string",
                "description": "Name of the tool to call on that server.",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments to pass to the tool (server-specific).",
            },
        },
        "required": ["server", "tool"],
    },
    required_capabilities=frozenset([Capability.MCP]),
)


class MCPGatewaySkill(Skill):
    """
    Lightweight gateway that proxies all MCP servers through 2 tools.

    Keeps the LLM tool list small no matter how many MCP servers or
    server-tools are configured.  Servers connect lazily on first
    ``mcp_call`` or ``mcp_list_tools(server=...)`` invocation.
    """

    def __init__(self, mcp_manager: MCPManager) -> None:
        self._manager = mcp_manager

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="mcp_gateway",
            version="1.0.0",
            description=(
                "Gateway to external MCP servers. "
                f"Configured: {', '.join(self._manager.server_names) or 'none'}"
            ),
            capabilities=frozenset([Capability.MCP]),
            tools=(_LIST_TOOLS_SPEC, _CALL_TOOL_SPEC),
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name == "mcp_list_tools":
            return await self._handle_list_tools(arguments)
        elif tool_name == "mcp_call":
            return await self._handle_call(arguments)
        else:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Unknown MCP gateway tool: {tool_name}",
            )

    # ── mcp_list_tools ─────────────────────────────────────────────

    async def _handle_list_tools(
        self, arguments: dict[str, Any],
    ) -> ToolResult:
        server_name = arguments.get("server")

        if not server_name:
            # List all configured servers (no connection needed)
            info_lines = []
            for info in self._manager.server_info():
                status = "connected" if info["connected"] else "not connected"
                line = (
                    f"  {info['name']}  ({info['transport']})  "
                    f"[{status}]  {info['tools']} tools loaded"
                )
                if info.get("hint"):
                    line += f"  — {info['hint']}"
                info_lines.append(line)
            if not info_lines:
                return ToolResult(
                    tool_call_id="",
                    success=True,
                    output="No MCP servers configured. Add servers to ~/.arc/mcp.json",
                )
            return ToolResult(
                tool_call_id="",
                success=True,
                output="MCP servers:\n" + "\n".join(info_lines),
            )

        # List tools for a specific server (connects lazily)
        skill = self._manager.get_skill(server_name)
        if not skill:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=(
                    f"Unknown MCP server: '{server_name}'. "
                    f"Available: {', '.join(self._manager.server_names)}"
                ),
            )

        try:
            await skill.activate()
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Failed to connect to MCP server '{server_name}': {e}",
            )

        # Return the tool list from this server
        tools = await skill._client.list_tools()
        lines = [f"Tools on '{server_name}' ({len(tools)}):"]
        for t in tools:
            desc = t.get("description", "")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            lines.append(f"  {t['name']}  —  {desc}")

            # Show parameters if any
            schema = t.get("inputSchema", {})
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            if props:
                for pname, pschema in props.items():
                    req_marker = " (required)" if pname in required else ""
                    ptype = pschema.get("type", "any")
                    pdesc = pschema.get("description", "")
                    if len(pdesc) > 60:
                        pdesc = pdesc[:57] + "..."
                    lines.append(f"    {pname}: {ptype}{req_marker}  {pdesc}")

        return ToolResult(
            tool_call_id="",
            success=True,
            output="\n".join(lines),
        )

    # ── mcp_call ───────────────────────────────────────────────────

    async def _handle_call(
        self, arguments: dict[str, Any],
    ) -> ToolResult:
        server_name = arguments.get("server", "")
        tool_name = arguments.get("tool", "")

        if not server_name or not tool_name:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="Both 'server' and 'tool' are required.",
            )

        skill = self._manager.get_skill(server_name)
        if not skill:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=(
                    f"Unknown MCP server: '{server_name}'. "
                    f"Available: {', '.join(self._manager.server_names)}"
                ),
            )

        # Lazy connect
        try:
            await skill.activate()
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Failed to connect to MCP server '{server_name}': {e}",
            )

        # Forward the call — use the raw tool name (no prefix needed)
        tool_args = arguments.get("arguments", {})
        try:
            output, is_error = await skill._client.call_tool(
                tool_name, tool_args,
            )

            if is_error:
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    output=output,
                    error=f"MCP server '{server_name}' returned error",
                )

            return ToolResult(
                tool_call_id="",
                success=True,
                output=output,
            )

        except Exception as e:
            logger.error(
                f"MCP tool call failed: {server_name}/{tool_name}: {e}"
            )
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"MCP tool call failed: {e}",
            )

    async def shutdown(self) -> None:
        """Shut down all connected MCP servers."""
        for skill in self._manager._skills.values():
            try:
                await skill.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down MCP server: {e}")
