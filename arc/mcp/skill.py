"""
MCPServerSkill — wraps a single MCP server as an Arc Skill.

One instance per configured MCP server.  Tools from the server are
exposed as native Arc tools — the LLM, agent loop, and skill manager
see no difference between MCP tools and builtin tools.

Lifecycle:
    1. ``__init__`` stores config (no connection yet)
    2. ``manifest()`` returns a placeholder — real tools are fetched on activate
    3. ``activate()`` connects to the MCP server, fetches tools
    4. ``execute_tool()`` forwards calls to the server
    5. ``shutdown()`` tears down the connection
"""

from __future__ import annotations

import logging
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.mcp.client import MCPClient
from arc.skills.base import Skill

logger = logging.getLogger(__name__)


class MCPServerSkill(Skill):
    """
    Arc skill backed by an external MCP server.

    Each configured MCP server becomes one MCPServerSkill instance.
    Tools are dynamically fetched from the server on activation
    and exposed via the standard ``manifest()`` interface.
    """

    def __init__(
        self,
        server_name: str,
        *,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str = "",
    ) -> None:
        self._server_name = server_name
        self._client = MCPClient(
            name=server_name,
            command=command,
            args=args,
            env=env,
            url=url,
        )
        self._tool_specs: tuple[ToolSpec, ...] = ()
        self._tool_names: set[str] = set()
        self._activated = False

    def manifest(self) -> SkillManifest:
        """
        Return skill metadata with dynamically-fetched tools.

        Before ``activate()``, returns an empty tool list — the skill
        manager will call ``activate()`` lazily on first tool use.
        After activation, returns the real tools from the MCP server.
        """
        return SkillManifest(
            name=f"mcp_{self._server_name}",
            version="1.0.0",
            description=f"MCP server: {self._server_name}",
            capabilities=frozenset([Capability.MCP]),
            tools=self._tool_specs,
        )

    async def activate(self) -> None:
        """Connect to the MCP server and fetch its tools."""
        if self._activated:
            return

        try:
            await self._client.connect()
            await self._refresh_tools()
            self._activated = True
            logger.info(
                f"MCP skill '{self._server_name}' activated "
                f"with {len(self._tool_specs)} tools"
            )
        except Exception as e:
            logger.error(
                f"Failed to activate MCP skill '{self._server_name}': {e}"
            )
            raise

    async def _refresh_tools(self) -> None:
        """Fetch the tool list from the MCP server and convert to ToolSpecs."""
        raw_tools = await self._client.list_tools()

        specs = []
        names = set()
        for tool in raw_tools:
            # Prefix tool names with server name to avoid collisions
            # e.g., "github" server's "list_repos" → "mcp_github__list_repos"
            original_name = tool["name"]
            prefixed_name = f"mcp_{self._server_name}__{original_name}"

            spec = ToolSpec(
                name=prefixed_name,
                description=(
                    f"[MCP: {self._server_name}] "
                    f"{tool.get('description', original_name)}"
                ),
                parameters=tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                }),
                required_capabilities=frozenset([Capability.MCP]),
            )
            specs.append(spec)
            names.add(prefixed_name)

        self._tool_specs = tuple(specs)
        self._tool_names = names

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Forward a tool call to the MCP server."""
        if not self._activated:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"MCP server '{self._server_name}' is not connected",
            )

        # Strip the prefix to get the original MCP tool name
        # "mcp_github__list_repos" → "list_repos"
        prefix = f"mcp_{self._server_name}__"
        if tool_name.startswith(prefix):
            original_name = tool_name[len(prefix):]
        else:
            original_name = tool_name

        try:
            output, is_error = await self._client.call_tool(
                original_name, arguments,
            )

            if is_error:
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    output=output,
                    error=f"MCP server returned error",
                )

            return ToolResult(
                tool_call_id="",
                success=True,
                output=output,
            )

        except Exception as e:
            logger.error(
                f"MCP tool call failed: {tool_name} on "
                f"'{self._server_name}': {e}"
            )
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"MCP tool call failed: {e}",
            )

    async def shutdown(self) -> None:
        """Disconnect from the MCP server."""
        if self._client:
            await self._client.close()
        self._activated = False
        logger.info(f"MCP skill '{self._server_name}' shut down")

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def is_connected(self) -> bool:
        return self._client.connected if self._client else False

    @property
    def transport_type(self) -> str:
        return self._client.transport_type if self._client else "unknown"
