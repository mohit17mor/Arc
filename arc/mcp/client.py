"""
MCPClient — low-level wrapper around the official ``mcp`` SDK.

Manages the lifecycle of a single MCP server connection:
  connect()   → spawn process (stdio) or open HTTP stream (SSE)
  list_tools() → fetch available tools from the server
  call_tool()  → invoke a tool and return the result text
  close()      → tear down transport + session cleanly

Thread-safety: one MCPClient per server, all calls are async.
Concurrency across multiple tool calls on the same client is handled
by the SDK's internal message routing.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Async client for a single MCP server.

    Usage::

        client = MCPClient(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "ghp_xxx"},
        )
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("list_repos", {"owner": "arc"})
        await client.close()
    """

    def __init__(
        self,
        name: str,
        *,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str = "",
    ) -> None:
        self.name = name
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._url = url

        self._exit_stack: AsyncExitStack | None = None
        self._session: Any | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def transport_type(self) -> str:
        return "sse" if self._url else "stdio"

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        if self._connected:
            return

        self._exit_stack = AsyncExitStack()

        try:
            if self._url:
                await self._connect_sse()
            else:
                await self._connect_stdio()

            # Initialize the MCP session (capability negotiation)
            result = await self._session.initialize()
            logger.info(
                f"MCP server '{self.name}' connected "
                f"(protocol={result.protocolVersion}, "
                f"server={result.serverInfo.name if result.serverInfo else 'unknown'})"
            )
            self._connected = True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.name}': {e}")
            await self.close()
            raise ConnectionError(
                f"Could not connect to MCP server '{self.name}': {e}"
            ) from e

    async def _connect_stdio(self) -> None:
        """Connect via stdio (spawn a subprocess)."""
        if not self._command:
            raise ValueError(
                f"MCP server '{self.name}': no command specified for stdio transport"
            )

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env={**self._env} if self._env else None,
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def _connect_sse(self) -> None:
        """Connect via SSE (HTTP streaming)."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        sse_transport = await self._exit_stack.enter_async_context(
            sse_client(url=self._url)
        )
        read_stream, write_stream = sse_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        Fetch available tools from the server.

        Returns a list of dicts with keys: name, description, inputSchema.
        """
        self._ensure_connected()

        result = await self._session.list_tools()

        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema or {
                    "type": "object",
                    "properties": {},
                },
            })

        logger.debug(
            f"MCP server '{self.name}': {len(tools)} tools available"
        )
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """
        Call a tool on the MCP server.

        Returns:
            (output_text, is_error) — the concatenated text content and
            whether the server flagged it as an error.
        """
        self._ensure_connected()

        result = await self._session.call_tool(tool_name, arguments or {})

        # Extract text from content blocks
        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(str(block.text))
            else:
                # Image, audio, resource — describe it
                parts.append(f"[{type(block).__name__}]")

        output = "\n".join(parts) if parts else "(no output)"
        return output, bool(result.isError)

    async def ping(self) -> bool:
        """Check if the server is responsive."""
        try:
            self._ensure_connected()
            await self._session.send_ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Tear down the connection cleanly."""
        self._connected = False
        self._session = None
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except BaseException as e:
                # CancelledError (BaseException) is common during shutdown
                # when the subprocess hasn't exited yet. Safe to suppress.
                logger.debug(f"Error closing MCP client '{self.name}': {e}")
            self._exit_stack = None

    def _ensure_connected(self) -> None:
        if not self._connected or self._session is None:
            raise RuntimeError(
                f"MCP server '{self.name}' is not connected. "
                f"Call connect() first."
            )
