"""Tests for the MCP client module."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.mcp.client import MCPClient


class TestMCPClientInit:
    """Test MCPClient construction."""

    def test_stdio_transport(self):
        client = MCPClient(
            name="test",
            command="npx",
            args=["-y", "some-server"],
            env={"KEY": "val"},
        )
        assert client.name == "test"
        assert client.transport_type == "stdio"
        assert not client.connected

    def test_sse_transport(self):
        client = MCPClient(name="remote", url="http://localhost:8080/sse")
        assert client.transport_type == "sse"
        assert not client.connected

    def test_defaults(self):
        client = MCPClient(name="bare")
        assert client._args == []
        assert client._env == {}
        assert client._url == ""
        assert client._command == ""


class TestMCPClientNotConnected:
    """Test guards when client is not connected."""

    def test_list_tools_raises(self):
        client = MCPClient(name="x", command="echo")
        with pytest.raises(RuntimeError, match="not connected"):
            # list_tools is async but _ensure_connected is sync guard
            client._ensure_connected()

    @pytest.mark.asyncio
    async def test_call_tool_raises(self):
        client = MCPClient(name="x", command="echo")
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("anything", {})

    @pytest.mark.asyncio
    async def test_list_tools_async_raises(self):
        client = MCPClient(name="x", command="echo")
        with pytest.raises(RuntimeError, match="not connected"):
            await client.list_tools()


class TestMCPClientStdio:
    """Test stdio connection path."""

    @pytest.mark.asyncio
    async def test_connect_no_command_raises(self):
        client = MCPClient(name="bad")
        with pytest.raises(ConnectionError, match="Could not connect"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self):
        client = MCPClient(name="x", command="echo")
        client._connected = True
        # Should return immediately without doing anything
        await client.connect()
        assert client.connected


class TestMCPClientListTools:
    """Test list_tools with mocked session."""

    @pytest.mark.asyncio
    async def test_list_tools_parses_response(self):
        client = MCPClient(name="gh", command="npx")
        client._connected = True

        # Mock tool objects
        tool1 = MagicMock()
        tool1.name = "list_repos"
        tool1.description = "List repositories"
        tool1.inputSchema = {"type": "object", "properties": {"owner": {"type": "string"}}}

        tool2 = MagicMock()
        tool2.name = "create_issue"
        tool2.description = None
        tool2.inputSchema = None

        mock_result = MagicMock()
        mock_result.tools = [tool1, tool2]

        client._session = AsyncMock()
        client._session.list_tools = AsyncMock(return_value=mock_result)

        tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "list_repos"
        assert tools[0]["description"] == "List repositories"
        assert "owner" in tools[0]["inputSchema"]["properties"]

        # None description/schema get defaults
        assert tools[1]["name"] == "create_issue"
        assert tools[1]["description"] == ""
        assert tools[1]["inputSchema"]["type"] == "object"


class TestMCPClientCallTool:
    """Test call_tool with mocked session."""

    @pytest.mark.asyncio
    async def test_call_tool_text_content(self):
        """TextContent is extracted properly."""
        client = MCPClient(name="gh", command="npx")
        client._connected = True

        text_block = MagicMock()
        text_block.text = "result data"
        mock_result = MagicMock()
        mock_result.content = [text_block]
        mock_result.isError = False

        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)

        output, is_error = await client.call_tool("list_repos", {"owner": "arc"})

        assert output == "result data"
        assert is_error is False
        client._session.call_tool.assert_called_once_with("list_repos", {"owner": "arc"})

    @pytest.mark.asyncio
    async def test_call_tool_error_flag(self):
        client = MCPClient(name="gh", command="npx")
        client._connected = True

        text_block = MagicMock()
        text_block.text = "something went wrong"
        mock_result = MagicMock()
        mock_result.content = [text_block]
        mock_result.isError = True

        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)

        output, is_error = await client.call_tool("bad_tool", {})

        assert is_error is True
        assert "something went wrong" in output

    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self):
        client = MCPClient(name="gh", command="npx")
        client._connected = True

        mock_result = MagicMock()
        mock_result.content = []
        mock_result.isError = False

        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)

        output, is_error = await client.call_tool("empty_tool", {})

        assert output == "(no output)"
        assert is_error is False

    @pytest.mark.asyncio
    async def test_call_tool_non_text_content(self):
        """Non-TextContent blocks produce a type description."""
        client = MCPClient(name="gh", command="npx")
        client._connected = True

        # Simulate an image content block
        image_block = MagicMock(spec=[])  # no text attribute
        type(image_block).__name__ = "ImageContent"

        mock_result = MagicMock()
        mock_result.content = [image_block]
        mock_result.isError = False

        client._session = AsyncMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)

        output, is_error = await client.call_tool("image_tool", {})

        assert "[ImageContent]" in output


class TestMCPClientClose:
    """Test close and ping."""

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        client = MCPClient(name="x", command="echo")
        client._connected = True
        client._session = AsyncMock()
        client._exit_stack = AsyncMock()

        await client.close()

        assert not client.connected
        assert client._session is None
        assert client._exit_stack is None

    @pytest.mark.asyncio
    async def test_ping_when_not_connected(self):
        client = MCPClient(name="x", command="echo")
        result = await client.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_when_connected(self):
        client = MCPClient(name="x", command="echo")
        client._connected = True
        client._session = AsyncMock()
        client._session.send_ping = AsyncMock()

        result = await client.ping()
        assert result is True
