"""Tests for the MCPServerSkill module."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.core.types import Capability, ToolResult
from arc.mcp.skill import MCPServerSkill


class TestMCPServerSkillInit:
    """Test construction and manifest."""

    def test_creates_client(self):
        skill = MCPServerSkill(
            server_name="github",
            command="npx",
            args=["-y", "@mcp/server-github"],
            env={"TOKEN": "abc"},
        )
        assert skill.server_name == "github"
        assert skill.transport_type == "stdio"
        assert not skill.is_connected

    def test_sse_transport(self):
        skill = MCPServerSkill(
            server_name="remote",
            url="http://localhost:8080/sse",
        )
        assert skill.transport_type == "sse"

    def test_manifest_before_activation(self):
        """Before activation, manifest has no tools."""
        skill = MCPServerSkill(server_name="test", command="echo")
        manifest = skill.manifest()

        assert manifest.name == "mcp_test"
        assert manifest.version == "1.0.0"
        assert "MCP server" in manifest.description
        assert Capability.MCP in manifest.capabilities
        assert len(manifest.tools) == 0


class TestMCPServerSkillActivation:
    """Test activation (connect + fetch tools)."""

    @pytest.mark.asyncio
    async def test_activate_connects_and_fetches_tools(self):
        skill = MCPServerSkill(server_name="gh", command="npx")

        # Mock the internal client
        skill._client = AsyncMock()
        skill._client.connected = True
        skill._client.transport_type = "stdio"
        skill._client.connect = AsyncMock()
        skill._client.list_tools = AsyncMock(return_value=[
            {
                "name": "list_repos",
                "description": "List repos",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "create_issue",
                "description": "Create an issue",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                    },
                },
            },
        ])

        await skill.activate()

        skill._client.connect.assert_called_once()
        skill._client.list_tools.assert_called_once()

        manifest = skill.manifest()
        assert len(manifest.tools) == 2

        # Check prefixed names
        tool_names = [t.name for t in manifest.tools]
        assert "mcp_gh__list_repos" in tool_names
        assert "mcp_gh__create_issue" in tool_names

        # Check descriptions include server prefix
        for tool in manifest.tools:
            assert "[MCP: gh]" in tool.description

        # Check capability
        for tool in manifest.tools:
            assert Capability.MCP in tool.required_capabilities

    @pytest.mark.asyncio
    async def test_activate_idempotent(self):
        """Calling activate() twice doesn't reconnect."""
        skill = MCPServerSkill(server_name="gh", command="npx")
        skill._client = AsyncMock()
        skill._client.connected = True
        skill._client.transport_type = "stdio"
        skill._client.connect = AsyncMock()
        skill._client.list_tools = AsyncMock(return_value=[])

        await skill.activate()
        await skill.activate()

        # Only called once
        skill._client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_failure_propagates(self):
        skill = MCPServerSkill(server_name="bad", command="nonexistent")
        skill._client = AsyncMock()
        skill._client.connect = AsyncMock(
            side_effect=ConnectionError("failed")
        )

        with pytest.raises(ConnectionError):
            await skill.activate()


class TestMCPServerSkillExecuteTool:
    """Test tool execution forwarding."""

    @pytest.mark.asyncio
    async def test_execute_tool_forwards_to_client(self):
        skill = MCPServerSkill(server_name="gh", command="npx")
        skill._activated = True
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(return_value=("repos list", False))

        result = await skill.execute_tool(
            "mcp_gh__list_repos", {"owner": "arc"}
        )

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "repos list"
        # Check it stripped the prefix correctly
        skill._client.call_tool.assert_called_once_with(
            "list_repos", {"owner": "arc"}
        )

    @pytest.mark.asyncio
    async def test_execute_tool_error_from_server(self):
        skill = MCPServerSkill(server_name="gh", command="npx")
        skill._activated = True
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(
            return_value=("not found", True)
        )

        result = await skill.execute_tool("mcp_gh__bad_tool", {})

        assert result.success is False
        assert "not found" in result.output

    @pytest.mark.asyncio
    async def test_execute_tool_not_activated(self):
        skill = MCPServerSkill(server_name="gh", command="npx")

        result = await skill.execute_tool("mcp_gh__anything", {})

        assert result.success is False
        assert "not connected" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self):
        skill = MCPServerSkill(server_name="gh", command="npx")
        skill._activated = True
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(side_effect=RuntimeError("boom"))

        result = await skill.execute_tool("mcp_gh__thing", {})

        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_unprefixed_name(self):
        """If tool_name doesn't have the prefix, pass it through."""
        skill = MCPServerSkill(server_name="gh", command="npx")
        skill._activated = True
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(return_value=("ok", False))

        await skill.execute_tool("raw_tool", {})

        # Should pass through as-is
        skill._client.call_tool.assert_called_once_with("raw_tool", {})


class TestMCPServerSkillShutdown:
    """Test shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        skill = MCPServerSkill(server_name="gh", command="npx")
        skill._activated = True
        skill._client = AsyncMock()
        skill._client.close = AsyncMock()

        await skill.shutdown()

        skill._client.close.assert_called_once()
        assert not skill._activated
