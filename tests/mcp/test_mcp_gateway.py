"""Tests for the MCPGatewaySkill — the 2-tool gateway pattern."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from arc.core.types import Capability, ToolResult
from arc.mcp.gateway import MCPGatewaySkill
from arc.mcp.manager import MCPManager


# ── helpers ────────────────────────────────────────────────────────

def _make_manager(tmp_path: Path, servers: dict) -> MCPManager:
    """Create an MCPManager backed by a tmp mcp.json."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    mgr = MCPManager(config_path=cfg)
    mgr.discover()
    return mgr


def _make_gateway(tmp_path: Path, servers: dict | None = None) -> MCPGatewaySkill:
    if servers is None:
        servers = {
            "github": {"command": "npx", "args": ["server-gh"]},
            "db": {"command": "python", "args": ["-m", "db_server"]},
        }
    mgr = _make_manager(tmp_path, servers)
    return MCPGatewaySkill(mgr)


# ── manifest ───────────────────────────────────────────────────────

class TestGatewayManifest:
    def test_always_two_tools(self, tmp_path):
        gw = _make_gateway(tmp_path)
        m = gw.manifest()
        assert m.name == "mcp_gateway"
        assert len(m.tools) == 2
        tool_names = {t.name for t in m.tools}
        assert tool_names == {"mcp_list_tools", "mcp_call"}
        assert Capability.MCP in m.capabilities

    def test_description_mentions_servers(self, tmp_path):
        gw = _make_gateway(tmp_path)
        m = gw.manifest()
        assert "github" in m.description
        assert "db" in m.description

    def test_empty_servers(self, tmp_path):
        gw = _make_gateway(tmp_path, servers={})
        m = gw.manifest()
        # Still 2 tools — the gateway always exists
        assert len(m.tools) == 2
        assert "none" in m.description


# ── mcp_list_tools (no server arg) ─────────────────────────────────

class TestListToolsOverview:
    @pytest.mark.asyncio
    async def test_lists_all_servers(self, tmp_path):
        gw = _make_gateway(tmp_path)
        result = await gw.execute_tool("mcp_list_tools", {})
        assert result.success
        assert "github" in result.output
        assert "db" in result.output

    @pytest.mark.asyncio
    async def test_no_servers(self, tmp_path):
        gw = _make_gateway(tmp_path, servers={})
        result = await gw.execute_tool("mcp_list_tools", {})
        assert result.success
        assert "No MCP servers" in result.output


# ── mcp_list_tools (specific server) ──────────────────────────────

class TestListToolsServer:
    @pytest.mark.asyncio
    async def test_unknown_server(self, tmp_path):
        gw = _make_gateway(tmp_path)
        result = await gw.execute_tool("mcp_list_tools", {"server": "nope"})
        assert not result.success
        assert "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_connect_failure(self, tmp_path):
        gw = _make_gateway(tmp_path)
        # Mock the skill's activate to fail
        skill = gw._manager.get_skill("github")
        skill.activate = AsyncMock(side_effect=ConnectionError("refused"))

        result = await gw.execute_tool("mcp_list_tools", {"server": "github"})
        assert not result.success
        assert "Failed to connect" in result.error

    @pytest.mark.asyncio
    async def test_lists_server_tools(self, tmp_path):
        gw = _make_gateway(tmp_path)
        skill = gw._manager.get_skill("github")

        # Mock activate + client.list_tools
        skill.activate = AsyncMock()
        skill._client = AsyncMock()
        skill._client.list_tools = AsyncMock(return_value=[
            {
                "name": "list_repos",
                "description": "List repositories for a user",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repo owner"},
                    },
                    "required": ["owner"],
                },
            },
            {
                "name": "create_issue",
                "description": "Create a new issue",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            },
        ])

        result = await gw.execute_tool("mcp_list_tools", {"server": "github"})
        assert result.success
        assert "list_repos" in result.output
        assert "create_issue" in result.output
        assert "owner" in result.output  # parameter detail
        assert "(required)" in result.output


# ── mcp_call ───────────────────────────────────────────────────────

class TestMCPCall:
    @pytest.mark.asyncio
    async def test_missing_server(self, tmp_path):
        gw = _make_gateway(tmp_path)
        result = await gw.execute_tool("mcp_call", {"tool": "x"})
        assert not result.success
        assert "required" in result.error

    @pytest.mark.asyncio
    async def test_missing_tool(self, tmp_path):
        gw = _make_gateway(tmp_path)
        result = await gw.execute_tool("mcp_call", {"server": "github"})
        assert not result.success
        assert "required" in result.error

    @pytest.mark.asyncio
    async def test_unknown_server(self, tmp_path):
        gw = _make_gateway(tmp_path)
        result = await gw.execute_tool("mcp_call", {
            "server": "nope",
            "tool": "x",
        })
        assert not result.success
        assert "Unknown MCP server" in result.error

    @pytest.mark.asyncio
    async def test_connect_failure(self, tmp_path):
        gw = _make_gateway(tmp_path)
        skill = gw._manager.get_skill("github")
        skill.activate = AsyncMock(side_effect=ConnectionError("refused"))

        result = await gw.execute_tool("mcp_call", {
            "server": "github",
            "tool": "list_repos",
        })
        assert not result.success
        assert "Failed to connect" in result.error

    @pytest.mark.asyncio
    async def test_successful_call(self, tmp_path):
        gw = _make_gateway(tmp_path)
        skill = gw._manager.get_skill("github")
        skill.activate = AsyncMock()
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(
            return_value=("repo1\nrepo2\nrepo3", False)
        )

        result = await gw.execute_tool("mcp_call", {
            "server": "github",
            "tool": "list_repos",
            "arguments": {"owner": "arc"},
        })

        assert result.success
        assert "repo1" in result.output
        skill._client.call_tool.assert_called_once_with(
            "list_repos", {"owner": "arc"}
        )

    @pytest.mark.asyncio
    async def test_server_error_flag(self, tmp_path):
        gw = _make_gateway(tmp_path)
        skill = gw._manager.get_skill("github")
        skill.activate = AsyncMock()
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(
            return_value=("not found", True)
        )

        result = await gw.execute_tool("mcp_call", {
            "server": "github",
            "tool": "bad_tool",
        })
        assert not result.success
        assert "not found" in result.output

    @pytest.mark.asyncio
    async def test_call_exception(self, tmp_path):
        gw = _make_gateway(tmp_path)
        skill = gw._manager.get_skill("github")
        skill.activate = AsyncMock()
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        result = await gw.execute_tool("mcp_call", {
            "server": "github",
            "tool": "list_repos",
        })
        assert not result.success
        assert "network error" in result.error

    @pytest.mark.asyncio
    async def test_call_without_arguments(self, tmp_path):
        """Arguments default to {} when omitted."""
        gw = _make_gateway(tmp_path)
        skill = gw._manager.get_skill("github")
        skill.activate = AsyncMock()
        skill._client = AsyncMock()
        skill._client.call_tool = AsyncMock(return_value=("ok", False))

        result = await gw.execute_tool("mcp_call", {
            "server": "github",
            "tool": "status",
        })
        assert result.success
        skill._client.call_tool.assert_called_once_with("status", {})


# ── unknown tool ───────────────────────────────────────────────────

class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_name(self, tmp_path):
        gw = _make_gateway(tmp_path)
        result = await gw.execute_tool("mcp_whatever", {})
        assert not result.success
        assert "Unknown" in result.error


# ── shutdown ───────────────────────────────────────────────────────

class TestGatewayShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_all(self, tmp_path):
        gw = _make_gateway(tmp_path)
        for skill in gw._manager._skills.values():
            skill.shutdown = AsyncMock()

        await gw.shutdown()

        for skill in gw._manager._skills.values():
            skill.shutdown.assert_called_once()
