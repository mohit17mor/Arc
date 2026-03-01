"""Tests for the MCPManager module."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from arc.mcp.manager import MCPManager
from arc.mcp.skill import MCPServerSkill


class TestMCPManagerDiscover:
    """Test config loading and server discovery."""

    def test_no_config_file(self, tmp_path):
        """Missing config → empty list, no error."""
        manager = MCPManager(config_path=tmp_path / "nonexistent.json")
        skills = manager.discover()
        assert skills == []
        assert not manager.has_servers

    def test_empty_config(self, tmp_path):
        """Empty JSON object → no servers."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text("{}", encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        skills = manager.discover()
        assert skills == []

    def test_claude_desktop_format(self, tmp_path):
        """Standard mcpServers key works."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@mcp/server-github"],
                    "env": {"TOKEN": "abc"},
                },
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@mcp/server-fs", "/home"],
                },
            }
        }), encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        skills = manager.discover()

        assert len(skills) == 2
        assert all(isinstance(s, MCPServerSkill) for s in skills)
        assert manager.has_servers
        assert set(manager.server_names) == {"github", "filesystem"}

    def test_flat_servers_format(self, tmp_path):
        """Alternate 'servers' key also works."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "servers": {
                "db": {"command": "python", "args": ["-m", "db_server"]},
            }
        }), encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        skills = manager.discover()

        assert len(skills) == 1
        assert skills[0].server_name == "db"

    def test_sse_server(self, tmp_path):
        """SSE servers use url instead of command."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "remote": {"url": "http://localhost:8080/sse"},
            }
        }), encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        skills = manager.discover()

        assert len(skills) == 1
        assert skills[0].transport_type == "sse"

    def test_skip_invalid_entry(self, tmp_path):
        """Entries without command or url are skipped."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "good": {"command": "echo"},
                "bad": {"env": {"only": "env"}},  # no command or url
                "also_bad": "just a string",  # not a dict
            }
        }), encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        skills = manager.discover()

        assert len(skills) == 1
        assert skills[0].server_name == "good"

    def test_invalid_json(self, tmp_path):
        """Malformed JSON → empty list, no crash."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text("{not valid json", encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        skills = manager.discover()
        assert skills == []

    def test_mixed_transports(self, tmp_path):
        """Mix of stdio and SSE servers."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "local": {"command": "node", "args": ["server.js"]},
                "remote": {"url": "http://api.example.com/mcp"},
            }
        }), encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        skills = manager.discover()

        assert len(skills) == 2
        transports = {s.server_name: s.transport_type for s in skills}
        assert transports["local"] == "stdio"
        assert transports["remote"] == "sse"


class TestMCPManagerServerInfo:
    """Test the server_info reporting method."""

    def test_server_info_before_discover(self):
        manager = MCPManager(config_path=Path("/nonexistent"))
        assert manager.server_info() == []

    def test_server_info_after_discover(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "gh": {"command": "npx", "args": ["server-gh"]},
            }
        }), encoding="utf-8")

        manager = MCPManager(config_path=cfg)
        manager.discover()

        info_list = manager.server_info()
        assert len(info_list) == 1
        info = info_list[0]
        assert info["name"] == "gh"
        assert info["connected"] is False  # not activated yet
        assert info["transport"] == "stdio"
        assert info["tools"] == 0  # no tools until activated
        assert info["skill_name"] == "mcp_gh"


class TestMCPManagerDefaultPath:
    """Test default config path."""

    def test_default_path(self):
        manager = MCPManager()
        assert manager._config_path == Path.home() / ".arc" / "mcp.json"
