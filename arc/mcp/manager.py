"""
MCPManager — discovers and manages all configured MCP servers.

Reads ``~/.arc/mcp.json``, instantiates one ``MCPServerSkill`` per
server, and provides them to the skill manager for registration.

The MCP config format is compatible with Claude Desktop / Cursor::

    {
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "ghp_xxx"}
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
            }
        }
    }

For SSE servers, use ``"url"`` instead of ``"command"``::

    {
        "mcpServers": {
            "remote": {
                "url": "http://localhost:8080/sse"
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from arc.mcp.skill import MCPServerSkill

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Manages MCP server discovery and lifecycle.

    Usage::

        manager = MCPManager()
        skills = manager.discover()        # → list[MCPServerSkill]

        # Register with Arc's SkillManager
        for skill in skills:
            await skill_manager.register(skill)

        # Status
        for info in manager.server_info():
            print(f"{info['name']} — {info['transport']} — {info['tools']} tools")
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or Path.home() / ".arc" / "mcp.json"
        self._skills: dict[str, MCPServerSkill] = {}
        self._server_configs: dict[str, dict[str, Any]] = {}  # raw config per server

    def discover(self) -> list[MCPServerSkill]:
        """
        Load mcp.json and create one MCPServerSkill per server.

        Does NOT connect — skills are lazily activated on first tool use.
        Returns an empty list if the config file doesn't exist or is invalid.
        """
        if not self._config_path.exists():
            logger.debug(f"No MCP config at {self._config_path}")
            return []

        try:
            data = json.loads(
                self._config_path.read_text(encoding="utf-8")
            )
        except Exception as e:
            logger.warning(f"Failed to parse {self._config_path}: {e}")
            return []

        # Support Claude Desktop format ("mcpServers") and flat ("servers")
        raw_servers: dict[str, Any] = data.get(
            "mcpServers", data.get("servers", {})
        )

        if not raw_servers:
            logger.debug("No MCP servers configured")
            return []

        skills: list[MCPServerSkill] = []

        for name, cfg in raw_servers.items():
            if not isinstance(cfg, dict):
                logger.warning(f"MCP server '{name}': invalid config, skipping")
                continue

            # Validate: need either command or url
            command = cfg.get("command", "")
            url = cfg.get("url", "")
            if not command and not url:
                logger.warning(
                    f"MCP server '{name}': no command or url, skipping"
                )
                continue

            skill = MCPServerSkill(
                server_name=name,
                command=command,
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                url=url,
            )
            self._skills[name] = skill
            self._server_configs[name] = cfg
            skills.append(skill)

            transport = "SSE" if url else f"stdio ({command})"
            logger.info(f"MCP server '{name}' configured — {transport}")

        logger.info(
            f"Discovered {len(skills)} MCP server(s) from {self._config_path}"
        )
        return skills

    def server_info(self) -> list[dict[str, Any]]:
        """
        Return status info for all configured servers.

        Useful for ``/mcp`` chat command and ``arc mcp list``.
        """
        info = []
        for name, skill in self._skills.items():
            tool_count = len(skill.manifest().tools)
            cfg = self._server_configs.get(name, {})
            # Build a short hint from the command args (e.g. allowed dirs)
            args = cfg.get("args", [])
            hint = " ".join(str(a) for a in args if not str(a).startswith("-"))
            entry: dict[str, Any] = {
                "name": name,
                "connected": skill.is_connected,
                "transport": skill.transport_type,
                "tools": tool_count,
                "skill_name": f"mcp_{name}",
            }
            if hint:
                entry["hint"] = hint
            info.append(entry)
        return info

    def get_skill(self, name: str) -> MCPServerSkill | None:
        """Get a specific server skill by name."""
        return self._skills.get(name)

    @property
    def server_names(self) -> list[str]:
        """List configured server names."""
        return list(self._skills.keys())

    @property
    def has_servers(self) -> bool:
        """Whether any MCP servers are configured."""
        return bool(self._skills)
