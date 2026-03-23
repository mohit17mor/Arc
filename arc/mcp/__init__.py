"""
MCP (Model Context Protocol) integration for Arc.

Arc acts as an MCP **client** — it connects to external MCP servers
and exposes their tools as native Arc skills.  Users configure servers
in ``~/.arc/mcp.json`` (Claude Desktop-compatible format) and the tools
appear automatically in ``arc chat``.

Modules:
    client  — Low-level MCP connection (stdio / SSE transport)
    skill   — Wraps one MCP server as an Arc Skill
    manager — Discovers configured servers, creates MCPServerSkills
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arc.mcp.client import MCPClient
    from arc.mcp.gateway import MCPGatewaySkill
    from arc.mcp.manager import MCPManager
    from arc.mcp.skill import MCPServerSkill

__all__ = ["MCPClient", "MCPGatewaySkill", "MCPManager", "MCPServerSkill"]

_EXPORTS = {
    "MCPClient": "arc.mcp.client",
    "MCPGatewaySkill": "arc.mcp.gateway",
    "MCPManager": "arc.mcp.manager",
    "MCPServerSkill": "arc.mcp.skill",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
