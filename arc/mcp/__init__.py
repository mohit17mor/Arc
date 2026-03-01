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

from arc.mcp.client import MCPClient
from arc.mcp.gateway import MCPGatewaySkill
from arc.mcp.manager import MCPManager
from arc.mcp.skill import MCPServerSkill

__all__ = ["MCPClient", "MCPGatewaySkill", "MCPManager", "MCPServerSkill"]
