"""Tests for MCP config management skill."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc.skills.builtin.mcp_config import MCPConfigSkill


@pytest.mark.asyncio
async def test_mcp_config_skill_returns_current_config_text():
    skill = MCPConfigSkill()
    await skill.initialize(None, {})

    service = SimpleNamespace(
        inspect=lambda: SimpleNamespace(
            text='{"mcpServers":{}}',
            valid=True,
            errors=[],
            path="/tmp/mcp.json",
        ),
    )
    skill.set_dependencies(config_service=service)

    result = await skill.execute_tool("mcp_get_config", {})

    assert result.success is True
    assert '"mcpServers"' in result.output


@pytest.mark.asyncio
async def test_mcp_config_skill_applies_new_config():
    skill = MCPConfigSkill()
    await skill.initialize(None, {})

    async def save_and_reload(text: str):
        assert '"github"' in text
        return SimpleNamespace(
            applied=True,
            valid=True,
            errors=[],
            active_server_names=["github"],
        )

    skill.set_dependencies(
        config_service=SimpleNamespace(
            inspect=lambda: SimpleNamespace(
                text='{"mcpServers":{}}',
                valid=True,
                errors=[],
                path="/tmp/mcp.json",
            ),
            save_and_reload=save_and_reload,
        )
    )

    result = await skill.execute_tool(
        "mcp_set_config",
        {"text": '{"mcpServers":{"github":{"command":"npx"}}}'},
    )

    assert result.success is True
    assert "github" in result.output


@pytest.mark.asyncio
async def test_mcp_config_skill_reports_validation_errors():
    skill = MCPConfigSkill()
    await skill.initialize(None, {})

    async def save_and_reload(text: str):
        return SimpleNamespace(
            applied=False,
            valid=False,
            errors=["Server 'broken' must define command or url."],
            active_server_names=["github"],
        )

    skill.set_dependencies(
        config_service=SimpleNamespace(
            inspect=lambda: SimpleNamespace(
                text='{"mcpServers":{}}',
                valid=True,
                errors=[],
                path="/tmp/mcp.json",
            ),
            save_and_reload=save_and_reload,
        )
    )

    result = await skill.execute_tool(
        "mcp_set_config",
        {"text": '{"mcpServers":{"broken":{}}}'},
    )

    assert result.success is False
    assert "broken" in result.error
