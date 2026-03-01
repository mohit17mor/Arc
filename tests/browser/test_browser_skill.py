"""
Tests for arc.skills.builtin.browser_control — BrowserControlSkill.

Tests the skill interface, tool dispatch, and tool output formatting
without a real browser (BrowserEngine is mocked).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.browser.actions import ActionResult, ActionsResult
from arc.browser.snapshot import InteractiveElement, Obstacle, PageSnapshot
from arc.skills.builtin.browser_control import BrowserControlSkill


def _make_snapshot(
    url="https://example.com",
    title="Example",
    elements=None,
    obstacles=None,
) -> PageSnapshot:
    return PageSnapshot(
        url=url,
        title=title,
        page_type="other",
        elements=elements or [],
        obstacles=obstacles or [],
        text_content="Page content here",
        forms_count=0,
        links_count=0,
    )


class TestBrowserControlSkillManifest:
    def test_manifest_name(self):
        skill = BrowserControlSkill()
        m = skill.manifest()
        assert m.name == "browser_control"

    def test_manifest_has_three_tools(self):
        skill = BrowserControlSkill()
        m = skill.manifest()
        tool_names = {t.name for t in m.tools}
        assert tool_names == {"browser_go", "browser_look", "browser_act"}

    def test_manifest_capabilities(self):
        from arc.core.types import Capability
        skill = BrowserControlSkill()
        m = skill.manifest()
        assert Capability.BROWSER in m.capabilities


class TestBrowserControlSkillExecution:
    @pytest.fixture
    def skill(self):
        """Create a BrowserControlSkill with mocked engine."""
        skill = BrowserControlSkill()

        # Mock the engine directly
        mock_engine = AsyncMock()
        mock_engine.current_url = "https://example.com"
        mock_engine.is_launched = True
        mock_engine.last_snapshot = _make_snapshot()
        mock_engine.navigate = AsyncMock(return_value=_make_snapshot())
        mock_engine.snapshot = AsyncMock(return_value=_make_snapshot())
        mock_engine.act = AsyncMock(return_value=ActionsResult(
            results=[ActionResult(success=True, action_type="click", target="Login")],
            snapshot=_make_snapshot(),
        ))

        mock_human = AsyncMock()
        mock_human.can_escalate = False
        mock_human.handle_obstacles = AsyncMock(return_value=[])

        skill._engine = mock_engine
        skill._human = mock_human
        skill._activated = True

        return skill

    @pytest.mark.asyncio
    async def test_browser_go(self, skill):
        result = await skill.execute_tool("browser_go", {"url": "https://example.com"})

        assert result.success is True
        assert "example.com" in result.output.lower()

    @pytest.mark.asyncio
    async def test_browser_go_no_url(self, skill):
        result = await skill.execute_tool("browser_go", {"url": ""})

        assert result.success is False
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_browser_look(self, skill):
        result = await skill.execute_tool("browser_look", {})

        assert result.success is True
        assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_browser_look_no_page(self, skill):
        skill._engine.current_url = ""

        result = await skill.execute_tool("browser_look", {})

        assert result.success is False
        assert "no page" in result.error.lower()

    @pytest.mark.asyncio
    async def test_browser_look_with_extract(self, skill):
        result = await skill.execute_tool("browser_look", {"extract": "prices"})

        assert result.success is True
        assert "prices" in result.output.lower()

    @pytest.mark.asyncio
    async def test_browser_act(self, skill):
        result = await skill.execute_tool("browser_act", {
            "actions": [{"type": "click", "target": "Login"}],
        })

        assert result.success is True
        assert "Login" in result.output

    @pytest.mark.asyncio
    async def test_browser_act_no_actions(self, skill):
        result = await skill.execute_tool("browser_act", {"actions": []})

        assert result.success is False

    @pytest.mark.asyncio
    async def test_browser_act_no_page(self, skill):
        skill._engine.current_url = ""

        result = await skill.execute_tool("browser_act", {
            "actions": [{"type": "click", "target": "Submit"}],
        })

        assert result.success is False

    @pytest.mark.asyncio
    async def test_unknown_tool(self, skill):
        result = await skill.execute_tool("browser_dance", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_engine_error_handled(self, skill):
        skill._engine.navigate = AsyncMock(side_effect=Exception("Connection refused"))

        result = await skill.execute_tool("browser_go", {"url": "https://down.com"})

        assert result.success is False
        assert "Connection refused" in result.error


class TestBrowserControlSkillObstacles:
    @pytest.mark.asyncio
    async def test_obstacles_trigger_human_assist(self):
        skill = BrowserControlSkill()

        snapshot_with_captcha = _make_snapshot(
            obstacles=[Obstacle(type="captcha", description="reCAPTCHA")],
        )
        snapshot_clean = _make_snapshot()

        mock_engine = AsyncMock()
        mock_engine.current_url = "https://example.com"
        mock_engine.is_launched = True
        mock_engine.navigate = AsyncMock(return_value=snapshot_with_captcha)
        mock_engine.snapshot = AsyncMock(return_value=snapshot_clean)

        mock_human = AsyncMock()
        mock_human.can_escalate = True
        mock_human.handle_obstacles = AsyncMock(return_value=["captcha: resolved by human"])

        skill._engine = mock_engine
        skill._human = mock_human
        skill._activated = True

        result = await skill.execute_tool("browser_go", {"url": "https://captcha-site.com"})

        # Should have called human assist
        mock_human.handle_obstacles.assert_called_once()
        assert result.success is True
