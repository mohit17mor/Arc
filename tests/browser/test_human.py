"""Tests for human-assisted browser escalation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from arc.browser.human import HumanAssist


def _obstacle(kind: str, description: str) -> SimpleNamespace:
    return SimpleNamespace(type=kind, description=description)


@pytest.mark.asyncio
class TestHumanAssist:
    async def test_handle_obstacles_returns_empty_for_no_obstacles(self):
        engine = AsyncMock()
        assist = HumanAssist(engine=engine, escalation_bus=AsyncMock())

        result = await assist.handle_obstacles([])

        assert result == []
        engine.switch_to_headed.assert_not_awaited()

    async def test_handle_obstacles_without_escalation_bus_returns_best_effort_notes(self):
        engine = AsyncMock()
        assist = HumanAssist(engine=engine, escalation_bus=None)
        obstacles = [_obstacle("captcha", "Solve captcha"), _obstacle("login_wall", "Log in")]

        with patch("arc.browser.human.logger.warning") as warning:
            result = await assist.handle_obstacles(obstacles)

        assert len(result) == 2
        assert "no human assist available" in result[0]
        engine.switch_to_headed.assert_not_awaited()
        warning.assert_called_once()

    async def test_handle_obstacles_switches_to_headed_once_and_handles_each_obstacle(self):
        engine = AsyncMock()
        bus = AsyncMock()
        assist = HumanAssist(engine=engine, escalation_bus=bus)
        assist._handle_single = AsyncMock(side_effect=["captcha: resolved by human", "login_wall: resolved by human"])
        obstacles = [_obstacle("captcha", "Solve captcha"), _obstacle("login_wall", "Log in")]

        result = await assist.handle_obstacles(obstacles)

        assert result == ["captcha: resolved by human", "login_wall: resolved by human"]
        engine.switch_to_headed.assert_awaited_once()
        assert assist._handle_single.await_count == 2

    async def test_request_help_escalates_generic_message(self):
        engine = AsyncMock()
        bus = AsyncMock()
        bus.ask_manager.return_value = "done"
        assist = HumanAssist(engine=engine, escalation_bus=bus, agent_name="browser-agent")

        result = await assist.request_help("Choose the right cookie option")

        assert result == "Human resolved: done"
        engine.switch_to_headed.assert_awaited_once()
        bus.ask_manager.assert_awaited_once()
        kwargs = bus.ask_manager.await_args.kwargs
        assert kwargs["from_agent"] == "browser-agent"
        assert "Choose the right cookie option" in kwargs["question"]

    async def test_request_help_without_bus_returns_fallback_message(self):
        assist = HumanAssist(engine=AsyncMock(), escalation_bus=None)

        result = await assist.request_help("Need help")

        assert "No human assist available" in result

    async def test_handle_single_uses_template_and_includes_details(self):
        engine = AsyncMock()
        bus = AsyncMock()
        bus.ask_manager.return_value = "done"
        assist = HumanAssist(engine=engine, escalation_bus=bus, agent_name="browser-agent")

        result = await assist._handle_single(_obstacle("captcha", "Cloudflare page"))

        assert result == "captcha: resolved by human"
        question = bus.ask_manager.await_args.kwargs["question"]
        assert "CAPTCHA detected" in question
        assert "Cloudflare page" in question

    async def test_handle_single_uses_generic_template_for_unknown_obstacle(self):
        engine = AsyncMock()
        bus = AsyncMock()
        bus.ask_manager.return_value = "fixed"
        assist = HumanAssist(engine=engine, escalation_bus=bus)

        result = await assist._handle_single(_obstacle("weird_wall", "A strange blocker"))

        assert result == "weird_wall: resolved by human"
        question = bus.ask_manager.await_args.kwargs["question"]
        assert "Obstacle detected" in question
        assert "A strange blocker" in question
