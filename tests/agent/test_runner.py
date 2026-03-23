"""Tests for the shared virtual-platform runner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from arc.agent.runner import run_agent_on_virtual_platform


class FakeVirtualPlatform:
    def __init__(self, name: str = "virtual") -> None:
        self.name = name
        self.run = AsyncMock()
        self.send_message = AsyncMock()
        self.stop = AsyncMock()


@pytest.mark.asyncio
class TestRunAgentOnVirtualPlatform:
    async def test_returns_content_on_success(self):
        platform = FakeVirtualPlatform("job:test")
        platform.send_message.return_value = "result text"

        with patch("arc.platforms.virtual.app.VirtualPlatform", return_value=platform):
            content, error = await run_agent_on_virtual_platform(
                agent=Mock(run=Mock()),
                prompt="hello",
                name="job:test",
                timeout_seconds=2.0,
            )

        assert content == "result text"
        assert error is None
        platform.send_message.assert_awaited_once_with("hello")
        platform.stop.assert_awaited_once()

    async def test_returns_timeout_error_and_cancels_task(self):
        platform = FakeVirtualPlatform("job:timeout")

        async def slow_send(prompt: str):
            await asyncio.sleep(0.05)
            return "never reached"

        platform.send_message.side_effect = slow_send

        with patch("arc.platforms.virtual.app.VirtualPlatform", return_value=platform):
            content, error = await run_agent_on_virtual_platform(
                agent=Mock(run=Mock()),
                prompt="hello",
                name="job:timeout",
                timeout_seconds=0.001,
            )

        assert content == ""
        assert error == "Timed out after 0s"
        platform.stop.assert_not_awaited()

    async def test_returns_exception_text_when_send_fails(self):
        platform = FakeVirtualPlatform("job:error")
        platform.send_message.side_effect = RuntimeError("network down")

        with patch("arc.platforms.virtual.app.VirtualPlatform", return_value=platform):
            content, error = await run_agent_on_virtual_platform(
                agent=Mock(run=Mock()),
                prompt="hello",
                name="job:error",
                timeout_seconds=2.0,
            )

        assert content == ""
        assert error == "network down"
        platform.stop.assert_not_awaited()
