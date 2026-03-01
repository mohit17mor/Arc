"""
Tests for arc.browser.engine — BrowserEngine lifecycle and navigation.

These tests mock Playwright internals so they run without a real browser.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.browser.engine import BrowserEngine


class TestBrowserEngineLifecycle:
    def test_initial_state(self):
        engine = BrowserEngine()
        assert engine.is_launched is False
        assert engine.current_url == ""
        assert engine.last_snapshot is None

    @pytest.mark.asyncio
    async def test_launch_without_playwright_raises(self):
        engine = BrowserEngine()
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            with pytest.raises((RuntimeError, ImportError)):
                await engine.launch()

    @pytest.mark.asyncio
    async def test_ensure_launched_raises_when_not_launched(self):
        engine = BrowserEngine()
        with pytest.raises(RuntimeError, match="not launched"):
            engine._ensure_launched()

    def test_default_user_agent(self):
        engine = BrowserEngine()
        assert "Mozilla" in engine._user_agent

    def test_custom_user_agent(self):
        engine = BrowserEngine(user_agent="TestBot/1.0")
        assert engine._user_agent == "TestBot/1.0"


class TestBrowserEngineWithMock:
    """Tests that mock Playwright to test engine logic."""

    @pytest.fixture
    def engine_and_mocks(self):
        """Create a BrowserEngine with mocked Playwright internals."""
        engine = BrowserEngine(headless=True, profile="test")

        # Helper: create a mock locator (sync return, async methods)
        def make_locator(count=0, inner_text=""):
            loc = MagicMock()
            loc.count = AsyncMock(return_value=count)
            loc.inner_text = AsyncMock(return_value=inner_text)
            return loc

        # Mock the page
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.route = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.go_back = AsyncMock()
        mock_page.go_forward = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=[])  # no elements, then text
        mock_page.screenshot = AsyncMock(return_value=b"PNG_DATA")

        # page.locator() is SYNC in Playwright
        mock_page.locator = MagicMock(side_effect=lambda sel: make_locator(0, ""))

        # Mock context
        mock_context = AsyncMock()
        mock_context.pages = [mock_page]
        mock_context.cookies = AsyncMock(return_value=[])
        mock_context.close = AsyncMock()

        # Manually set engine internals to bypass actual Playwright launch
        engine._page = mock_page
        engine._context = mock_context
        engine._launched = True

        return engine, mock_page, mock_context

    @pytest.mark.asyncio
    async def test_navigate_normalizes_url(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks

        await engine.navigate("example.com")

        # Should have added https://
        mock_page.goto.assert_called_once()
        call_url = mock_page.goto.call_args[0][0]
        assert call_url == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_preserves_https(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks

        await engine.navigate("https://example.com")

        call_url = mock_page.goto.call_args[0][0]
        assert call_url == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_returns_snapshot(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks
        mock_page.title = AsyncMock(return_value="Test Page")

        snapshot = await engine.navigate("https://example.com")

        assert snapshot is not None
        assert snapshot.url == "https://example.com"
        assert engine.last_snapshot is snapshot

    @pytest.mark.asyncio
    async def test_snapshot_caches(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks

        snap1 = await engine.snapshot()
        snap2 = await engine.snapshot()

        # Same URL, not forced — should return cached
        assert snap1 is snap2

    @pytest.mark.asyncio
    async def test_snapshot_force_refresh(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks

        snap1 = await engine.snapshot()
        snap2 = await engine.snapshot(force=True)

        # Forced — should be a new snapshot
        assert snap1 is not snap2

    @pytest.mark.asyncio
    async def test_act_returns_results(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks

        # Set up a last snapshot so elements are available
        await engine.snapshot(force=True)

        result = await engine.act([{"type": "scroll", "direction": "down"}])

        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_go_back(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks

        snapshot = await engine.go_back()

        mock_page.go_back.assert_called_once()
        assert snapshot is not None

    @pytest.mark.asyncio
    async def test_screenshot(self, engine_and_mocks):
        engine, mock_page, _ = engine_and_mocks

        data = await engine.screenshot()
        assert data == b"PNG_DATA"

    @pytest.mark.asyncio
    async def test_get_cookies(self, engine_and_mocks):
        engine, _, mock_context = engine_and_mocks

        cookies = await engine.get_cookies()
        assert cookies == []

    @pytest.mark.asyncio
    async def test_close(self, engine_and_mocks):
        engine, _, mock_context = engine_and_mocks

        await engine.close()

        assert engine.is_launched is False
        assert engine.current_url == ""
        mock_context.close.assert_called_once()
