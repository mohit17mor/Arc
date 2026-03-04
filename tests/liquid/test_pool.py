"""Tests for BrowserPool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.liquid.pool import BrowserPool, ExtractionResult
from arc.liquid.extract import ProductData


@pytest.fixture
def mock_playwright():
    """Mock Playwright browser and context."""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()
    mock_page.content = AsyncMock(return_value="<html></html>")
    mock_page.evaluate = AsyncMock(return_value=[])
    mock_page.route = AsyncMock()

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mock_pw_instance = AsyncMock()
    mock_pw_instance.chromium = AsyncMock()
    mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw_instance.stop = AsyncMock()

    return mock_pw_instance, mock_browser, mock_context, mock_page


async def test_pool_extract_all_returns_results(mock_playwright):
    """Test that extract_all returns one result per URL."""
    pw_instance, browser, context, page = mock_playwright

    with patch("arc.liquid.pool.extract_products", new_callable=AsyncMock) as mock_extract:

        mock_extract.return_value = [
            ProductData(name="Test Product", price="100", url="https://example.com")
        ]

        pool = BrowserPool(max_concurrent=3)
        pool._playwright = pw_instance
        pool._browser = browser

        results = await pool.extract_all(
            ["https://example.com/1", "https://example.com/2"]
        )
        await pool.close()

    assert len(results) == 2
    for r in results:
        assert isinstance(r, ExtractionResult)
        assert len(r.products) == 1
        assert r.error is None


async def test_pool_handles_extraction_error(mock_playwright):
    """Test that extraction errors are captured, not raised."""
    pw_instance, browser, context, page = mock_playwright

    with patch("arc.liquid.pool.extract_products", new_callable=AsyncMock) as mock_extract:
        mock_extract.side_effect = Exception("Page blocked by CAPTCHA")

        pool = BrowserPool(max_concurrent=2)
        pool._playwright = pw_instance
        pool._browser = browser

        results = await pool.extract_all(["https://example.com/blocked"])
        await pool.close()

    assert len(results) == 1
    assert results[0].error is not None
    assert "CAPTCHA" in results[0].error


async def test_pool_respects_concurrency_limit(mock_playwright):
    """Test that max_concurrent limits parallel extractions."""
    pw_instance, browser, context, page = mock_playwright

    # Track concurrent calls
    import asyncio
    concurrent = 0
    max_seen = 0

    original_new_context = browser.new_context

    async def tracked_new_context(**kwargs):
        nonlocal concurrent, max_seen
        concurrent += 1
        max_seen = max(max_seen, concurrent)
        result = await original_new_context(**kwargs)
        return result

    browser.new_context = tracked_new_context

    with patch("arc.liquid.pool.extract_products", new_callable=AsyncMock) as mock_extract:
        async def slow_extract(page, url):
            nonlocal concurrent
            await asyncio.sleep(0.1)
            concurrent -= 1
            return [ProductData(name="P")]

        mock_extract.side_effect = slow_extract

        pool = BrowserPool(max_concurrent=2)
        pool._playwright = pw_instance
        pool._browser = browser

        urls = [f"https://example.com/{i}" for i in range(5)]
        await pool.extract_all(urls)
        await pool.close()

    assert max_seen <= 2


async def test_pool_close():
    """Test that close() cleans up browser and playwright."""
    mock_browser = AsyncMock()
    mock_pw = AsyncMock()

    pool = BrowserPool()
    pool._browser = mock_browser
    pool._playwright = mock_pw

    await pool.close()

    mock_browser.close.assert_called_once()
    mock_pw.stop.assert_called_once()
    assert pool._browser is None
    assert pool._playwright is None
