"""
BrowserPool — parallel headless extraction with concurrency control.

Shares a single Chromium browser with multiple contexts for efficient
parallel page scraping.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from arc.liquid.extract import ProductData, extract_products

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from extracting a single URL."""

    url: str
    products: list[ProductData] = field(default_factory=list)
    error: str | None = None


class BrowserPool:
    """
    Manages parallel headless browser extraction.

    Uses a single Chromium browser with multiple isolated contexts,
    gated by a semaphore for concurrency control.
    """

    def __init__(self, max_concurrent: int = 5, timeout_per_url: float = 30.0):
        self._max_concurrent = max_concurrent
        self._timeout_per_url = timeout_per_url
        self._browser = None
        self._playwright = None

    async def _ensure_browser(self):
        """Launch browser if not already running."""
        if self._browser is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            logger.info("BrowserPool: launched headless Chromium")

    async def extract_all(
        self,
        urls: list[str],
        *,
        overall_timeout: float = 60.0,
    ) -> list[ExtractionResult]:
        """
        Extract products from multiple URLs in parallel.

        Args:
            urls: List of URLs to scrape.
            overall_timeout: Max time for entire batch.

        Returns:
            List of ExtractionResult, one per URL.
        """
        await self._ensure_browser()
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _extract_one(url: str) -> ExtractionResult:
            async with semaphore:
                context = None
                try:
                    context = await self._browser.new_context(
                        user_agent=(
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/131.0.0.0 Safari/537.36"
                        ),
                        viewport={"width": 1280, "height": 720},
                        locale="en-US",
                    )
                    page = await context.new_page()

                    # Block only truly heavy resources (fonts, video, media)
                    # DO NOT block images — img.src and img.alt are critical
                    # for product extraction (especially on Amazon)
                    await page.route(
                        "**/*.{woff,woff2,ttf,eot,mp4,webm,ogg,mp3,wav}",
                        lambda route: route.abort(),
                    )

                    await page.goto(
                        url,
                        timeout=int(self._timeout_per_url * 1000),
                        wait_until="domcontentloaded",
                    )

                    # Wait for dynamic content to render
                    await _smart_wait(page, url)

                    products = await extract_products(page, url)
                    return ExtractionResult(url=url, products=products)

                except Exception as e:
                    logger.warning("Extraction failed for %s: %s", url, e)
                    return ExtractionResult(url=url, error=str(e))

                finally:
                    if context:
                        await context.close()

        # Run all extractions with overall timeout
        tasks = [asyncio.create_task(_extract_one(u)) for u in urls]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=overall_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("BrowserPool: overall timeout (%.0fs) reached", overall_timeout)
            # Collect whatever completed
            results = []
            for task in tasks:
                if task.done() and not task.cancelled():
                    results.append(task.result())
                else:
                    task.cancel()

        # Normalize: exceptions become error results
        final: list[ExtractionResult] = []
        for i, r in enumerate(results):
            if isinstance(r, ExtractionResult):
                final.append(r)
            elif isinstance(r, Exception):
                final.append(ExtractionResult(url=urls[i], error=str(r)))
        return final

    async def close(self):
        """Shut down browser and playwright."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("BrowserPool: closed")


async def _smart_wait(page, url: str) -> None:
    """
    Wait for dynamic content to render based on the site.

    Amazon, Flipkart, and many e-commerce sites load search results
    via JavaScript *after* the initial DOM. A plain domcontentloaded
    is not enough — we need to wait for the product cards to appear.
    """
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.removeprefix("www.")

    try:
        if "amazon" in domain:
            # Wait for Amazon search result cards or product title
            await page.wait_for_selector(
                '[data-component-type="s-search-result"], #productTitle',
                timeout=12000,
            )
            # Give a bit extra for images to load (needed for img.alt names)
            await asyncio.sleep(1.5)
        elif "flipkart" in domain:
            # Flipkart product cards
            await page.wait_for_selector(
                '[data-id], a[href*="/p/"]',
                timeout=12000,
            )
            await asyncio.sleep(1.0)
        else:
            # Generic: try networkidle, fall back to a short sleep
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                await asyncio.sleep(2.0)
    except Exception as e:
        logger.debug("Smart wait failed for %s: %s (continuing anyway)", url, e)
        # Still try to extract from whatever is on the page
        await asyncio.sleep(1.0)
