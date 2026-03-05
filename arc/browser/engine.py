"""
BrowserEngine — manages a Playwright browser instance and orchestrates
page analysis + action execution.

Responsibilities:
  - Launch / close the browser (headless by default)
  - Navigate to URLs with smart wait strategies
  - Take structured page snapshots via PageAnalyzer
  - Execute action batches via ActionExecutor
  - Manage persistent browser profiles for cookies/sessions
  - Switch between headless and headed mode for human-assist

Design principle: the engine is *thick* — it handles all mechanical
browser operations without needing the LLM.  The LLM only sees
text snapshots and decides *what* to do next.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

from arc.browser.actions import ActionExecutor, ActionsResult
from arc.browser.snapshot import PageAnalyzer, PageSnapshot

logger = logging.getLogger(__name__)

# Default dirs
_PROFILES_DIR = Path.home() / ".arc" / "browser" / "profiles"

# Navigation timeouts (ms)
_NAV_TIMEOUT = 15_000
_STABLE_TIMEOUT = 1_000


class BrowserEngine:
    """
    Manages the lifecycle of a Playwright browser and provides
    high-level navigation and interaction primitives.

    Usage::

        engine = BrowserEngine()
        await engine.launch()
        snapshot = await engine.navigate("https://example.com")
        result = await engine.act([{"type": "click", "target": "Login"}])
        await engine.close()
    """

    def __init__(
        self,
        headless: bool = True,
        profile: str = "default",
        user_agent: str | None = None,
    ) -> None:
        self._headless = headless
        self._profile = profile
        self._user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
        self._playwright: "Playwright | None" = None
        self._browser: "Browser | None" = None
        self._context: "BrowserContext | None" = None
        self._page: "Page | None" = None
        self._analyzer = PageAnalyzer()
        self._executor = ActionExecutor(self._analyzer)
        self._launched = False
        self._last_snapshot: PageSnapshot | None = None

    # ━━━ Properties ━━━

    @property
    def is_launched(self) -> bool:
        return self._launched and self._page is not None

    @property
    def current_url(self) -> str:
        if self._page:
            return self._page.url
        return ""

    @property
    def last_snapshot(self) -> PageSnapshot | None:
        return self._last_snapshot

    @property
    def page(self) -> "Page | None":
        return self._page

    # ━━━ Lifecycle ━━━

    async def launch(self) -> None:
        """Launch the browser. Installs Playwright browsers if needed."""
        if self._launched:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. "
                "Install it with: pip install 'arc-agent[browsing]'"
            )

        self._playwright = await async_playwright().start()

        # Ensure profile storage directory exists
        profile_dir = _PROFILES_DIR / self._profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Launch Chromium with persistent context for cookies/sessions
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=self._headless,
            user_agent=self._user_agent,
            viewport={"width": 1280, "height": 720},
            locale="en-US",
            timezone_id="America/Los_Angeles",
            # Block unnecessary resource types for speed
            # (images are blocked — we do accessibility-tree-first)
            java_script_enabled=True,
            bypass_csp=True,
            ignore_https_errors=True,
        )

        # Use the first page or create one
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()

        # Block heavy resources to speed things up
        await self._page.route(
            "**/*.{png,jpg,jpeg,gif,svg,webp,ico,woff,woff2,ttf,eot,mp4,webm}",
            lambda route: route.abort(),
        )

        # Track new tabs/popups — auto-switch to them
        self._context.on("page", self._on_new_page)

        self._launched = True
        logger.info(
            f"Browser launched (headless={self._headless}, "
            f"profile={self._profile})"
        )

    async def close(self) -> None:
        """Close the browser and clean up resources."""
        if self._context:
            try:
                await self._context.close()
            except Exception as e:
                logger.debug(f"Error closing context: {e}")
            self._context = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.debug(f"Error stopping playwright: {e}")
            self._playwright = None

        self._page = None
        self._browser = None
        self._launched = False
        self._last_snapshot = None
        logger.info("Browser closed")

    async def switch_to_headed(self) -> None:
        """
        Restart the browser in headed (visible) mode.

        Used when human assistance is needed (CAPTCHAs, login, etc.).
        The persistent profile preserves cookies across the switch.
        """
        if not self._headless:
            return  # Already headed

        current_url = self.current_url
        await self.close()

        self._headless = False
        await self.launch()

        if current_url and current_url != "about:blank":
            await self._navigate_raw(current_url)

        logger.info("Switched to headed mode for human assistance")

    async def switch_to_headless(self) -> None:
        """Switch back to headless mode after human assistance."""
        if self._headless:
            return  # Already headless

        current_url = self.current_url
        await self.close()

        self._headless = True
        await self.launch()

        if current_url and current_url != "about:blank":
            await self._navigate_raw(current_url)

        logger.info("Switched back to headless mode")

    # ━━━ Navigation ━━━

    async def navigate(self, url: str) -> PageSnapshot:
        """
        Navigate to a URL and return a structured page snapshot.

        Handles common URL patterns — adds https:// if missing,
        waits for the page to be stable before analyzing.
        """
        self._ensure_launched()

        # Normalize URL
        if not url.startswith(("http://", "https://", "file://")):
            url = f"https://{url}"

        await self._navigate_raw(url)

        # Take snapshot
        snapshot = await self._analyzer.analyze(self._page)
        self._last_snapshot = snapshot

        logger.info(
            f"Navigated to {url} — "
            f"{snapshot.forms_count} forms, "
            f"{snapshot.links_count} links, "
            f"{len(snapshot.obstacles)} obstacles"
        )

        return snapshot

    async def _navigate_raw(self, url: str) -> None:
        """Navigate without snapshotting."""
        try:
            await self._page.goto(url, timeout=_NAV_TIMEOUT, wait_until="domcontentloaded")
        except Exception as e:
            logger.warning(f"Navigation to {url} had issues: {e}")
            # Page may still be usable even if timeout hit

        # Wait for page to settle
        try:
            await self._page.wait_for_load_state("networkidle", timeout=_STABLE_TIMEOUT)
        except Exception:
            pass  # Pages with persistent connections never reach networkidle

    # ━━━ Page Analysis ━━━

    async def snapshot(self, force: bool = False) -> PageSnapshot:
        """
        Take a fresh snapshot of the current page.

        Args:
            force: If True, always re-analyze even if URL hasn't changed
        """
        self._ensure_launched()

        if not force and self._last_snapshot and self._last_snapshot.url == self._page.url:
            return self._last_snapshot

        snapshot = await self._analyzer.analyze(self._page)
        self._last_snapshot = snapshot
        return snapshot

    # ━━━ Action Execution ━━━

    async def act(self, actions: list[dict[str, Any]]) -> ActionsResult:
        """
        Execute a sequence of actions on the current page.

        Each action is a dict with at least a ``type`` key.
        See ``ActionExecutor`` for supported action types.

        Returns an ``ActionsResult`` with per-action results and
        a fresh page snapshot afterwards.
        """
        self._ensure_launched()

        pages_before = len(self._context.pages)

        # Get current elements for fuzzy matching
        elements = []
        if self._last_snapshot:
            elements = self._last_snapshot.elements

        result = await self._executor.execute(self._page, actions, elements)

        # If a new tab was opened, switch to it
        if len(self._context.pages) > pages_before:
            new_page = self._context.pages[-1]
            logger.info(f"New tab detected — switching to {new_page.url}")
            self._page = new_page
            self._last_snapshot = None
            # Re-snapshot from the new page
            try:
                await self._wait_stable()
                result.snapshot = await self._analyzer.analyze(self._page)
                self._last_snapshot = result.snapshot
            except Exception as e:
                logger.warning(f"Failed to snapshot new tab: {e}")
        elif result.snapshot:
            self._last_snapshot = result.snapshot

        return result

    # ━━━ Utility ━━━

    async def go_back(self) -> PageSnapshot:
        """Navigate back and return a snapshot."""
        self._ensure_launched()
        await self._page.go_back(timeout=_NAV_TIMEOUT)
        await self._wait_stable()
        snapshot = await self._analyzer.analyze(self._page)
        self._last_snapshot = snapshot
        return snapshot

    async def go_forward(self) -> PageSnapshot:
        """Navigate forward and return a snapshot."""
        self._ensure_launched()
        await self._page.go_forward(timeout=_NAV_TIMEOUT)
        await self._wait_stable()
        snapshot = await self._analyzer.analyze(self._page)
        self._last_snapshot = snapshot
        return snapshot

    async def evaluate_js(self, code: str) -> Any:
        """Run arbitrary JavaScript on the page."""
        self._ensure_launched()
        return await self._page.evaluate(code)

    async def screenshot(self, path: str | None = None) -> bytes:
        """Take a screenshot of the current page. Returns PNG bytes."""
        self._ensure_launched()
        return await self._page.screenshot(path=path, full_page=False)

    async def get_cookies(self) -> list[dict]:
        """Get all cookies for the current context."""
        self._ensure_launched()
        return await self._context.cookies()

    def _on_new_page(self, page: "Page") -> None:
        """Handle new tab/popup opened by a click."""
        logger.info(f"New page/tab opened: {page.url}")

    async def _wait_stable(self) -> None:
        """Wait for page to settle."""
        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=_STABLE_TIMEOUT)
        except Exception:
            pass
        try:
            await self._page.wait_for_load_state("networkidle", timeout=1000)
        except Exception:
            pass

    def _ensure_launched(self) -> None:
        """Raise if the browser hasn't been launched."""
        if not self._launched or not self._page:
            raise RuntimeError(
                "Browser not launched. Call `await engine.launch()` first."
            )
