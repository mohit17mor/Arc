"""
HumanAssist — bridges browser obstacles to the human via EscalationBus.

When the browser encounters an obstacle it can't handle mechanically
(CAPTCHA, login wall, cookie consent requiring choices, bot detection),
this module:

1. Switches the browser to headed (visible) mode
2. Escalates to the user with a clear description of what's needed
3. Waits for the user to resolve it in the visible browser
4. Saves cookies/session to persistent profile
5. Optionally switches back to headless mode

All communication goes through the EscalationBus so the CLI
platform handles the user interaction — this module never touches
the terminal directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc.browser.engine import BrowserEngine
    from arc.browser.snapshot import Obstacle
    from arc.core.escalation import EscalationBus

logger = logging.getLogger(__name__)

# Messages for each obstacle type
_OBSTACLE_MESSAGES = {
    "captcha": (
        "🔐 CAPTCHA detected on the page.\n"
        "A browser window has been opened. Please solve the CAPTCHA, "
        "then type 'done' here when you're finished."
    ),
    "cookie_banner": (
        "🍪 Cookie consent banner detected.\n"
        "A browser window has been opened. Please accept/reject cookies "
        "as you prefer, then type 'done' here."
    ),
    "bot_wall": (
        "🤖 Bot detection wall detected.\n"
        "A browser window has been opened. Please complete the "
        "verification, then type 'done' here."
    ),
    "login_wall": (
        "🔑 Login required to continue.\n"
        "A browser window has been opened. Please log in to the site, "
        "then type 'done' here. Your session will be saved."
    ),
}


class HumanAssist:
    """
    Handles obstacles that require human intervention.

    Works with the EscalationBus to communicate with the user and
    the BrowserEngine to switch display modes.
    """

    def __init__(
        self,
        engine: "BrowserEngine",
        escalation_bus: "EscalationBus | None" = None,
        agent_name: str = "browser-control",
    ) -> None:
        self._engine = engine
        self._escalation_bus = escalation_bus
        self._agent_name = agent_name

    @property
    def can_escalate(self) -> bool:
        """Whether we have an escalation bus to communicate through."""
        return self._escalation_bus is not None

    async def handle_obstacles(
        self, obstacles: list["Obstacle"],
    ) -> list[str]:
        """
        Handle a list of detected obstacles.

        For each obstacle:
        1. Switch to headed mode (if not already)
        2. Ask the user to resolve it
        3. Wait for confirmation

        Returns a list of resolution descriptions.
        """
        if not obstacles:
            return []

        if not self.can_escalate:
            logger.warning(
                "Obstacles detected but no escalation bus available. "
                "Cannot request human assistance."
            )
            return [
                f"⚠ {obs.type}: {obs.description} (no human assist available)"
                for obs in obstacles
            ]

        # Switch to headed mode so the user can see the browser
        await self._engine.switch_to_headed()

        resolutions = []
        for obstacle in obstacles:
            resolution = await self._handle_single(obstacle)
            resolutions.append(resolution)

        # NOTE: We do NOT switch back to headless here.
        # The browser is already launched in headed mode, and switching
        # would close+reopen the browser, losing any state the human
        # just resolved (cookies, session, page state).
        # The browser stays headed for the rest of the session.

        return resolutions

    async def request_help(self, description: str) -> str:
        """
        Generic escalation — ask the human for help with something
        that doesn't fit standard obstacle types.
        """
        if not self.can_escalate:
            return "[No human assist available — proceeding with best effort]"

        await self._engine.switch_to_headed()

        message = (
            f"🖐 Browser needs your help:\n{description}\n"
            "Please handle this in the browser window, "
            "then type 'done' here."
        )

        answer = await self._escalation_bus.ask_manager(
            from_agent=self._agent_name,
            question=message,
        )

        # NOTE: Do NOT switch to headless — browser stays headed.
        # Closing+reopening would lose the page state the human just fixed.
        return f"Human resolved: {answer}"

    async def _handle_single(self, obstacle: "Obstacle") -> str:
        """Handle a single obstacle via escalation."""
        message = _OBSTACLE_MESSAGES.get(
            obstacle.type,
            f"⚠ Obstacle detected: {obstacle.description}\n"
            "A browser window has been opened. Please resolve this "
            "issue, then type 'done' here.",
        )

        # Add the specific description if it differs from the template
        if obstacle.description:
            message += f"\n\nDetails: {obstacle.description}"

        logger.info(f"Escalating obstacle: {obstacle.type} — {obstacle.description}")

        answer = await self._escalation_bus.ask_manager(
            from_agent=self._agent_name,
            question=message,
        )

        logger.info(f"Obstacle {obstacle.type} resolved by human: {answer[:100]}")
        return f"{obstacle.type}: resolved by human"
