"""
VirtualPlatform — silent platform for background agents.

Workers and expert agents run on VirtualPlatform instead of CLIPlatform so
that all their output is captured to an in-memory buffer and never written
directly to the terminal.

Only the main agent's CLIPlatform writes to the user.

Usage (expert agent)::

    platform = VirtualPlatform(name="research")
    task = asyncio.create_task(platform.run(agent.run))

    # Route a message and get back the full response
    response = await platform.send_message("summarise AI news today")

Usage (one-shot worker)::

    platform = VirtualPlatform(name="worker-abc")
    # Workers typically bypass the platform and call agent.run() directly;
    # VirtualPlatform is used when you want the platform.run() lifecycle.
"""

from __future__ import annotations

import asyncio
import logging

from arc.platforms.base import Platform, MessageHandler

logger = logging.getLogger(__name__)


class VirtualPlatform(Platform):
    """
    Silent, in-memory platform for non-interactive agents.

    - Accepts messages programmatically via ``send_message()``
    - Collects response chunks silently (never touches the terminal)
    - Supports sequential message turns (one in-flight message at a time)
    - Safe to stop/restart

    Not concurrent-safe for multiple simultaneous ``send_message`` callers —
    expert agents receive one message at a time from the main agent.
    """

    def __init__(self, name: str = "virtual") -> None:
        self._name_str = name
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._running = False
        self._current_buffer: list[str] = []
        self._response_ready = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Platform ABC                                                         #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self._name_str

    async def run(self, handler: MessageHandler) -> None:
        """
        Main loop — reads from the input queue and feeds messages to the handler.

        Runs until ``stop()`` is called or a ``None`` sentinel is put in the queue.
        """
        self._running = True
        logger.debug(f"VirtualPlatform '{self._name_str}' started")

        while self._running:
            try:
                message = await self._input_queue.get()
            except asyncio.CancelledError:
                break

            if message is None:  # stop sentinel
                break

            # Collect response chunks silently
            self._current_buffer = []
            try:
                async for chunk in handler(message):
                    self._current_buffer.append(chunk)
            except asyncio.CancelledError:
                logger.debug(
                    f"VirtualPlatform '{self._name_str}': handler cancelled mid-response"
                )
                break
            except Exception as exc:
                logger.error(
                    f"VirtualPlatform '{self._name_str}': handler error: {exc}",
                    exc_info=True,
                )
                self._current_buffer.append(f"[Error: {exc}]")
            finally:
                # Always signal that a response (possibly partial) is ready
                self._response_ready.set()

        self._running = False
        logger.debug(f"VirtualPlatform '{self._name_str}' stopped")

    async def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._running = False
        await self._input_queue.put(None)

    # ------------------------------------------------------------------ #
    # Programmatic message routing                                         #
    # ------------------------------------------------------------------ #

    async def send_message(self, text: str) -> str:
        """
        Send a message to the agent and wait for the full response.

        Returns the complete response as a single string.
        Raises ``RuntimeError`` if the platform is not running.
        """
        self._response_ready.clear()
        self._current_buffer = []
        await self._input_queue.put(text)
        await self._response_ready.wait()
        return "".join(self._current_buffer)

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._running

    def get_last_output(self) -> str:
        """Return the last collected response (useful for testing)."""
        return "".join(self._current_buffer)
