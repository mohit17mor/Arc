"""
Platform interface — the contract for user interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Awaitable

from arc.core.types import Message


# Type for the message handler
MessageHandler = Callable[[str], AsyncIterator[str]]


class Platform(ABC):
    """
    Abstract base class for platforms.

    A platform is a user interface (CLI, API, Telegram, etc.)
    that sends user messages to the agent and displays responses.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Platform identifier."""
        ...

    @abstractmethod
    async def run(self, handler: MessageHandler) -> None:
        """
        Run the platform.

        Args:
            handler: Async function that takes user input and yields response chunks.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the platform."""
        ...