"""
Context Composer — builds the messages list for LLM calls.

Handles token budget management and message truncation.
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from arc.core.types import ComposedContext, Message
from arc.memory.session import SessionMemory


class ContextComposer:
    """
    Composes the context (messages list) for each LLM call.

    Manages token budget and ensures we don't exceed model limits.

    Usage:
        composer = ContextComposer(
            token_counter=llm.count_tokens,
            max_tokens=100000,
            reserve_output=8000,
        )

        context = await composer.compose(
            session=session_memory,
            recent_window=20,
        )
    """

    def __init__(
        self,
        token_counter: Callable[[list[Message]], Awaitable[int]],
        max_tokens: int = 128000,
        reserve_output: int = 8192,
    ) -> None:
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._reserve_output = reserve_output

    @property
    def token_budget(self) -> int:
        """Available tokens for input context."""
        return self._max_tokens - self._reserve_output

    async def compose(
        self,
        session: SessionMemory,
        recent_window: int = 20,
    ) -> ComposedContext:
        """
        Compose context from session memory.

        Takes the most recent messages that fit within token budget.
        """
        # Start with system prompt
        messages = session.get_messages(include_system=True)

        # If within budget, use all messages
        token_count = await self._token_counter(messages)
        if token_count <= self.token_budget:
            return ComposedContext(
                messages=messages,
                token_count=token_count,
                token_budget=self.token_budget,
                breakdown={"all": token_count},
            )

        # Need to truncate — keep system + most recent
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        # Binary search for how many recent messages fit
        # Start with recent_window, reduce if needed
        window = min(recent_window, len(other_msgs))

        while window > 0:
            candidate = system_msgs + other_msgs[-window:]
            token_count = await self._token_counter(candidate)
            if token_count <= self.token_budget:
                return ComposedContext(
                    messages=candidate,
                    token_count=token_count,
                    token_budget=self.token_budget,
                    breakdown={
                        "system": len(system_msgs),
                        "recent": window,
                        "truncated": len(other_msgs) - window,
                    },
                )
            window -= 1

        # Worst case: only system prompt
        return ComposedContext(
            messages=system_msgs,
            token_count=await self._token_counter(system_msgs),
            token_budget=self.token_budget,
            breakdown={"system": len(system_msgs), "truncated": len(other_msgs)},
        )