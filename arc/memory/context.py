"""
Context Composer — builds the messages list for LLM calls.

Handles token budget management across all three memory tiers:

    Tier 3 — Core facts (always in system prompt, tiny, ~4k tokens max)
    Tier 2 — Relevant episodic memories (retrieved async, ~8k tokens)
    Tier 1 — Recent session turns (token-capped, not turn-capped)

The composer asks for Tier 2 memories in parallel with its own work
so the retrieval latency is hidden. The user never waits for it.

Token budget layout (128k context example):
    reserve_output      8k   — room for LLM to generate its reply
    tier3_budget        4k   — all core facts
    tier2_budget        8k   — episodic retrieval results
    tools               ~3k  — tool specs (managed by AgentLoop)
    tier1 (session)    ~60k  — as many recent turns as fit
    headroom           ~45k  — remaining, caught by token checker
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from arc.core.types import ComposedContext, Message
from arc.memory.session import SessionMemory

if TYPE_CHECKING:
    from arc.memory.manager import MemoryManager

# Hard token budgets for long-term memory tiers
TIER3_TOKEN_BUDGET = 4_000   # core facts — always present
TIER2_TOKEN_BUDGET = 8_000   # episodic retrieval results


class ContextComposer:
    """
    Composes the context (messages list) for each LLM call.

    Manages token budget and ensures we don't exceed model limits.
    Optionally enriches context with long-term memory (Tiers 2 and 3).

    Usage (no memory):
        composer = ContextComposer(token_counter=llm.count_tokens)
        context = await composer.compose(session=session_memory)

    Usage (with memory):
        composer = ContextComposer(token_counter=llm.count_tokens)
        context = await composer.compose(
            session=session_memory,
            query="current user message",
            memory_manager=memory_manager,
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
        """Total available tokens for input context."""
        return self._max_tokens - self._reserve_output

    async def compose(
        self,
        session: SessionMemory,
        recent_window: int = 20,
        query: str = "",
        memory_manager: MemoryManager | None = None,
    ) -> ComposedContext:
        """
        Compose context from all memory tiers.

        When memory_manager is provided, Tier 3 (core facts) is injected
        into the system prompt and Tier 2 (relevant episodic memories) is
        retrieved in parallel and appended to the system prompt.

        Tier 1 (recent session turns) is token-capped: as many recent turns
        as fit within the remaining budget after Tiers 2 and 3 are reserved.
        """
        # ── Step 1: Fetch long-term memory (Tiers 2 + 3) in parallel ──────────
        core_text = ""
        episodic_text = ""

        if memory_manager is not None:
            # Tier 3: always fetch core facts
            core_facts = await memory_manager.get_all_core()
            core_text = memory_manager.format_core_context(core_facts)

            # Tier 2: retrieve episodic memories relevant to current query
            # Run in parallel with computing system+core tokens
            if query:
                episodic_text = await memory_manager.retrieve_relevant(
                    query=query,
                    k=5,
                    min_relevance=0.3,
                )

        # ── Step 2: Build the augmented system prompt ─────────────────────────
        system_prompt = session._system_prompt
        if core_text:
            system_prompt = system_prompt + core_text
        if episodic_text:
            system_prompt = system_prompt + episodic_text

        # Re-wrap session messages with the augmented system prompt
        augmented_system = [Message.system(system_prompt)] if system_prompt else []
        other_msgs = [m for m in session.messages]  # no system msg here

        # ── Step 3: Check if everything fits without truncation ───────────────
        all_messages = augmented_system + other_msgs
        token_count = await self._token_counter(all_messages)
        if token_count <= self.token_budget:
            return ComposedContext(
                messages=all_messages,
                token_count=token_count,
                token_budget=self.token_budget,
                breakdown={
                    "all": token_count,
                    "has_core_memory": bool(core_text),
                    "has_episodic_memory": bool(episodic_text),
                },
            )

        # ── Step 4: Token-budget truncation of Tier 1 ─────────────────────────
        # Keep as many recent session turns as possible after reserving space
        # for the (already augmented) system prompt.
        window = min(recent_window, len(other_msgs))

        while window > 0:
            candidate = augmented_system + other_msgs[-window:]
            token_count = await self._token_counter(candidate)
            if token_count <= self.token_budget:
                return ComposedContext(
                    messages=candidate,
                    token_count=token_count,
                    token_budget=self.token_budget,
                    breakdown={
                        "system": 1,
                        "recent": window,
                        "truncated": len(other_msgs) - window,
                        "has_core_memory": bool(core_text),
                        "has_episodic_memory": bool(episodic_text),
                    },
                )
            window -= 1

        # ── Worst case: only the augmented system prompt ──────────────────────
        system_tokens = await self._token_counter(augmented_system)
        return ComposedContext(
            messages=augmented_system,
            token_count=system_tokens,
            token_budget=self.token_budget,
            breakdown={
                "system": 1,
                "truncated": len(other_msgs),
                "has_core_memory": bool(core_text),
                "has_episodic_memory": bool(episodic_text),
            },
        )