"""
Context Compaction — summarise old messages instead of dropping them.

When the session memory approaches the context window limit, this module
asks the LLM to produce a short summary of the older messages.  The
summary replaces those messages in-place so the LLM retains the gist
of the conversation without the raw bulk.

Two usage modes (same core function, different timing):

    Main agent (interactive) — background, called after each turn ends:
        task = asyncio.create_task(prepare_compaction_in_background(...))
        # ... next turn starts ...
        if compaction ready:
            apply it

    Task / worker / scheduler agents — synchronous, at iteration start:
        await maybe_compact_session(...)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from arc.core.types import Message

if TYPE_CHECKING:
    from arc.llm.base import LLMProvider
    from arc.memory.session import SessionMemory

logger = logging.getLogger(__name__)

# Compaction triggers when token usage exceeds this fraction of the budget.
COMPACTION_THRESHOLD = 0.75

# System prompt for the summarisation call — terse, factual, cheap.
_COMPACT_SYSTEM = (
    "You are a conversation summariser. Condense the following conversation "
    "into a concise summary that preserves:\n"
    "- Key decisions and conclusions\n"
    "- Important facts and data discovered\n"
    "- Current task status and progress\n"
    "- User preferences and requirements stated\n"
    "- File paths, URLs, names, and other specifics mentioned\n\n"
    "Do NOT include greetings, filler, or per-message attribution. "
    "Write in bullet points. Be specific — keep numbers, names, paths. "
    "Keep it under 400 words."
)


def find_safe_cut_index(messages: list[Message], keep_recent: int = 6) -> int:
    """
    Find the latest safe index where we can cut old messages for summary.

    A safe boundary is AFTER a complete assistant response that has no
    tool_calls (i.e. a natural turn end).  We also keep at least
    ``keep_recent`` messages untouched so the LLM has recent context.

    Returns the index (exclusive) up to which messages can be summarised.
    Returns 0 if no safe cut point is found (= don't compact).
    """
    if len(messages) <= keep_recent:
        return 0

    # Search backwards from (len - keep_recent) for a safe boundary
    search_end = len(messages) - keep_recent

    for i in range(search_end - 1, -1, -1):
        msg = messages[i]
        # Safe boundary: assistant message with content and no tool_calls
        if msg.role == "assistant" and msg.content and not msg.tool_calls:
            return i + 1  # exclusive — include this message in the summary
        # Also safe: user message (the previous turn is complete)
        if msg.role == "user" and i > 0:
            return i

    return 0  # no safe point found


async def summarise_messages(
    messages: list[Message],
    llm: "LLMProvider",
) -> str:
    """
    Ask the LLM to summarise a list of messages into a concise block.

    This is a single, cheap, no-tools LLM call (~500 token output).
    """
    # Build plain-text conversation for the summariser
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            continue  # skip system prompt — it's always present anyway
        if msg.content:
            label = msg.role.capitalize()
            # Truncate very long messages to keep the summarisation input sane
            text = msg.content if len(msg.content) <= 1000 else msg.content[:1000] + "..."
            parts.append(f"{label}: {text}")
        elif msg.tool_calls:
            names = ", ".join(tc.name for tc in msg.tool_calls)
            parts.append(f"Assistant: [called tools: {names}]")

    if not parts:
        return ""

    convo_text = "\n".join(parts)

    try:
        summary_parts: list[str] = []
        async for chunk in llm.generate(
            messages=[
                Message.system(_COMPACT_SYSTEM),
                Message.user(f"Summarise this conversation:\n\n{convo_text}"),
            ],
            tools=None,
            temperature=0.1,  # factual
            max_tokens=800,
        ):
            if chunk.text:
                summary_parts.append(chunk.text)

        return "".join(summary_parts).strip()
    except Exception as e:
        logger.warning(f"Compaction summarisation failed: {e}")
        return ""


class CompactionState:
    """
    Tracks the background compaction lifecycle for one agent.

    Main agent flow:
        1. After turn → check_and_start_background()  (fires bg task if needed)
        2. Before compose → apply_if_ready()           (swaps messages if done)

    Task/worker/scheduler flow:
        1. Before compose → maybe_compact_sync()       (blocks, compacts if needed)
    """

    def __init__(self) -> None:
        self._pending_task: asyncio.Task | None = None
        self._pending_summary: str | None = None
        self._pending_cut_index: int = 0
        self._pending_message_count: int = 0  # snapshot of len(messages) when we started

    @property
    def has_pending(self) -> bool:
        """True if a background compaction result is ready to apply."""
        return self._pending_summary is not None

    def check_and_start_background(
        self,
        session: "SessionMemory",
        token_count: int,
        token_budget: int,
        llm: "LLMProvider",
    ) -> None:
        """
        Check if compaction is needed and start background summarisation.

        Called after each turn completes (main agent only).
        """
        # Already have a pending compaction or one in progress?
        if self._pending_summary is not None:
            return
        if self._pending_task is not None and not self._pending_task.done():
            return

        usage_ratio = token_count / token_budget if token_budget > 0 else 0
        if usage_ratio < COMPACTION_THRESHOLD:
            return

        messages = session.messages  # excludes system prompt
        cut_index = find_safe_cut_index(messages)
        if cut_index == 0:
            return  # no safe point — too few messages

        # Snapshot the messages to summarise
        to_summarise = messages[:cut_index]
        msg_count_snapshot = len(messages)

        logger.info(
            f"Compaction triggered (usage={usage_ratio:.0%}): "
            f"summarising {cut_index} of {len(messages)} messages in background"
        )

        self._pending_cut_index = cut_index
        self._pending_message_count = msg_count_snapshot
        self._pending_task = asyncio.create_task(
            self._run_background(to_summarise, llm)
        )

    async def _run_background(
        self,
        messages: list[Message],
        llm: "LLMProvider",
    ) -> None:
        """Background coroutine: produce summary, store it for later apply."""
        summary = await summarise_messages(messages, llm)
        if summary:
            self._pending_summary = summary
            logger.info(
                f"Compaction summary ready ({len(summary)} chars) — "
                f"will apply on next turn"
            )
        else:
            logger.warning("Compaction produced empty summary — skipping")
            self._pending_cut_index = 0
            self._pending_message_count = 0

    def apply_if_ready(self, session: "SessionMemory") -> bool:
        """
        Apply pending compaction to session memory if ready.

        Called at the start of each turn (before compose).
        Returns True if compaction was applied.
        """
        if self._pending_summary is None:
            return False

        summary = self._pending_summary
        cut_index = self._pending_cut_index

        # Reset pending state
        self._pending_summary = None
        self._pending_task = None

        messages = session.messages
        if len(messages) < cut_index:
            # Messages were cleared between start and apply — skip
            logger.debug("Session shorter than cut index — skipping compaction")
            self._pending_cut_index = 0
            self._pending_message_count = 0
            return False

        # Replace messages[0:cut_index] with one summary message
        summary_msg = Message.user(
            f"[Conversation summary — earlier messages were compacted]\n\n"
            f"{summary}"
        )
        session.messages[:cut_index] = [summary_msg]

        logger.info(
            f"Compaction applied: replaced {cut_index} messages with summary, "
            f"{len(session.messages)} messages remain"
        )

        self._pending_cut_index = 0
        self._pending_message_count = 0
        return True

    async def maybe_compact_sync(
        self,
        session: "SessionMemory",
        token_count: int,
        token_budget: int,
        llm: "LLMProvider",
    ) -> bool:
        """
        Synchronous compaction for background agents (task/worker/scheduler).

        Checks threshold, summarises, and applies — all in one call.
        Returns True if compaction was applied.
        """
        usage_ratio = token_count / token_budget if token_budget > 0 else 0
        if usage_ratio < COMPACTION_THRESHOLD:
            return False

        messages = session.messages
        cut_index = find_safe_cut_index(messages)
        if cut_index == 0:
            return False

        to_summarise = messages[:cut_index]
        logger.info(
            f"Sync compaction triggered (usage={usage_ratio:.0%}): "
            f"summarising {cut_index} of {len(messages)} messages"
        )

        summary = await summarise_messages(to_summarise, llm)
        if not summary:
            return False

        summary_msg = Message.user(
            f"[Conversation summary — earlier messages were compacted]\n\n"
            f"{summary}"
        )
        session.messages[:cut_index] = [summary_msg]

        logger.info(
            f"Sync compaction applied: replaced {cut_index} messages with summary, "
            f"{len(session.messages)} messages remain"
        )
        return True
