"""
Memory Manager — orchestrates the three-tier memory system.

Tiers:
    Tier 1  SessionMemory (in-RAM, current session) — handled by AgentLoop
    Tier 2  Episodic memory (sqlite-vec, semantic search across sessions)
    Tier 3  Core memory (sqlite, always injected into system prompt)

The manager is the single point of contact for the agent loop.
All heavy work (embedding, distillation) is designed to be launched
as asyncio background tasks so the user never waits.

Flow per turn:
    1. [parallel with LLM thinking] retrieve_relevant(query)
       → inject top memories into context
    2. [after response streams] asyncio.create_task(store_turn(...))
       → embed + store the turn as episodic memory
    3. [every DISTILL_EVERY turns] asyncio.create_task(distill_to_core(...))
       → LLM extracts stable facts → upsert to core_memories
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from arc.core.types import Message
from arc.memory.embedding import EmbeddingProvider, FastEmbedProvider
from arc.memory.long_term import CoreMemory, EpisodicMemory, LongTermMemory
from arc.memory.session import SessionMemory

if TYPE_CHECKING:
    from arc.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# How many turns between distillation calls
DISTILL_EVERY = 5

# Max characters of an assistant response to store as episodic memory
# (avoids storing huge tool outputs verbatim)
MAX_EPISODIC_CHARS = 1000

# Importance threshold — turns below this are not stored episodically
MIN_IMPORTANCE = 0.2

# System prompt used when asking the LLM to extract core facts
_DISTILL_SYSTEM = """You are a memory extraction assistant.

Your job: read a short conversation excerpt and extract ONLY permanent, stable facts about the USER.

Extract facts about:
- Name, location, occupation
- Preferences and opinions ("prefers X over Y", "dislikes Z")
- Active projects and goals
- Skills and expertise
- Important decisions made

DO NOT extract:
- Current prices, market data, weather, news
- File contents or code snippets
- Search results or web page content
- Anything time-sensitive that will be outdated soon
- Chitchat or generic responses

Return ONLY a JSON array. Each item must have:
  "id"         : snake_case key (short, stable, e.g. "user_language_pref")
  "content"    : one complete sentence stating the fact
  "confidence" : float 0.0-1.0

If no stable facts are present, return [].

Example:
[
  {"id": "user_language_pref", "content": "User prefers Python over JavaScript.", "confidence": 0.95},
  {"id": "active_project",     "content": "User is building a personal AI agent called Arc.", "confidence": 1.0}
]"""


class MemoryManager:
    """
    Three-tier memory orchestrator.

    Usage:
        manager = MemoryManager(db_path="~/.arc/memory/memory.db")
        await manager.initialize()

        # At context composition time:
        core_ctx   = manager.format_core_context()         # str for system prompt
        epi_ctx    = await manager.retrieve_relevant(query) # str for context

        # After each turn (fire and forget):
        asyncio.create_task(
            manager.store_turn(user_text, assistant_text, session_id)
        )

        # Every N turns (fire and forget):
        asyncio.create_task(
            manager.distill_to_core(recent_messages, llm)
        )
    """

    def __init__(
        self,
        db_path: str = "~/.arc/memory/memory.db",
        embedding_provider: EmbeddingProvider | None = None,
        embed_dim: int | None = None,
    ) -> None:
        self._store = LongTermMemory(db_path, embed_dim=embed_dim)
        self._embedder: EmbeddingProvider = embedding_provider or FastEmbedProvider()
        self._initialized = False
        self._turn_count = 0  # tracks when to trigger distillation

    async def initialize(self) -> None:
        """Initialize DB and load embedding model (one-time, ~1-2s first run)."""
        await self._store.initialize()
        await self._embedder.initialize()
        self._initialized = True
        count = await self._store.episodic_count()
        core = await self._store.get_all_core()
        logger.info(
            f"MemoryManager ready — {len(core)} core facts, {count} episodic memories"
        )

    async def close(self) -> None:
        await self._store.close()

    # ━━━ Core memory (Tier 3) ━━━

    async def get_all_core(self) -> list[CoreMemory]:
        return await self._store.get_all_core()

    async def upsert_core(self, id: str, content: str, confidence: float = 1.0) -> None:
        await self._store.upsert_core(id, content, confidence)

    async def delete_core(self, id: str) -> bool:
        return await self._store.delete_core(id)

    def format_core_context(self, core_facts: list[CoreMemory]) -> str:
        """
        Format core memories as a string to inject into the system prompt.

        Returns empty string if no core facts exist yet.
        """
        if not core_facts:
            return ""
        lines = ["\n\n## What I Know About You"]
        for fact in core_facts:
            lines.append(f"- {fact.content}")
        return "\n".join(lines)

    # ━━━ Episodic memory (Tier 2) ━━━

    async def retrieve_relevant(
        self,
        query: str,
        k: int = 5,
        min_relevance: float = 0.3,
    ) -> str:
        """
        Find and format the most relevant episodic memories for a query.

        Returns a formatted string ready to inject into context,
        or empty string if nothing relevant is found.
        """
        if not self._initialized:
            return ""
        try:
            query_vec = await self._embedder.embed_one(query)
            results = await self._store.search_episodic(query_vec, k=k * 2)  # over-fetch

            # Filter by minimum relevance threshold
            results = [r for r in results if r.relevance_score >= min_relevance]
            results = results[:k]  # trim to k

            if not results:
                return ""

            lines = ["\n\n## Relevant Memories From Past Sessions"]
            for r in results:
                lines.append(f"- {r.memory.content}")
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Memory retrieval failed (non-fatal): {e}")
            return ""

    async def store_turn(
        self,
        user_content: str,
        assistant_content: str,
        session_id: str = "",
    ) -> None:
        """
        Extract and store the important information from a conversation turn.

        Designed to run as a background task — safe to fire-and-forget.
        Only knowledge is stored, never raw data values.
        """
        if not self._initialized:
            return

        self._turn_count += 1

        # Build a combined text representing this turn's knowledge
        # Skip turns that are too short or look like data-only responses
        combined = f"User: {user_content}\nAssistant: {assistant_content}"
        importance = _score_importance(combined)

        if importance < MIN_IMPORTANCE:
            logger.debug("Turn importance too low, skipping episodic storage")
            return

        # Truncate to avoid storing huge tool outputs
        if len(combined) > MAX_EPISODIC_CHARS:
            combined = combined[:MAX_EPISODIC_CHARS] + "..."

        try:
            embedding = await self._embedder.embed_one(combined)
            await self._store.store_episodic(
                content=combined,
                embedding=embedding,
                source="conversation",
                session_id=session_id,
                importance=importance,
            )
            logger.debug(
                f"Stored episodic memory (turn {self._turn_count}, "
                f"importance={importance:.2f})"
            )
        except Exception as e:
            logger.warning(f"Episodic storage failed (non-fatal): {e}")

    async def distill_to_core(
        self,
        messages: list[Message],
        llm: LLMProvider,
    ) -> None:
        """
        Ask the LLM to extract stable facts from recent messages and
        upsert them into core memory.

        Designed to run as a background task — safe to fire-and-forget.
        Fires every DISTILL_EVERY turns automatically when called from AgentLoop.
        """
        if not self._initialized:
            return

        # Format the conversation as plain text for the LLM
        convo_parts = []
        for msg in messages:
            if msg.role in ("user", "assistant") and msg.content:
                role = "User" if msg.role == "user" else "Assistant"
                text = msg.content
                if len(text) > 500:
                    text = text[:500] + "..."
                convo_parts.append(f"{role}: {text}")

        if not convo_parts:
            return

        convo_text = "\n".join(convo_parts[-20:])  # last 20 turns max

        try:
            distill_messages = [
                Message.system(_DISTILL_SYSTEM),
                Message.user(
                    f"Extract stable facts from this conversation:\n\n{convo_text}"
                ),
            ]

            full_response = ""
            async for chunk in llm.generate(
                messages=distill_messages,
                tools=None,
                temperature=0.1,  # low temp for factual extraction
            ):
                if chunk.text:
                    full_response += chunk.text

            # Parse JSON array from response
            facts = _parse_json_facts(full_response)
            if not facts:
                logger.debug("Distillation returned no facts")
                return

            for fact in facts:
                fid = fact.get("id", "").strip()
                content = fact.get("content", "").strip()
                confidence = float(fact.get("confidence", 0.8))

                if fid and content and confidence >= 0.6:
                    await self._store.upsert_core(fid, content, confidence)
                    logger.debug(f"Core memory upserted: {fid!r}")

            logger.info(f"Distilled {len(facts)} facts into core memory")

        except Exception as e:
            logger.warning(f"Distillation failed (non-fatal): {e}")

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def should_distill(self) -> bool:
        """True when the current turn count warrants a distillation pass."""
        return self._turn_count > 0 and (self._turn_count % DISTILL_EVERY == 0)

    # ━━━ List/delete for /memory command ━━━

    async def list_episodic(self, limit: int = 20) -> list[EpisodicMemory]:
        return await self._store.list_episodic(limit=limit)

    async def delete_episodic(self, id: int) -> bool:
        return await self._store.delete_episodic(id)

    async def episodic_count(self) -> int:
        return await self._store.episodic_count()


# ━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _score_importance(text: str) -> float:
    """
    Lightweight heuristic to decide if a turn is worth storing.

    Returns 0.0 for trivial exchanges, up to 1.0 for rich content.
    """
    words = text.split()
    if len(words) < 8:
        return 0.0

    text_lower = text.lower()

    # Short circuit for clearly trivial responses
    trivial = {"ok", "okay", "thanks", "sure", "yes", "no", "got it", "alright"}
    if text_lower.strip().split("\n")[0].strip() in trivial:
        return 0.1

    score = 0.4  # baseline for any turn with substance

    # Boost for knowledge-rich signals
    for signal in [
        "prefer", "working on", "project", "building", "using", "my ",
        "i am", "i'm", "i use", "i like", "i don't", "i do", "i want",
        "important", "remember", "goal", "plan", "decision",
    ]:
        if signal in text_lower:
            score += 0.1

    # Penalise data-heavy content we don't want to store
    for signal in ["price", "weather", "$", "€", "rate:", "volume:"]:
        if signal in text_lower:
            score -= 0.2

    return max(0.0, min(1.0, score))


def _parse_json_facts(text: str) -> list[dict]:
    """
    Extract a JSON array from an LLM response that may include markdown fences
    or extra prose around the JSON.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    # Find the first [ ... ] JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group(0))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse distillation JSON: {e}")

    return []
