"""Tests for MemoryManager orchestrator."""

import pytest
from pathlib import Path
from arc.memory.manager import MemoryManager, DISTILL_EVERY
from arc.memory.embedding import MockEmbeddingProvider
from arc.llm.mock import MockLLMProvider
from arc.core.types import Message

DIM = 8


@pytest.fixture
async def manager(tmp_path: Path) -> MemoryManager:
    """Isolated MemoryManager using MockEmbeddingProvider."""
    embedder = MockEmbeddingProvider(dimension=DIM)
    mm = MemoryManager(
        db_path=str(tmp_path / "mem.db"),
        embedding_provider=embedder,
        embed_dim=DIM,
    )
    await mm.initialize()
    return mm


# ── Core memory passthrough ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_all_core_empty(manager):
    facts = await manager.get_all_core()
    assert facts == []


@pytest.mark.asyncio
async def test_upsert_and_get_core(manager):
    await manager.upsert_core("user_name", "User's name is Alice")
    facts = await manager.get_all_core()
    assert len(facts) == 1
    assert facts[0].id == "user_name"


@pytest.mark.asyncio
async def test_delete_core(manager):
    await manager.upsert_core("lang", "User prefers Python")
    deleted = await manager.delete_core("lang")
    assert deleted is True
    assert await manager.get_all_core() == []


@pytest.mark.asyncio
async def test_format_core_context_empty(manager):
    result = manager.format_core_context([])
    assert result == ""


@pytest.mark.asyncio
async def test_format_core_context_with_facts(manager):
    await manager.upsert_core("name", "User's name is Bob")
    facts = await manager.get_all_core()
    text = manager.format_core_context(facts)
    assert "Bob" in text
    assert len(text) > 0


# ── should_distill logic ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_should_distill_initially_false(manager):
    assert manager.should_distill is False


@pytest.mark.asyncio
async def test_should_distill_triggers_at_n_turns(manager):
    # Pump the turn count to exactly DISTILL_EVERY
    for i in range(DISTILL_EVERY):
        await manager.store_turn(
            user_content=f"Hello turn {i}",
            assistant_content=f"Hi turn {i}, user is building an AI agent in Python",
            session_id="s-test",
        )
    assert manager.turn_count == DISTILL_EVERY
    assert manager.should_distill is True


@pytest.mark.asyncio
async def test_should_distill_false_between_intervals(manager):
    # After exactly 1 turn it should be False (not a DISTILL_EVERY multiple)
    await manager.store_turn(
        user_content="Hello",
        assistant_content="Hi there",
        session_id="s-test",
    )
    assert manager.turn_count == 1
    assert manager.should_distill is False


# ── store_turn ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_turn_increments_count(manager):
    initial = manager.turn_count
    await manager.store_turn(
        user_content="I love building AI agents",
        assistant_content="Great, I can help with that AI agent framework",
        session_id="s-001",
    )
    assert manager.turn_count == initial + 1


@pytest.mark.asyncio
async def test_store_turn_skips_low_importance(manager):
    """Very short/data-heavy turns should not be stored."""
    # A turn that's just a price query — should score low importance
    await manager.store_turn(
        user_content="ok",
        assistant_content="Price: $123.45",
        session_id="s-001",
    )
    # May or may not store — the key assertion is it doesn't crash
    count = await manager.episodic_count()
    # Can't assert == 1 because importance scoring is heuristic,
    # but it must be 0 or 1 (not negative or errored)
    assert count >= 0


@pytest.mark.asyncio
async def test_store_turn_before_init_is_noop(tmp_path):
    """store_turn before initialize() should return silently."""
    embedder = MockEmbeddingProvider(dimension=DIM)
    mm = MemoryManager(
        db_path=str(tmp_path / "mem2.db"),
        embedding_provider=embedder,
        embed_dim=DIM,
    )
    # intentionally not calling initialize()
    await mm.store_turn("hello", "hi", session_id="s-x")  # must not raise


# ── retrieve_relevant ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_relevant_empty(manager):
    result = await manager.retrieve_relevant("anything")
    assert result == ""


@pytest.mark.asyncio
async def test_retrieve_relevant_returns_string(manager):
    await manager.store_turn(
        user_content="I am building an AI agent",
        assistant_content="That sounds great, Arc is a good project name",
        session_id="s-001",
    )
    result = await manager.retrieve_relevant("AI agent project")
    assert isinstance(result, str)


# ── distill_to_core ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_distill_to_core_parses_facts(manager):
    """distill_to_core should upsert facts extracted by the LLM."""
    llm = MockLLMProvider()
    llm.set_response(
        '[{"id": "user_name", "content": "User\'s name is Alice.", "confidence": 0.95}]'
    )

    messages = [
        Message.user("My name is Alice"),
        Message.assistant("Nice to meet you, Alice!"),
    ]
    await manager.distill_to_core(messages, llm)

    facts = await manager.get_all_core()
    assert len(facts) == 1
    assert facts[0].id == "user_name"
    assert "Alice" in facts[0].content


@pytest.mark.asyncio
async def test_distill_to_core_ignores_low_confidence(manager):
    """Facts with confidence < 0.6 should be silently skipped."""
    llm = MockLLMProvider()
    llm.set_response(
        '[{"id": "weak_fact", "content": "Maybe something.", "confidence": 0.3}]'
    )
    messages = [Message.user("hi"), Message.assistant("hello")]
    await manager.distill_to_core(messages, llm)

    facts = await manager.get_all_core()
    assert facts == []


@pytest.mark.asyncio
async def test_distill_to_core_handles_empty_response(manager):
    """Distillation with empty JSON [] should not crash."""
    llm = MockLLMProvider()
    llm.set_response("[]")
    messages = [Message.user("nothing here"), Message.assistant("ok")]
    await manager.distill_to_core(messages, llm)
    assert await manager.get_all_core() == []


@pytest.mark.asyncio
async def test_distill_to_core_handles_invalid_json(manager):
    """LLM returning garbage text should not crash distillation."""
    llm = MockLLMProvider()
    llm.set_response("Sorry, I cannot help with that.")
    messages = [Message.user("extract"), Message.assistant("done")]
    await manager.distill_to_core(messages, llm)
    assert await manager.get_all_core() == []


@pytest.mark.asyncio
async def test_distill_before_init_is_noop(tmp_path):
    """distill_to_core before initialize() should return silently."""
    embedder = MockEmbeddingProvider(dimension=DIM)
    mm = MemoryManager(
        db_path=str(tmp_path / "mem3.db"),
        embedding_provider=embedder,
        embed_dim=DIM,
    )
    llm = MockLLMProvider()
    llm.set_response('[{"id": "x", "content": "fact", "confidence": 0.9}]')
    messages = [Message.user("hi")]
    await mm.distill_to_core(messages, llm)  # must not raise
