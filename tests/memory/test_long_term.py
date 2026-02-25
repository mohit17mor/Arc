"""Tests for LongTermMemory (SQLite + sqlite-vec)."""

import pytest
from pathlib import Path
from arc.memory.long_term import LongTermMemory


@pytest.fixture
async def mem(tmp_path: Path) -> LongTermMemory:
    """Isolated in-memory (tmp_path) LongTermMemory for each test."""
    m = LongTermMemory(tmp_path / "test_memory.db", embed_dim=DIM)
    await m.initialize()
    return m


# ── Core memory (Tier 3) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_and_get_core(mem):
    await mem.upsert_core("user_name", "User's name is Alice", confidence=0.9)
    facts = await mem.get_all_core()
    assert len(facts) == 1
    assert facts[0].id == "user_name"
    assert "Alice" in facts[0].content
    assert facts[0].confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_upsert_core_updates_existing(mem):
    await mem.upsert_core("user_name", "User's name is Bob")
    await mem.upsert_core("user_name", "User's name is Charlie", confidence=0.7)
    facts = await mem.get_all_core()
    assert len(facts) == 1
    assert "Charlie" in facts[0].content


@pytest.mark.asyncio
async def test_get_all_core_empty(mem):
    facts = await mem.get_all_core()
    assert facts == []


@pytest.mark.asyncio
async def test_delete_core(mem):
    await mem.upsert_core("lang", "User prefers Python")
    deleted = await mem.delete_core("lang")
    assert deleted is True
    facts = await mem.get_all_core()
    assert facts == []


@pytest.mark.asyncio
async def test_delete_core_missing(mem):
    deleted = await mem.delete_core("nonexistent_key")
    assert deleted is False


@pytest.mark.asyncio
async def test_multiple_core_facts(mem):
    await mem.upsert_core("name", "Alice")
    await mem.upsert_core("project", "Arc agent framework")
    facts = await mem.get_all_core()
    ids = {f.id for f in facts}
    assert {"name", "project"} == ids


# ── Episodic memory (Tier 2) ─────────────────────────────────────────────────

DIM = 8


def _vec(seed: float = 0.5) -> list[float]:
    """Create a normalised dummy vector."""
    import math
    raw = [seed + i * 0.01 for i in range(DIM)]
    norm = math.sqrt(sum(v * v for v in raw))
    return [v / norm for v in raw]


@pytest.mark.asyncio
async def test_store_episodic(mem):
    row_id = await mem.store_episodic(
        content="User is building an AI agent",
        embedding=_vec(0.5),
        source="conversation",
        session_id="s-001",
        importance=0.8,
    )
    assert isinstance(row_id, int)
    assert row_id > 0


@pytest.mark.asyncio
async def test_episodic_count(mem):
    assert await mem.episodic_count() == 0
    await mem.store_episodic("memory 1", _vec(0.5), importance=0.6)
    await mem.store_episodic("memory 2", _vec(0.6), importance=0.7)
    assert await mem.episodic_count() == 2


@pytest.mark.asyncio
async def test_list_episodic(mem):
    await mem.store_episodic("first", _vec(0.5), importance=0.5)
    await mem.store_episodic("second", _vec(0.6), importance=0.6)
    items = await mem.list_episodic(limit=10)
    assert len(items) == 2
    contents = {i.content for i in items}
    assert {"first", "second"} == contents


@pytest.mark.asyncio
async def test_delete_episodic(mem):
    row_id = await mem.store_episodic("to delete", _vec(0.5), importance=0.5)
    deleted = await mem.delete_episodic(row_id)
    assert deleted is True
    assert await mem.episodic_count() == 0


@pytest.mark.asyncio
async def test_delete_episodic_missing(mem):
    deleted = await mem.delete_episodic(9999)
    assert deleted is False


@pytest.mark.asyncio
async def test_search_episodic_returns_results(mem):
    # Store two memories with very different embeddings
    await mem.store_episodic("Python programming", _vec(0.1), importance=0.7)
    await mem.store_episodic("Cooking recipes", _vec(0.9), importance=0.7)

    results = await mem.search_episodic(query_embedding=_vec(0.1), k=2)
    assert len(results) <= 2
    # "Python programming" should rank higher (closer vector)
    assert results[0].memory.content == "Python programming"


@pytest.mark.asyncio
async def test_search_episodic_empty_db(mem):
    results = await mem.search_episodic(query_embedding=_vec(), k=5)
    assert results == []


@pytest.mark.asyncio
async def test_search_increments_access_count(mem):
    row_id = await mem.store_episodic("test memory", _vec(0.5), importance=0.5)
    items_before = await mem.list_episodic()
    initial_access = items_before[0].access_count

    await mem.search_episodic(query_embedding=_vec(0.5), k=1)

    items_after = await mem.list_episodic()
    final_access = items_after[0].access_count
    assert final_access > initial_access
