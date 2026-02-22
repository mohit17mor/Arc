"""Tests for storage backends."""

import pytest
from pathlib import Path
from arc.store.memory import InMemoryStorage
from arc.store.sqlite import SQLiteStorage


# ━━━ In-Memory Storage Tests ━━━


@pytest.mark.asyncio
async def test_memory_set_and_get():
    store = InMemoryStorage()
    await store.set("key1", b"value1")
    result = await store.get("key1")
    assert result == b"value1"
    await store.close()


@pytest.mark.asyncio
async def test_memory_get_missing():
    store = InMemoryStorage()
    result = await store.get("nonexistent")
    assert result is None
    await store.close()


@pytest.mark.asyncio
async def test_memory_overwrite():
    store = InMemoryStorage()
    await store.set("key", b"old")
    await store.set("key", b"new")
    result = await store.get("key")
    assert result == b"new"
    await store.close()


@pytest.mark.asyncio
async def test_memory_delete():
    store = InMemoryStorage()
    await store.set("key", b"value")
    deleted = await store.delete("key")
    assert deleted is True
    result = await store.get("key")
    assert result is None
    await store.close()


@pytest.mark.asyncio
async def test_memory_delete_missing():
    store = InMemoryStorage()
    deleted = await store.delete("nonexistent")
    assert deleted is False
    await store.close()


@pytest.mark.asyncio
async def test_memory_exists():
    store = InMemoryStorage()
    assert await store.exists("key") is False
    await store.set("key", b"value")
    assert await store.exists("key") is True
    await store.close()


@pytest.mark.asyncio
async def test_memory_list_keys():
    store = InMemoryStorage()
    await store.set("user/name", b"Alex")
    await store.set("user/age", b"30")
    await store.set("settings/theme", b"dark")

    all_keys = await store.list_keys()
    assert len(all_keys) == 3

    user_keys = await store.list_keys("user/")
    assert len(user_keys) == 2
    assert "user/name" in user_keys
    assert "user/age" in user_keys

    settings_keys = await store.list_keys("settings/")
    assert len(settings_keys) == 1
    await store.close()


@pytest.mark.asyncio
async def test_memory_empty_list():
    store = InMemoryStorage()
    keys = await store.list_keys()
    assert keys == []
    await store.close()


# ━━━ SQLite Storage Tests ━━━


@pytest.mark.asyncio
async def test_sqlite_set_and_get(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    await store.set("key1", b"value1")
    result = await store.get("key1")
    assert result == b"value1"
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_get_missing(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    result = await store.get("nonexistent")
    assert result is None
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_overwrite(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    await store.set("key", b"old")
    await store.set("key", b"new")
    result = await store.get("key")
    assert result == b"new"
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_delete(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    await store.set("key", b"value")
    deleted = await store.delete("key")
    assert deleted is True
    result = await store.get("key")
    assert result is None
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_delete_missing(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    deleted = await store.delete("nonexistent")
    assert deleted is False
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_exists(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    assert await store.exists("key") is False
    await store.set("key", b"value")
    assert await store.exists("key") is True
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_list_keys(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    await store.set("user/name", b"Alex")
    await store.set("user/age", b"30")
    await store.set("settings/theme", b"dark")

    all_keys = await store.list_keys()
    assert len(all_keys) == 3

    user_keys = await store.list_keys("user/")
    assert len(user_keys) == 2

    settings_keys = await store.list_keys("settings/")
    assert len(settings_keys) == 1
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_empty_list(tmp_path: Path):
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    keys = await store.list_keys()
    assert keys == []
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_creates_directory(tmp_path: Path):
    """SQLite storage creates parent directories."""
    db_path = tmp_path / "deep" / "nested" / "dir" / "test.db"
    store = SQLiteStorage(db_path)
    await store.initialize()
    await store.set("test", b"value")
    assert await store.get("test") == b"value"
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_persistence(tmp_path: Path):
    """Data persists across connections."""
    db_path = tmp_path / "persist.db"

    # Write
    store1 = SQLiteStorage(db_path)
    await store1.initialize()
    await store1.set("key", b"persisted")
    await store1.close()

    # Read with new connection
    store2 = SQLiteStorage(db_path)
    await store2.initialize()
    result = await store2.get("key")
    assert result == b"persisted"
    await store2.close()