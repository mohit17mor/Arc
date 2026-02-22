"""
In-memory storage backend — for testing.

Simple dict-based storage. Data lost when process exits.
"""

from __future__ import annotations

from arc.store.base import StorageProvider


class InMemoryStorage(StorageProvider):
    """
    In-memory key-value store for testing.

    Usage:
        storage = InMemoryStorage()
        await storage.set("key", b"value")
        assert await storage.get("key") == b"value"
    """

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def list_keys(self, prefix: str = "") -> list[str]:
        if prefix:
            return sorted(k for k in self._data if k.startswith(prefix))
        return sorted(self._data.keys())

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def close(self) -> None:
        self._data.clear()