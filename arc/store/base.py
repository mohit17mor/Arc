"""
Storage Provider interface.

Simple key-value store with optional vector similarity search.
Used by the memory system to persist data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageProvider(ABC):
    """
    Abstract base class for storage backends.

    Keys are hierarchical strings: "identity/facts/001"
    Values are bytes (serialization is caller's responsibility).

    Implementations:
        SQLiteStorage — file-based, default
        InMemoryStorage — for testing
    """

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Get a value by key. Returns None if not found."""
        ...

    @abstractmethod
    async def set(self, key: str, value: bytes) -> None:
        """Set a value. Overwrites if exists."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""
        ...

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys matching a prefix."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend."""
        ...