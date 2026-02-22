"""
SQLite storage backend.

Uses aiosqlite for async SQLite access.
WAL mode enabled for concurrent read support.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

from arc.core.errors import StorageError
from arc.store.base import StorageProvider

logger = logging.getLogger(__name__)


class SQLiteStorage(StorageProvider):
    """
    SQLite-based key-value storage.

    Usage:
        storage = SQLiteStorage("~/.arc/memory/data.db")
        await storage.initialize()

        await storage.set("user/name", b"Alex")
        value = await storage.get("user/name")  # b"Alex"
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._db = await aiosqlite.connect(str(self._db_path))

            # Enable WAL mode for better concurrent access
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")

            # Create key-value table
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
                    updated_at REAL NOT NULL DEFAULT (unixepoch('now'))
                )
                """
            )

            # Index for prefix queries
            await self._db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kv_key ON kv(key)
                """
            )

            await self._db.commit()
            logger.debug(f"SQLite storage initialized at {self._db_path}")

        except Exception as e:
            raise StorageError(f"Failed to initialize SQLite at {self._db_path}: {e}")

    async def _ensure_db(self) -> aiosqlite.Connection:
        """Ensure database is initialized."""
        if self._db is None:
            await self.initialize()
        return self._db  # type: ignore[return-value]

    async def get(self, key: str) -> bytes | None:
        db = await self._ensure_db()
        try:
            async with db.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            raise StorageError(f"Failed to get key '{key}': {e}")

    async def set(self, key: str, value: bytes) -> None:
        db = await self._ensure_db()
        try:
            await db.execute(
                """
                INSERT INTO kv (key, value, updated_at) VALUES (?, ?, unixepoch('now'))
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = unixepoch('now')
                """,
                (key, value, value),
            )
            await db.commit()
        except Exception as e:
            raise StorageError(f"Failed to set key '{key}': {e}")

    async def delete(self, key: str) -> bool:
        db = await self._ensure_db()
        try:
            cursor = await db.execute("DELETE FROM kv WHERE key = ?", (key,))
            await db.commit()
            return cursor.rowcount > 0
        except Exception as e:
            raise StorageError(f"Failed to delete key '{key}': {e}")

    async def list_keys(self, prefix: str = "") -> list[str]:
        db = await self._ensure_db()
        try:
            if prefix:
                async with db.execute(
                    "SELECT key FROM kv WHERE key LIKE ? ORDER BY key",
                    (f"{prefix}%",),
                ) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with db.execute(
                    "SELECT key FROM kv ORDER BY key"
                ) as cursor:
                    rows = await cursor.fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            raise StorageError(f"Failed to list keys with prefix '{prefix}': {e}")

    async def exists(self, key: str) -> bool:
        db = await self._ensure_db()
        try:
            async with db.execute(
                "SELECT 1 FROM kv WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
                return row is not None
        except Exception as e:
            raise StorageError(f"Failed to check key '{key}': {e}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None