"""
Long-Term Memory — persistent storage across sessions using SQLite + sqlite-vec.

Two tiers of storage:
    Tier 2 — Episodic memory: semantic chunks from past conversations,
              searched by vector similarity (KNN via sqlite-vec).
    Tier 3 — Core memory: high-confidence facts about the user, always
              injected into the system prompt.

All DB operations run in a thread-pool executor so they never block
the asyncio event loop. The connection uses WAL mode for safe
concurrent access from multiple platform processes (CLI + Telegram etc.)

Schema:
    core_memories     — key/value facts, always in system prompt
    episodic_memories — full-text memory entries with metadata
    episodic_vecs     — sqlite-vec virtual table for KNN search
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Dimension must match the embedding provider model (BAAI/bge-small-en-v1.5)
EMBED_DIM = 384

# Relevance scoring weights
W_SIMILARITY = 0.70
W_RECENCY    = 0.20
W_FREQUENCY  = 0.10
MAX_AGE_DAYS = 90.0


# ━━━ Data classes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CoreMemory:
    """A high-confidence, long-lived fact about the user."""
    id: str           # snake_case key, e.g. "user_name"
    content: str      # human-readable fact, e.g. "User's name is Mohit"
    confidence: float = 1.0
    updated_at: int = 0


@dataclass
class EpisodicMemory:
    """A semantic chunk from a past conversation."""
    id: int
    content: str
    source: str       # "conversation", "file_read", "web_search", etc.
    session_id: str
    importance: float
    access_count: int
    created_at: int
    last_accessed: int


@dataclass
class EpisodicResult:
    """An episodic memory with its relevance score."""
    memory: EpisodicMemory
    distance: float       # raw vec distance (lower = closer)
    relevance_score: float  # combined score after re-ranking (higher = better)


# ━━━ LongTermMemory ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LongTermMemory:
    """
    Persistent memory store backed by SQLite + sqlite-vec.

    Thread-safe: all operations run in a thread-pool executor.
    WAL mode enabled: safe for multiple concurrent processes.

    Usage:
        mem = LongTermMemory("~/.arc/memory/memory.db")
        await mem.initialize()

        # Store a core fact
        await mem.upsert_core("user_name", "User's name is Mohit")

        # Store an episodic memory with its embedding vector
        mem_id = await mem.store_episodic(
            content="User is building an AI agent framework in Python",
            embedding=vector,       # list[float] of EMBED_DIM length
            source="conversation",
            session_id="s-001",
        )

        # Retrieve most relevant memories for a query
        results = await mem.search_episodic(query_embedding=vector, k=5)
        for r in results:
            print(r.memory.content, r.relevance_score)
    """

    def __init__(self, db_path: str | Path, embed_dim: int | None = None) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db: sqlite3.Connection | None = None
        self._embed_dim = embed_dim if embed_dim is not None else EMBED_DIM

    # ━━━ Lifecycle ━━━

    async def initialize(self) -> None:
        """Set up the database and tables. Safe to call multiple times."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)

    def _init_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = self._get_db()

        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")

        # Tier 3 — core facts
        db.execute("""
            CREATE TABLE IF NOT EXISTS core_memories (
                id          TEXT    PRIMARY KEY,
                content     TEXT    NOT NULL,
                confidence  REAL    NOT NULL DEFAULT 1.0,
                updated_at  INTEGER NOT NULL DEFAULT (unixepoch())
            )
        """)

        # Tier 2 — episodic entries
        db.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                content      TEXT    NOT NULL,
                source       TEXT    NOT NULL DEFAULT 'conversation',
                session_id   TEXT    NOT NULL DEFAULT '',
                importance   REAL    NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at   INTEGER NOT NULL DEFAULT (unixepoch()),
                last_accessed INTEGER NOT NULL DEFAULT (unixepoch())
            )
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_session
            ON episodic_memories(session_id)
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_importance
            ON episodic_memories(importance DESC)
        """)

        # Tier 2 — vector index (rowid matches episodic_memories.id)
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_vecs
            USING vec0(embedding float[{self._embed_dim}])
        """)

        db.commit()
        logger.debug(f"LongTermMemory initialized at {self._db_path}")

    def _get_db(self) -> sqlite3.Connection:
        """Get or create the synchronous SQLite connection."""
        if self._db is None:
            import sqlite_vec
            db = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,  # executor threads vary
            )
            db.row_factory = sqlite3.Row
            # Load sqlite-vec extension
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            self._db = db
        return self._db

    async def close(self) -> None:
        if self._db:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._db.close)
            self._db = None

    # ━━━ Core memory (Tier 3) ━━━

    async def upsert_core(
        self,
        id: str,
        content: str,
        confidence: float = 1.0,
    ) -> CoreMemory:
        """Insert or update a core fact."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._upsert_core_sync, id, content, confidence
        )

    def _upsert_core_sync(
        self, id: str, content: str, confidence: float
    ) -> CoreMemory:
        db = self._get_db()
        now = int(time.time())
        db.execute(
            """
            INSERT INTO core_memories (id, content, confidence, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (id, content, confidence, now),
        )
        db.commit()
        return CoreMemory(id=id, content=content, confidence=confidence, updated_at=now)

    async def get_all_core(self) -> list[CoreMemory]:
        """Return all core memories sorted by confidence then recency."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_all_core_sync)

    def _get_all_core_sync(self) -> list[CoreMemory]:
        db = self._get_db()
        rows = db.execute(
            "SELECT id, content, confidence, updated_at "
            "FROM core_memories ORDER BY confidence DESC, updated_at DESC"
        ).fetchall()
        return [
            CoreMemory(
                id=r["id"],
                content=r["content"],
                confidence=r["confidence"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    async def delete_core(self, id: str) -> bool:
        """Delete a core memory by id. Returns True if it existed."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_core_sync, id)

    def _delete_core_sync(self, id: str) -> bool:
        db = self._get_db()
        cursor = db.execute("DELETE FROM core_memories WHERE id = ?", (id,))
        db.commit()
        return cursor.rowcount > 0

    # ━━━ Episodic memory (Tier 2) ━━━

    async def store_episodic(
        self,
        content: str,
        embedding: list[float],
        source: str = "conversation",
        session_id: str = "",
        importance: float = 0.5,
    ) -> int:
        """Store an episodic memory + its vector. Returns the new row id."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._store_episodic_sync,
            content, embedding, source, session_id, importance,
        )

    def _store_episodic_sync(
        self,
        content: str,
        embedding: list[float],
        source: str,
        session_id: str,
        importance: float,
    ) -> int:
        import sqlite_vec
        db = self._get_db()
        now = int(time.time())

        cursor = db.execute(
            """
            INSERT INTO episodic_memories
                (content, source, session_id, importance, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (content, source, session_id, importance, now, now),
        )
        row_id = cursor.lastrowid

        # Store the corresponding vector (rowid links the two tables)
        db.execute(
            "INSERT INTO episodic_vecs(rowid, embedding) VALUES (?, ?)",
            (row_id, sqlite_vec.serialize_float32(embedding)),
        )
        db.commit()
        logger.debug(f"Stored episodic memory id={row_id} source={source}")
        return row_id

    async def search_episodic(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[EpisodicResult]:
        """
        Find the k most relevant episodic memories for a query vector.

        Results are re-ranked using:
            relevance = 0.7*similarity + 0.2*recency + 0.1*frequency
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._search_episodic_sync, query_embedding, k
        )

    def _search_episodic_sync(
        self, query_embedding: list[float], k: int
    ) -> list[EpisodicResult]:
        import sqlite_vec
        db = self._get_db()
        now = time.time()

        # KNN search via sqlite-vec
        vec_rows = db.execute(
            """
            SELECT rowid, distance
            FROM episodic_vecs
            WHERE embedding MATCH ?
            AND k = ?
            ORDER BY distance
            """,
            (sqlite_vec.serialize_float32(query_embedding), k),
        ).fetchall()

        if not vec_rows:
            return []

        # Fetch metadata for matched rows
        ids = [r["rowid"] for r in vec_rows]
        dist_by_id = {r["rowid"]: r["distance"] for r in vec_rows}

        placeholders = ",".join("?" * len(ids))
        meta_rows = db.execute(
            f"SELECT * FROM episodic_memories WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        meta_by_id = {r["id"]: r for r in meta_rows}

        # Update access counts in batch
        db.execute(
            f"""
            UPDATE episodic_memories
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE id IN ({placeholders})
            """,
            [int(now)] + ids,
        )
        db.commit()

        # Build results with re-ranked relevance score
        results: list[EpisodicResult] = []
        for row_id in ids:
            meta = meta_by_id.get(row_id)
            if meta is None:
                continue
            distance = dist_by_id[row_id]
            relevance = _compute_relevance(
                distance=distance,
                created_at=meta["created_at"],
                access_count=meta["access_count"],
                now=now,
            )
            results.append(
                EpisodicResult(
                    memory=EpisodicMemory(
                        id=meta["id"],
                        content=meta["content"],
                        source=meta["source"],
                        session_id=meta["session_id"],
                        importance=meta["importance"],
                        access_count=meta["access_count"] + 1,
                        created_at=meta["created_at"],
                        last_accessed=int(now),
                    ),
                    distance=distance,
                    relevance_score=relevance,
                )
            )

        # Sort by relevance descending (best first)
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    async def delete_episodic(self, id: int) -> bool:
        """Delete an episodic memory and its vector."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_episodic_sync, id)

    def _delete_episodic_sync(self, id: int) -> bool:
        db = self._get_db()
        cursor = db.execute("DELETE FROM episodic_memories WHERE id = ?", (id,))
        db.execute("DELETE FROM episodic_vecs WHERE rowid = ?", (id,))
        db.commit()
        return cursor.rowcount > 0

    async def list_episodic(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EpisodicMemory]:
        """List episodic memories by recency."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._list_episodic_sync, limit, offset
        )

    def _list_episodic_sync(self, limit: int, offset: int) -> list[EpisodicMemory]:
        db = self._get_db()
        rows = db.execute(
            """
            SELECT * FROM episodic_memories
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [
            EpisodicMemory(
                id=r["id"],
                content=r["content"],
                source=r["source"],
                session_id=r["session_id"],
                importance=r["importance"],
                access_count=r["access_count"],
                created_at=r["created_at"],
                last_accessed=r["last_accessed"],
            )
            for r in rows
        ]

    async def episodic_count(self) -> int:
        """Return total number of episodic memories."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._episodic_count_sync)

    def _episodic_count_sync(self) -> int:
        db = self._get_db()
        row = db.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()
        return row[0] if row else 0


# ━━━ Helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _compute_relevance(
    distance: float,
    created_at: int,
    access_count: int,
    now: float,
) -> float:
    """
    Combine vector similarity, recency, and frequency into one score.

    distance: sqlite-vec cosine distance (0 = identical, 2 = opposite)
    Returns a score in [0, 1] — higher is more relevant.
    """
    similarity = max(0.0, 1.0 - (distance / 2.0))  # normalise 0→1

    age_days = (now - created_at) / 86400.0
    recency = max(0.0, 1.0 - (age_days / MAX_AGE_DAYS))

    frequency = min(1.0, access_count / 20.0)  # cap at 20 accesses

    return W_SIMILARITY * similarity + W_RECENCY * recency + W_FREQUENCY * frequency
