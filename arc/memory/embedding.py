"""
Embedding Provider — converts text to vectors for semantic search.

Abstracted behind an interface so the backend can be swapped without
touching the memory or retrieval code.

Default implementation: FastEmbedProvider
    - Uses fastembed + ONNX runtime (no GPU needed)
    - Model: BAAI/bge-small-en-v1.5 (384-dim, ~25MB, fast)
    - Downloaded once to ~/.cache/fastembed on first use
    - Fully offline after first run
    - Runs in a thread pool to avoid blocking the event loop
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Embedding dimension for BAAI/bge-small-en-v1.5
EMBED_DIM = 384
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    async def initialize(self) -> None:
        """Load or warm up the model. Override if startup work is needed."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, returning one vector per text.

        Vectors are normalized (unit length) — cosine similarity
        is equivalent to dot product.
        """
        ...

    async def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single text."""
        results = await self.embed([text])
        return results[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of output vectors."""
        ...


class FastEmbedProvider(EmbeddingProvider):
    """
    Local embedding provider using fastembed + ONNX runtime.

    Zero API cost. Works offline after first model download (~25MB).
    Thread-pool based so it never blocks the asyncio event loop.

    Usage:
        provider = FastEmbedProvider()
        await provider.initialize()  # downloads model on first run

        vectors = await provider.embed(["hello world", "foo bar"])
        # vectors: list of 384-dim float lists
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model_name = model
        self._model: Any = None   # fastembed.TextEmbedding, lazy init

    async def initialize(self) -> None:
        """
        Load the model (downloads if not cached).

        Runs in a thread pool — download/load can take a few seconds
        the first time but never blocks the event loop.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        """Synchronous model load — intended to run in an executor."""
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(
                model_name=self._model_name,
                # Cache inside user's home dir so it survives venv recreations
            )
            logger.info(f"FastEmbed model loaded: {self._model_name}")
        except ImportError as e:
            raise ImportError(
                "fastembed is not installed. Run: pip install fastembed"
            ) from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, returning 384-dim normalized float vectors."""
        if self._model is None:
            await self.initialize()

        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, self._embed_sync, texts)
        return vectors

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding — runs in thread pool."""
        # fastembed returns a generator of numpy arrays
        raw = list(self._model.embed(texts))
        return [vec.tolist() for vec in raw]

    @property
    def dimension(self) -> int:
        return EMBED_DIM


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic mock for tests — no model download needed.

    Generates pseudo-random but stable vectors based on text hash,
    so the same text always returns the same vector.
    """

    def __init__(self, dimension: int = EMBED_DIM) -> None:
        self._dim = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            vec = rng.random(self._dim).astype(np.float32)
            # Normalize to unit length (cosine-compatible)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results

    @property
    def dimension(self) -> int:
        return self._dim
