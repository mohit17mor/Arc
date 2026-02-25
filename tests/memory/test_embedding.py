"""Tests for EmbeddingProvider implementations."""

import pytest
from arc.memory.embedding import MockEmbeddingProvider


@pytest.fixture
def embedder():
    return MockEmbeddingProvider(dimension=8)


@pytest.mark.asyncio
async def test_mock_dimension(embedder):
    vec = await embedder.embed_one("hello")
    assert len(vec) == 8
    assert embedder.dimension == 8


@pytest.mark.asyncio
async def test_mock_embed_returns_floats(embedder):
    vec = await embedder.embed_one("test text")
    assert all(isinstance(v, float) for v in vec)


@pytest.mark.asyncio
async def test_mock_embed_batch(embedder):
    texts = ["hello", "world", "foo"]
    vecs = await embedder.embed(texts)
    assert len(vecs) == 3
    assert all(len(v) == 8 for v in vecs)


@pytest.mark.asyncio
async def test_mock_deterministic(embedder):
    """Same text always produces the same vector."""
    v1 = await embedder.embed_one("deterministic text")
    v2 = await embedder.embed_one("deterministic text")
    assert v1 == v2


@pytest.mark.asyncio
async def test_mock_different_texts_different_vectors(embedder):
    """Different texts should produce different vectors."""
    v1 = await embedder.embed_one("apple")
    v2 = await embedder.embed_one("banana")
    assert v1 != v2


@pytest.mark.asyncio
async def test_mock_embed_one_matches_batch(embedder):
    """embed_one result should match first result of embed([...])."""
    text = "consistency check"
    single = await embedder.embed_one(text)
    batch = await embedder.embed([text])
    assert single == batch[0]
