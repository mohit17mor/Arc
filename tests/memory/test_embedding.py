"""Tests for EmbeddingProvider implementations."""

import pytest
from arc.memory.embedding import FastEmbedProvider, MockEmbeddingProvider


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


@pytest.mark.asyncio
async def test_mock_vectors_are_normalized(embedder):
    vec = await embedder.embed_one("normalized")
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0, rel=1e-6)


@pytest.mark.asyncio
async def test_fastembed_initialize_raises_helpful_error_when_dependency_missing(monkeypatch):
    provider = FastEmbedProvider()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="fastembed is not installed"):
        await provider.initialize()


@pytest.mark.asyncio
async def test_fastembed_embed_initializes_once_and_returns_lists(monkeypatch):
    provider = FastEmbedProvider()
    embedded_inputs = []

    class FakeVector:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            embedded_inputs.append(list(texts))
            return [FakeVector([1.0, 2.0]), FakeVector([3.0, 4.0])]

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            return type("FakeModule", (), {"TextEmbedding": FakeTextEmbedding})
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    vectors = await provider.embed(["hello", "world"])
    second = await provider.embed(["again"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert second == [[1.0, 2.0], [3.0, 4.0]]
    assert embedded_inputs == [["hello", "world"], ["again"]]
    assert provider.dimension == 384
