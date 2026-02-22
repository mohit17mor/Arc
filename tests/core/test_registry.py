"""Tests for the Registry."""

import pytest
from arc.core.registry import Registry
from arc.core.errors import ProviderNotFoundError, RegistryError


def test_register_and_get(registry: Registry):
    registry.register("llm", "ollama", "ollama_instance")
    result = registry.get("llm", "ollama")
    assert result == "ollama_instance"


def test_get_default_returns_first(registry: Registry):
    registry.register("llm", "ollama", "ollama_instance")
    registry.register("llm", "openai", "openai_instance")

    result = registry.get("llm")  # no name → default (first registered)
    assert result == "ollama_instance"


def test_get_default_explicit(registry: Registry):
    registry.register("llm", "ollama", "ollama_instance")
    registry.register("llm", "openai", "openai_instance")
    registry.set_default("llm", "openai")

    result = registry.get("llm")
    assert result == "openai_instance"


def test_get_all(registry: Registry):
    registry.register("skill", "fs", "fs_instance")
    registry.register("skill", "terminal", "term_instance")
    registry.register("skill", "git", "git_instance")

    result = registry.get_all("skill")
    assert result == ["fs_instance", "term_instance", "git_instance"]


def test_get_all_preserves_order(registry: Registry):
    registry.register("skill", "c", "c")
    registry.register("skill", "a", "a")
    registry.register("skill", "b", "b")

    result = registry.get_all("skill")
    assert result == ["c", "a", "b"]  # insertion order, not alphabetical


def test_has(registry: Registry):
    assert registry.has("llm") is False
    assert registry.has("llm", "ollama") is False

    registry.register("llm", "ollama", "instance")

    assert registry.has("llm") is True
    assert registry.has("llm", "ollama") is True
    assert registry.has("llm", "openai") is False


def test_missing_category_raises(registry: Registry):
    with pytest.raises(ProviderNotFoundError, match="No providers registered"):
        registry.get("llm")


def test_missing_name_raises(registry: Registry):
    registry.register("llm", "ollama", "instance")

    with pytest.raises(ProviderNotFoundError, match="'openai' not found"):
        registry.get("llm", "openai")


def test_set_default_missing_raises(registry: Registry):
    with pytest.raises(RegistryError, match="not found"):
        registry.set_default("llm", "nonexistent")


def test_overwrite(registry: Registry):
    registry.register("llm", "ollama", "v1")
    registry.register("llm", "ollama", "v2")

    assert registry.get("llm", "ollama") == "v2"
    # Should not duplicate in registration order
    assert registry.get_names("llm") == ["ollama"]


def test_remove(registry: Registry):
    registry.register("llm", "ollama", "instance")
    registry.register("llm", "openai", "instance2")

    registry.remove("llm", "ollama")

    assert not registry.has("llm", "ollama")
    assert registry.has("llm", "openai")
    assert registry.get("llm") == "instance2"


def test_remove_default(registry: Registry):
    registry.register("llm", "ollama", "instance")
    registry.set_default("llm", "ollama")
    registry.remove("llm", "ollama")

    assert not registry.has("llm")


def test_clear(registry: Registry):
    registry.register("llm", "a", "1")
    registry.register("skill", "b", "2")
    registry.clear()

    assert not registry.has("llm")
    assert not registry.has("skill")


def test_get_names(registry: Registry):
    registry.register("skill", "fs", "1")
    registry.register("skill", "terminal", "2")

    assert registry.get_names("skill") == ["fs", "terminal"]
    assert registry.get_names("nonexistent") == []