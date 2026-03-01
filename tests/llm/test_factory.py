"""Tests for the LLM provider factory."""

import pytest

from arc.llm.factory import (
    create_llm,
    get_preset,
    get_presets,
    list_provider_names,
)
from arc.llm.base import LLMProvider


# ── Preset registry ──────────────────────────────────────────────


class TestPresets:
    """Tests for preset metadata."""

    def test_list_provider_names_includes_core(self):
        names = list_provider_names()
        assert "ollama" in names
        assert "openai" in names
        assert "openrouter" in names
        assert "groq" in names
        assert "together" in names
        assert "lmstudio" in names
        assert "custom" in names

    def test_get_presets_returns_all(self):
        presets = get_presets()
        assert len(presets) >= 7

    def test_get_preset_known(self):
        p = get_preset("openai")
        assert p is not None
        assert p["base_url"] == "https://api.openai.com/v1"
        assert p["needs_key"] is True

    def test_get_preset_unknown(self):
        assert get_preset("nonexistent_provider") is None

    def test_preset_labels(self):
        presets = get_presets()
        for name, preset in presets.items():
            assert "label" in preset, f"Preset {name} missing label"
            assert "default_model" in preset, f"Preset {name} missing default_model"

    def test_ollama_preset_no_key(self):
        p = get_preset("ollama")
        assert p["needs_key"] is False
        assert p["class"] == "ollama"

    def test_cloud_presets_need_key(self):
        for name in ("openai", "openrouter", "groq", "together"):
            p = get_preset(name)
            assert p["needs_key"] is True, f"{name} should require API key"


# ── Factory — Ollama ─────────────────────────────────────────────


class TestCreateOllama:
    """Tests for creating Ollama providers via factory."""

    def test_creates_ollama_provider(self):
        llm = create_llm("ollama", model="llama3.2")
        assert isinstance(llm, LLMProvider)
        info = llm.get_model_info()
        assert info.provider == "ollama"
        assert info.model == "llama3.2"

    def test_ollama_default_model(self):
        llm = create_llm("ollama")
        info = llm.get_model_info()
        assert info.model == "llama3.2"  # preset default

    def test_ollama_custom_url(self):
        llm = create_llm("ollama", base_url="http://remote:11434")
        assert isinstance(llm, LLMProvider)

    def test_ollama_case_insensitive(self):
        llm = create_llm("Ollama", model="test-model")
        info = llm.get_model_info()
        assert info.provider == "ollama"


# ── Factory — OpenAI-compatible ──────────────────────────────────


class TestCreateOpenAICompat:
    """Tests for creating OpenAI-compatible providers via factory."""

    def test_creates_openai_provider(self):
        llm = create_llm("openai", model="gpt-4o", api_key="sk-test")
        assert isinstance(llm, LLMProvider)
        info = llm.get_model_info()
        assert info.provider == "openai"
        assert info.model == "gpt-4o"

    def test_creates_openrouter_provider(self):
        llm = create_llm("openrouter", model="anthropic/claude-sonnet-4-20250514", api_key="sk-or-test")
        info = llm.get_model_info()
        assert info.provider == "openrouter"

    def test_creates_groq_provider(self):
        llm = create_llm("groq", api_key="gsk_test")
        info = llm.get_model_info()
        assert info.provider == "groq"
        assert info.model == "llama-3.3-70b-versatile"  # preset default

    def test_creates_together_provider(self):
        llm = create_llm("together", api_key="tog_test")
        info = llm.get_model_info()
        assert info.provider == "together"

    def test_creates_lmstudio_provider(self):
        llm = create_llm("lmstudio", model="my-local-model")
        info = llm.get_model_info()
        assert info.provider == "lmstudio"
        assert info.model == "my-local-model"

    def test_custom_with_base_url(self):
        llm = create_llm(
            "custom",
            model="my-model",
            base_url="http://localhost:9000/v1",
        )
        info = llm.get_model_info()
        assert info.provider == "custom"
        assert info.model == "my-model"

    def test_custom_missing_base_url_raises(self):
        with pytest.raises(ValueError, match="base_url is required"):
            create_llm("custom", model="test")

    def test_custom_missing_model_raises(self):
        with pytest.raises(ValueError, match="model is required"):
            create_llm("custom", base_url="http://localhost:1234/v1")

    def test_unknown_provider_needs_base_url(self):
        """Unknown provider name requires explicit base_url."""
        with pytest.raises(ValueError, match="base_url is required"):
            create_llm("totally_unknown", model="x")

    def test_unknown_provider_with_base_url_works(self):
        llm = create_llm(
            "totally_unknown",
            model="whatever",
            base_url="http://example.com/v1",
        )
        assert isinstance(llm, LLMProvider)
        info = llm.get_model_info()
        assert info.provider == "totally_unknown"

    def test_context_window_passthrough(self):
        llm = create_llm(
            "openai",
            model="gpt-4o",
            api_key="sk-test",
            context_window=200000,
        )
        info = llm.get_model_info()
        assert info.context_window == 200000
