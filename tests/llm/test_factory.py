"""Tests for the LLM provider factory."""

from unittest.mock import patch, MagicMock

import httpx
import pytest

from arc.llm.factory import (
    create_llm,
    fetch_models,
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


# ── fetch_models ─────────────────────────────────────────────────


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "http://fake"),
    )


class TestFetchModels:
    """Tests for live model fetching via fetch_models()."""

    def test_openai_compat(self):
        """Parses OpenAI-compatible /models response."""
        fake = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"id": "gpt-3.5"}]}
        with patch("arc.llm.factory.httpx.Client") as MC:
            mc = MagicMock()
            mc.get.return_value = _mock_response(fake)
            MC.return_value = mc

            models = fetch_models("openai", "https://api.openai.com/v1", "sk-x")

        assert models == ["gpt-3.5", "gpt-4o", "gpt-4o-mini"]  # sorted
        mc.get.assert_called_once_with("https://api.openai.com/v1/models")
        mc.close.assert_called_once()

    def test_ollama(self):
        """Parses Ollama /api/tags response."""
        fake = {"models": [{"name": "llama3.2"}, {"name": "mistral"}, {"name": "gemma2"}]}
        with patch("arc.llm.factory.httpx.Client") as MC:
            mc = MagicMock()
            mc.get.return_value = _mock_response(fake)
            MC.return_value = mc

            models = fetch_models("ollama", "http://localhost:11434")

        assert models == ["gemma2", "llama3.2", "mistral"]
        mc.get.assert_called_once_with("http://localhost:11434/api/tags")

    def test_groq(self):
        """Works with Groq (OpenAI-compatible)."""
        fake = {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "gpt-oss-120b"}]}
        with patch("arc.llm.factory.httpx.Client") as MC:
            mc = MagicMock()
            mc.get.return_value = _mock_response(fake)
            MC.return_value = mc

            models = fetch_models("groq", "https://api.groq.com/openai/v1", "gsk_x")

        assert "gpt-oss-120b" in models
        assert "llama-3.3-70b-versatile" in models

    def test_auth_failure_raises(self):
        """HTTPStatusError raised on 401."""
        resp = httpx.Response(401, json={"error": "bad"}, request=httpx.Request("GET", "http://f"))
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("401", request=resp.request, response=resp)
        )
        with patch("arc.llm.factory.httpx.Client") as MC:
            mc = MagicMock()
            mc.get.return_value = resp
            MC.return_value = mc

            with pytest.raises(httpx.HTTPStatusError):
                fetch_models("openai", "https://api.openai.com/v1", "bad")

        mc.close.assert_called_once()  # cleanup even on failure

    def test_empty_response(self):
        """Returns empty list when provider has no models."""
        with patch("arc.llm.factory.httpx.Client") as MC:
            mc = MagicMock()
            mc.get.return_value = _mock_response({"data": []})
            MC.return_value = mc

            assert fetch_models("openai", "https://api.openai.com/v1", "sk-x") == []

    def test_strips_trailing_slash(self):
        """Handles trailing slash in base_url."""
        with patch("arc.llm.factory.httpx.Client") as MC:
            mc = MagicMock()
            mc.get.return_value = _mock_response({"data": [{"id": "m1"}]})
            MC.return_value = mc

            fetch_models("openai", "https://api.openai.com/v1/", "sk-x")

        mc.get.assert_called_once_with("https://api.openai.com/v1/models")
