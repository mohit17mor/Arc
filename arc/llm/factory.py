"""
LLM provider factory — creates the right provider from config.

Usage::

    from arc.llm.factory import create_llm

    llm = create_llm("openai", model="gpt-4o", api_key="sk-...")
    llm = create_llm("ollama", model="llama3.2")
    llm = create_llm("openrouter", model="anthropic/claude-sonnet-4-20250514", api_key="sk-or-...")
    llm = create_llm("groq", model="llama-3.3-70b-versatile", api_key="gsk_...")

Provider presets handle base_url and defaults automatically.
"""

from __future__ import annotations

import logging
from typing import Any

from arc.llm.base import LLMProvider

logger = logging.getLogger(__name__)


# ── Provider presets ───────────────────────────────────────────────
# Each preset maps a friendly name to the config needed for
# OpenAICompatibleProvider.  "ollama" is special — it uses
# OllamaProvider instead.

_PRESETS: dict[str, dict[str, Any]] = {
    "ollama": {
        "class": "ollama",
        "base_url": "http://localhost:11434",
        "default_model": "llama3.2",
        "needs_key": False,
        "label": "Ollama (local, free)",
    },
    "openai": {
        "class": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "needs_key": True,
        "label": "OpenAI",
    },
    "openrouter": {
        "class": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "needs_key": True,
        "label": "OpenRouter (Claude, GPT, Gemini, Llama, etc.)",
    },
    "groq": {
        "class": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "needs_key": True,
        "label": "Groq (ultra-fast inference)",
    },
    "together": {
        "class": "openai_compat",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "needs_key": True,
        "label": "Together AI",
    },
    "lmstudio": {
        "class": "openai_compat",
        "base_url": "http://localhost:1234/v1",
        "default_model": "default",
        "needs_key": False,
        "label": "LM Studio (local)",
    },
    "custom": {
        "class": "openai_compat",
        "base_url": "",
        "default_model": "",
        "needs_key": False,
        "label": "Custom OpenAI-compatible API",
    },
}


def get_presets() -> dict[str, dict[str, Any]]:
    """Return all provider presets (for arc init menu)."""
    return dict(_PRESETS)


def get_preset(name: str) -> dict[str, Any] | None:
    """Get a preset by name."""
    return _PRESETS.get(name)


def list_provider_names() -> list[str]:
    """List all known provider names."""
    return list(_PRESETS.keys())


def create_llm(
    provider: str = "ollama",
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    context_window: int = 128000,
    max_output_tokens: int = 4096,
    **extra: Any,
) -> LLMProvider:
    """
    Create an LLM provider from config values.

    Args:
        provider: Provider name (ollama, openai, openrouter, groq, etc.)
        model: Model name/ID. Falls back to preset default.
        base_url: API base URL. Falls back to preset default.
        api_key: API key (required for cloud providers).
        context_window: Max input tokens.
        max_output_tokens: Max output tokens.
        **extra: Provider-specific kwargs.

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ValueError: If provider is unknown or required config is missing.
    """
    provider = provider.lower().strip()

    # Look up preset (may be None for totally unknown providers)
    preset = _PRESETS.get(provider)

    if preset and preset["class"] == "ollama":
        return _create_ollama(
            model=model or preset["default_model"],
            base_url=base_url or preset["base_url"],
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )

    # Everything else → OpenAI-compatible
    return _create_openai_compat(
        provider_name=provider,
        model=model or (preset["default_model"] if preset else ""),
        base_url=base_url or (preset["base_url"] if preset else ""),
        api_key=api_key,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


def _create_ollama(
    model: str,
    base_url: str,
    context_window: int,
    max_output_tokens: int,
) -> LLMProvider:
    from arc.llm.ollama import OllamaProvider

    logger.info(f"Creating Ollama provider: {model} at {base_url}")
    return OllamaProvider(
        base_url=base_url,
        model=model,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


def _create_openai_compat(
    provider_name: str,
    model: str,
    base_url: str,
    api_key: str,
    context_window: int,
    max_output_tokens: int,
) -> LLMProvider:
    from arc.llm.openai_compat import OpenAICompatibleProvider

    if not base_url:
        raise ValueError(
            f"Provider '{provider_name}': base_url is required. "
            f"Known providers: {', '.join(_PRESETS.keys())}"
        )

    if not model:
        raise ValueError(
            f"Provider '{provider_name}': model is required."
        )

    logger.info(
        f"Creating OpenAI-compatible provider: "
        f"{provider_name}/{model} at {base_url}"
    )
    return OpenAICompatibleProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        provider_name=provider_name,
    )
