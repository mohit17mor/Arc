"""Tests for setup wizard helper flows."""

from __future__ import annotations

import io

from rich.console import Console

from arc.identity import setup as setup_mod
from arc.llm.factory import get_presets


def test_pick_provider_codex_prompts_for_base_url(monkeypatch):
    """Codex provider should ask for base URL during setup."""
    presets = get_presets()
    provider_keys = list(presets.keys())
    codex_index = provider_keys.index("codex") + 1  # menu is 1-based

    answers = iter(
        [
            str(codex_index),  # provider choice
            "http://localhost:4000/v1",  # base URL
            "sk-test",  # api key
            "codex-mini-latest",  # model
        ]
    )
    prompts: list[str] = []

    def fake_prompt_ask(message: str, *args, **kwargs):
        prompts.append(message)
        return next(answers)

    monkeypatch.setattr(setup_mod.Prompt, "ask", fake_prompt_ask)
    monkeypatch.setattr(
        "arc.llm.factory.fetch_models",
        lambda provider, base_url, api_key="": [],
    )

    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    result = setup_mod._pick_provider(console, interactive=False, defaults={})

    assert result["provider"] == "codex"
    assert result["base_url"] == "http://localhost:4000/v1"
    assert result["api_key"] == "sk-test"
    assert result["model"] == "codex-mini-latest"
    assert any("Base URL" in p for p in prompts)


def test_pick_provider_codex_prefills_existing_configured_base_url(monkeypatch):
    """When a base URL is already configured, use it as codex default."""
    presets = get_presets()
    provider_keys = list(presets.keys())
    codex_index = provider_keys.index("codex") + 1  # menu is 1-based

    answers = iter(
        [
            str(codex_index),  # provider choice
            "",  # keep default base URL
            "sk-test",  # api key
            "codex-mini-latest",  # model
        ]
    )

    def fake_prompt_ask(message: str, *args, **kwargs):
        value = next(answers)
        if value == "":
            return kwargs.get("default", "")
        return value

    monkeypatch.setattr(setup_mod.Prompt, "ask", fake_prompt_ask)
    monkeypatch.setattr(
        "arc.llm.factory.fetch_models",
        lambda provider, base_url, api_key="": [],
    )

    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    result = setup_mod._pick_provider(
        console,
        interactive=False,
        defaults={
            "provider": "openai",
            "base_url": "http://configured.example/v1",
        },
    )

    assert result["provider"] == "codex"
    assert result["base_url"] == "http://configured.example/v1"
