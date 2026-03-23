"""Tests for setup wizard helper flows."""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console

from arc.identity import setup as setup_mod
from arc.identity.personality import get_personality
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


def _install_fake_questionary(
    monkeypatch,
    *,
    select_result="value",
    text_result="text",
    password_result="secret",
    confirm_result=True,
):
    """Install a lightweight fake questionary module for helper tests."""
    calls: dict[str, dict] = {}

    class FakePrompt:
        def __init__(self, result):
            self._result = result

        def ask(self):
            return self._result

    class FakeChoice:
        def __init__(self, title, value):
            self.title = title
            self.value = value

    def fake_select(message, **kwargs):
        calls["select"] = {"message": message, **kwargs}
        return FakePrompt(select_result)

    def fake_text(message, **kwargs):
        calls["text"] = {"message": message, **kwargs}
        return FakePrompt(text_result)

    def fake_password(message, **kwargs):
        calls["password"] = {"message": message, **kwargs}
        return FakePrompt(password_result)

    def fake_confirm(message, **kwargs):
        calls["confirm"] = {"message": message, **kwargs}
        return FakePrompt(confirm_result)

    fake_module = types.SimpleNamespace(
        select=fake_select,
        text=fake_text,
        password=fake_password,
        confirm=fake_confirm,
        Choice=FakeChoice,
        Style=lambda styles: styles,
    )
    monkeypatch.setitem(sys.modules, "questionary", fake_module)
    return calls


def test_q_select_clears_invalid_default_and_maps_choices(monkeypatch):
    calls = _install_fake_questionary(monkeypatch, select_result="openai")

    result = setup_mod._q_select(
        "Pick provider",
        [
            {"value": "ollama", "name": "Ollama"},
            {"value": "openai", "name": "OpenAI"},
        ],
        default="missing",
    )

    assert result == "openai"
    assert calls["select"]["default"] is None
    assert [choice.title for choice in calls["select"]["choices"]] == ["Ollama", "OpenAI"]
    assert [choice.value for choice in calls["select"]["choices"]] == ["ollama", "openai"]


@pytest.mark.parametrize(
    ("func_name", "install_kwargs", "args"),
    [
        ("_q_text", {"text_result": None}, ("Prompt",)),
        ("_q_password", {"password_result": None}, ("Secret",)),
        ("_q_confirm", {"confirm_result": None}, ("Confirm?",)),
    ],
)
def test_questionary_helpers_raise_keyboard_interrupt_on_cancel(
    monkeypatch, func_name, install_kwargs, args
):
    _install_fake_questionary(monkeypatch, **install_kwargs)

    with pytest.raises(KeyboardInterrupt, match="Setup cancelled"):
        getattr(setup_mod, func_name)(*args)


def test_q_multiline_text_returns_default_when_terminated_immediately():
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, width=100)
    inputs = iter(["END"])
    console.input = lambda _prompt="": next(inputs)

    result = setup_mod._q_multiline_text(
        console,
        "Enter prompt",
        default="Keep this prompt",
    )

    assert result == "Keep this prompt"


def test_q_multiline_text_collects_lines_until_terminator():
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, width=100)
    inputs = iter(["line one", "line two", "STOP"])
    console.input = lambda _prompt="": next(inputs)

    result = setup_mod._q_multiline_text(
        console,
        "Enter prompt",
        terminator="STOP",
    )

    assert result == "line one\nline two"


def test_pick_worker_model_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(setup_mod.Confirm, "ask", lambda *args, **kwargs: False)
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    result = setup_mod._pick_worker_model(console, interactive=False)

    assert result is None


def test_pick_worker_model_delegates_to_provider(monkeypatch):
    monkeypatch.setattr(setup_mod, "_q_confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        setup_mod,
        "_pick_provider",
        lambda *args, **kwargs: {"provider": "openai", "model": "gpt-4o-mini"},
    )
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    result = setup_mod._pick_worker_model(
        console,
        interactive=True,
        defaults={"provider": "openai", "model": "gpt-4o-mini"},
    )

    assert result == {"provider": "openai", "model": "gpt-4o-mini"}


def test_pick_tavily_disables_when_key_missing(monkeypatch):
    monkeypatch.setattr(setup_mod.Confirm, "ask", lambda *args, **kwargs: True)
    monkeypatch.setattr(setup_mod.Prompt, "ask", lambda *args, **kwargs: "   ")
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    result = setup_mod._pick_tavily(console, interactive=False)

    assert result is None


def test_pick_tavily_strips_api_and_ngrok_tokens(monkeypatch):
    monkeypatch.setattr(setup_mod.Confirm, "ask", lambda *args, **kwargs: True)
    answers = iter(["  tavily-key  ", "  ngrok-token  "])
    monkeypatch.setattr(setup_mod.Prompt, "ask", lambda *args, **kwargs: next(answers))
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    result = setup_mod._pick_tavily(console, interactive=False)

    assert result == {"api_key": "tavily-key", "ngrok_token": "ngrok-token"}


def test_pick_telegram_disables_when_token_missing(monkeypatch):
    monkeypatch.setattr(setup_mod.Confirm, "ask", lambda *args, **kwargs: True)
    answers = iter(["   ", "12345", "1,2"])
    monkeypatch.setattr(setup_mod.Prompt, "ask", lambda *args, **kwargs: next(answers))
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    result = setup_mod._pick_telegram(console, interactive=False)

    assert result is None


def test_pick_telegram_parses_allowed_users(monkeypatch):
    monkeypatch.setattr(setup_mod.Confirm, "ask", lambda *args, **kwargs: True)
    answers = iter(["  bot-token  ", "  12345  ", " alpha, beta ,, gamma "])
    monkeypatch.setattr(setup_mod.Prompt, "ask", lambda *args, **kwargs: next(answers))
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    result = setup_mod._pick_telegram(console, interactive=False)

    assert result == {
        "token": "bot-token",
        "chat_id": "12345",
        "allowed_users": ["alpha", "beta", "gamma"],
    }


def test_show_current_config_displays_enabled_and_disabled_sections():
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, width=100)

    setup_mod._show_current_config(
        console,
        {
            "user_name": "Mina",
            "agent_name": "Arc",
            "personality": "mentor",
            "provider": "openai",
            "model": "gpt-4o",
            "worker_provider": "openai",
            "worker_model": "gpt-4o-mini",
            "tavily_api_key": "tavily-key",
            "ngrok_auth_token": "ngrok-token",
        },
    )

    output = stream.getvalue()
    assert "Mina" in output
    assert "openai/gpt-4o" in output
    assert "worker: openai/gpt-4o-mini" in output
    assert "enabled" in output
    assert "not configured" in output


def test_pick_section_non_interactive_uses_selected_number(monkeypatch):
    monkeypatch.setattr(setup_mod.Prompt, "ask", lambda *args, **kwargs: "4")
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    result = setup_mod._pick_section(console, interactive=False)

    assert result == "telegram"


def test_run_sections_writes_full_config_and_identity(tmp_path, monkeypatch):
    config_path = tmp_path / "config" / "arc.toml"
    identity_path = tmp_path / "identity.md"
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    text_answers = iter(["Alice", "Atlas"])
    monkeypatch.setattr(setup_mod, "_q_text", lambda *args, **kwargs: next(text_answers))
    monkeypatch.setattr(setup_mod, "_q_select", lambda *args, **kwargs: "mentor")
    monkeypatch.setattr(
        setup_mod,
        "_pick_provider",
        lambda *args, **kwargs: {
            "provider": "openai",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-main",
        },
    )
    monkeypatch.setattr(
        setup_mod,
        "_pick_worker_model",
        lambda *args, **kwargs: {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-worker",
        },
    )
    monkeypatch.setattr(
        setup_mod,
        "_pick_tavily",
        lambda *args, **kwargs: {"api_key": "tv-key", "ngrok_token": "ng-token"},
    )
    monkeypatch.setattr(
        setup_mod,
        "_pick_telegram",
        lambda *args, **kwargs: {
            "token": "tg-token",
            "chat_id": "12345",
            "allowed_users": ["111", "222"],
        },
    )

    result = setup_mod._run_sections(
        frozenset({"identity", "models", "tavily", "telegram"}),
        {},
        console,
        True,
        config_path,
        identity_path,
    )

    config_text = config_path.read_text(encoding="utf-8")
    identity_text = identity_path.read_text(encoding="utf-8")

    assert result["personality"] == "mentor"
    assert result["worker_model"] == "gpt-4o-mini"
    assert result["tavily_api_key"] == "tv-key"
    assert result["telegram_allowed_users"] == ["111", "222"]
    assert 'default_provider = "openai"' in config_text
    assert 'worker_api_key = "sk-worker"' in config_text
    assert '[tavily]' in config_text
    assert 'auth_token = "ng-token"' in config_text
    assert 'allowed_users = ["111", "222"]' in config_text
    assert "personality: mentor" in identity_text
    assert "Alice" in identity_text
    assert "Atlas" in identity_text


def test_run_sections_custom_prompt_blank_falls_back_to_helpful_prompt(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    identity_path = tmp_path / "identity.md"
    console = Console(file=io.StringIO(), force_terminal=False, width=100)

    setup_mod.SoulManager(identity_path).create(
        "Old Arc",
        "Old User",
        "custom",
        custom_system_prompt="Previous custom prompt",
    )

    text_answers = iter(["Nina", "Nova"])
    monkeypatch.setattr(setup_mod, "_q_text", lambda *args, **kwargs: next(text_answers))
    monkeypatch.setattr(setup_mod, "_q_select", lambda *args, **kwargs: "custom")
    monkeypatch.setattr(setup_mod, "_q_multiline_text", lambda *args, **kwargs: "   ")

    result = setup_mod._run_sections(
        frozenset({"identity"}),
        {
            "provider": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
        },
        console,
        True,
        config_path,
        identity_path,
    )

    identity_text = identity_path.read_text(encoding="utf-8")

    assert result["personality"] == "custom"
    assert result["provider"] == "ollama"
    assert get_personality("helpful").system_prompt in identity_text
    assert "Previous custom prompt" not in identity_text
