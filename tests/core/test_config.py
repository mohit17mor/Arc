"""Tests for the Config system."""

import os
import pytest
from pathlib import Path
from arc.core.config import ArcConfig, _deep_merge, _substitute_env_vars, _convert_value


def test_default_config():
    """Default config has sensible values."""
    config = ArcConfig()

    assert config.agent.max_iterations == 25
    assert config.agent.temperature == 0.7
    assert config.llm.default_provider == "ollama"
    assert config.llm.default_model == "llama3.1"
    assert config.llm.base_url == "http://localhost:11434"
    assert config.security.audit_enabled is True
    assert config.cost.enabled is True
    assert config.shell.provider == "auto"
    assert config.identity.agent_name == "Arc"


def test_load_with_overrides():
    """Explicit overrides take highest precedence."""
    config = ArcConfig.load(
        overrides={
            "llm": {
                "default_model": "mistral",
                "base_url": "http://my-server:11434",
            },
            "agent": {"max_iterations": 10},
        }
    )

    assert config.llm.default_model == "mistral"
    assert config.llm.base_url == "http://my-server:11434"
    assert config.agent.max_iterations == 10
    # Defaults still work for non-overridden values
    assert config.agent.temperature == 0.7


def test_env_var_loading(monkeypatch):
    """ARC_* environment variables are loaded."""
    monkeypatch.setenv("ARC_LLM_MODEL", "codellama")
    monkeypatch.setenv("ARC_LLM_BASE_URL", "http://remote:11434")
    monkeypatch.setenv("ARC_AGENT_MAX_ITERATIONS", "50")
    monkeypatch.setenv("ARC_IDENTITY_USER_NAME", "TestUser")

    config = ArcConfig.load()

    assert config.llm.default_model == "codellama"
    assert config.llm.base_url == "http://remote:11434"
    assert config.agent.max_iterations == 50
    assert config.identity.user_name == "TestUser"


def test_env_var_substitution():
    """${VAR} in config values gets replaced with env var values."""
    data = {"key": "${HOME}/something", "nested": {"api": "${MY_KEY}"}}

    os.environ["MY_KEY"] = "secret123"
    _substitute_env_vars(data)

    assert "something" in data["key"]
    assert data["nested"]["api"] == "secret123"

    # Cleanup
    del os.environ["MY_KEY"]


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
    override = {"b": {"c": 20, "f": 6}, "g": 7}

    _deep_merge(base, override)

    assert base == {"a": 1, "b": {"c": 20, "d": 3, "f": 6}, "e": 5, "g": 7}


def test_convert_value():
    assert _convert_value("true") is True
    assert _convert_value("false") is False
    assert _convert_value("42") == 42
    assert _convert_value("3.14") == 3.14
    assert _convert_value("hello") == "hello"


def test_get_workspace():
    config = ArcConfig()
    workspace = config.get_workspace()
    assert workspace.is_absolute()


def test_load_nonexistent_toml():
    """Loading from nonexistent files just uses defaults."""
    config = ArcConfig.load(
        project_path=Path("/nonexistent/arc.toml"),
        user_path=Path("/nonexistent/config.toml"),
    )
    # Should still work with defaults
    assert config.llm.default_provider == "ollama"