"""Tests for CLI commands."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from arc.cli.main import app


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    """arc version shows version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_init_creates_files(runner, tmp_path, monkeypatch):
    """arc init creates config and identity files."""
    # Redirect home directory
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Mock input — new setup flow:
    # 1. User name, 2. Agent name, 3. Personality (3),
    # 4. Provider (1=Ollama), 5. Base URL, 6. Model,
    # 7. Worker model? (n)
    result = runner.invoke(
        app,
        ["init"],
        input="Alex\nFriday\n3\n1\nhttp://localhost:11434\nllama3.1\nn\n",
    )

    assert result.exit_code == 0
    assert (tmp_path / ".arc" / "config.toml").exists()
    assert (tmp_path / ".arc" / "identity.md").exists()


def test_chat_without_init(runner, tmp_path, monkeypatch):
    """arc chat fails gracefully without init."""
    # Redirect to empty home
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 1
    assert "not configured" in result.stdout.lower() or "arc init" in result.stdout.lower()