"""Tests for soul manager."""

import pytest
from pathlib import Path
from arc.identity.soul import SoulManager


@pytest.fixture
def soul(tmp_path: Path):
    """SoulManager with temp path."""
    return SoulManager(tmp_path / "identity.md")


def test_exists_false(soul):
    """exists() returns False for new soul."""
    assert soul.exists() is False


def test_create_and_exists(soul):
    """create() makes exists() return True."""
    soul.create("Friday", "Alex", "sarcastic")
    assert soul.exists() is True


def test_create_content(soul, tmp_path):
    """create() writes correct content."""
    soul.create("Friday", "Alex", "sarcastic")

    content = (tmp_path / "identity.md").read_text()
    assert "Friday" in content
    assert "Alex" in content
    assert "sarcastic" in content


def test_load_default(soul):
    """load() returns defaults when no file exists."""
    identity = soul.load()
    assert identity["agent_name"] == "Arc"
    assert identity["user_name"] == "User"
    assert identity["personality_id"] == "helpful"
    assert "system_prompt" in identity


def test_load_created(soul):
    """load() returns created values."""
    soul.create("Friday", "Alex", "sarcastic")
    identity = soul.load()

    assert identity["agent_name"] == "Friday"
    assert identity["user_name"] == "Alex"
    assert identity["personality_id"] == "sarcastic"


def test_system_prompt(soul):
    """System prompt includes identity info."""
    soul.create("Friday", "Alex", "sarcastic")
    identity = soul.load()

    prompt = identity["system_prompt"]
    assert "Friday" in prompt
    assert "Alex" in prompt


def test_get_system_prompt(soul):
    """get_system_prompt() returns working prompt."""
    soul.create("Friday", "Alex", "helpful")
    prompt = soul.get_system_prompt()

    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_load_custom_personality_uses_how_i_behave_section(soul, tmp_path):
    """When personality is custom, system prompt comes from identity.md content."""
    custom_prompt = "You are strict and concise.\nNever use emojis.\nPrefer numbered lists."
    content = f"""# Friday's Soul

## Identity
name: Friday
created: 2026-03-13
personality: custom

## My Human
user_name: Alex

## How I Behave
{custom_prompt}

## Things I've Learned About Alex
(This section grows as we interact)
"""
    (tmp_path / "identity.md").write_text(content, encoding="utf-8")

    identity = soul.load()
    prompt = identity["system_prompt"]

    assert identity["personality_id"] == "custom"
    assert "Friday" in prompt
    assert "Alex" in prompt
    assert "strict and concise" in prompt
    assert "Never use emojis." in prompt
