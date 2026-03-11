"""Tests for arc/tasks/agents.py — AgentDef TOML loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc.tasks.agents import load_agent_defs, save_agent_def, _parse_agent_toml
from arc.tasks.types import AgentDef


class TestLoadAgentDefs:
    def test_empty_dir(self, tmp_path):
        agents = load_agent_defs(tmp_path)
        assert agents == {}

    def test_nonexistent_dir(self, tmp_path):
        agents = load_agent_defs(tmp_path / "nope")
        assert agents == {}

    def test_loads_single_agent(self, tmp_path):
        toml_content = """
name = "researcher"
role = "Web research"
personality = "thorough"
max_concurrent = 2

[llm]
provider = "ollama"
model = "llama3.2"
"""
        (tmp_path / "researcher.toml").write_text(toml_content, encoding="utf-8")

        agents = load_agent_defs(tmp_path)
        assert "researcher" in agents

        a = agents["researcher"]
        assert a.name == "researcher"
        assert a.role == "Web research"
        assert a.personality == "thorough"
        assert a.max_concurrent == 2
        assert a.llm_provider == "ollama"
        assert a.llm_model == "llama3.2"
        assert a.has_llm_override

    def test_loads_multiple_agents(self, tmp_path):
        (tmp_path / "a.toml").write_text('name = "a"\nrole = "role a"', encoding="utf-8")
        (tmp_path / "b.toml").write_text('name = "b"\nrole = "role b"', encoding="utf-8")

        agents = load_agent_defs(tmp_path)
        assert len(agents) == 2
        assert "a" in agents
        assert "b" in agents

    def test_skips_invalid_toml(self, tmp_path):
        (tmp_path / "good.toml").write_text('name = "good"\nrole = "ok"', encoding="utf-8")
        (tmp_path / "bad.toml").write_text("this is not valid toml {{{{", encoding="utf-8")

        agents = load_agent_defs(tmp_path)
        assert len(agents) == 1
        assert "good" in agents

    def test_defaults_name_to_stem(self, tmp_path):
        (tmp_path / "myagent.toml").write_text('role = "test"', encoding="utf-8")

        agents = load_agent_defs(tmp_path)
        assert "myagent" in agents

    def test_no_llm_section(self, tmp_path):
        (tmp_path / "simple.toml").write_text(
            'name = "simple"\nrole = "test"\npersonality = "friendly"',
            encoding="utf-8",
        )

        agents = load_agent_defs(tmp_path)
        a = agents["simple"]
        assert not a.has_llm_override
        assert a.llm_provider == ""
        assert a.llm_model == ""

    def test_skill_whitelist(self, tmp_path):
        (tmp_path / "restricted.toml").write_text(
            'name = "restricted"\nrole = "r"\nskills = ["browsing", "filesystem"]',
            encoding="utf-8",
        )
        agents = load_agent_defs(tmp_path)
        assert agents["restricted"].skills == ["browsing", "filesystem"]

    def test_skill_blacklist(self, tmp_path):
        (tmp_path / "loose.toml").write_text(
            'name = "loose"\nrole = "r"\nexclude_skills = ["terminal"]',
            encoding="utf-8",
        )
        agents = load_agent_defs(tmp_path)
        assert agents["loose"].exclude_skills == ["terminal"]


class TestSaveAgentDef:
    def test_save_and_reload(self, tmp_path):
        agent = AgentDef(
            name="writer",
            role="Content creation",
            personality="creative, engaging",
            llm_provider="openai",
            llm_model="gpt-4o",
            max_concurrent=2,
        )

        path = save_agent_def(agent, agents_dir=tmp_path)
        assert path.exists()
        assert path.name == "writer.toml"

        # Reload and verify
        agents = load_agent_defs(tmp_path)
        assert "writer" in agents
        restored = agents["writer"]
        assert restored.role == "Content creation"
        assert restored.personality == "creative, engaging"
        assert restored.llm_provider == "openai"
        assert restored.llm_model == "gpt-4o"
        assert restored.max_concurrent == 2

    def test_save_minimal_agent(self, tmp_path):
        agent = AgentDef(name="minimal", role="test")
        path = save_agent_def(agent, agents_dir=tmp_path)
        assert path.exists()

        agents = load_agent_defs(tmp_path)
        assert "minimal" in agents
        assert not agents["minimal"].has_llm_override

    def test_save_with_skills(self, tmp_path):
        agent = AgentDef(
            name="strict",
            role="test",
            skills=["browsing", "filesystem"],
        )
        save_agent_def(agent, agents_dir=tmp_path)
        agents = load_agent_defs(tmp_path)
        assert agents["strict"].skills == ["browsing", "filesystem"]

    def test_save_and_reload_system_prompt(self, tmp_path):
        prompt = (
            "You are a senior researcher.\n\n"
            "## Protocol\n"
            "1. Search 3 sources\n"
            "2. Cross-reference claims\n"
            "3. Cite with URLs"
        )
        agent = AgentDef(
            name="prompted",
            role="test",
            system_prompt=prompt,
        )
        save_agent_def(agent, agents_dir=tmp_path)
        agents = load_agent_defs(tmp_path)
        assert agents["prompted"].system_prompt == prompt

    def test_system_prompt_from_toml_inline(self, tmp_path):
        toml_content = '''
name = "custom"
role = "test"

system_prompt = """
You are a focused analyst.
Always use bullet points.
Cite your sources.
"""
'''
        (tmp_path / "custom.toml").write_text(toml_content, encoding="utf-8")
        agents = load_agent_defs(tmp_path)
        assert "focused analyst" in agents["custom"].system_prompt
        assert "bullet points" in agents["custom"].system_prompt

    def test_system_prompt_used_in_build(self, tmp_path):
        toml_content = '''
name = "expert"
role = "test"
system_prompt = "You are the world's best coder."
'''
        (tmp_path / "expert.toml").write_text(toml_content, encoding="utf-8")
        agents = load_agent_defs(tmp_path)
        # build_system_prompt should return the custom prompt, not auto-generated
        prompt = agents["expert"].build_system_prompt()
        assert prompt == "You are the world's best coder."
