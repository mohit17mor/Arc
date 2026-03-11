"""
Agent definition loader — reads named agent configs from TOML files.

Scans two directories:
    ~/.arc/agents/*.toml    — user-defined agents
    (future: arc/tasks/builtin_agents/ for bundled presets)

Each file defines one agent: name, role, personality, LLM config,
skill restrictions, concurrency limits.

Example TOML::

    name = "researcher"
    role = "Deep web research and analysis"
    personality = "thorough, cites sources, structured output"
    max_concurrent = 1

    # Skill restrictions (optional — omit both for full access)
    # skills = ["browsing", "filesystem"]       # whitelist
    # exclude_skills = ["terminal"]             # blacklist

    [llm]
    provider = "ollama"
    model = "llama3.2"
    # base_url = "http://localhost:11434"
    # api_key = ""

    [memory]
    enabled = false   # future: per-agent persistent memory
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arc.tasks.types import AgentDef

logger = logging.getLogger(__name__)

# Default directory for agent definitions
_AGENTS_DIR = Path.home() / ".arc" / "agents"


def load_agent_defs(agents_dir: Path | None = None) -> dict[str, AgentDef]:
    """
    Scan a directory for *.toml agent definitions.

    Returns a dict of agent_name → AgentDef.
    Skips files that fail to parse (logged as warnings).
    """
    directory = agents_dir or _AGENTS_DIR
    if not directory.exists():
        return {}

    agents: dict[str, AgentDef] = {}
    for toml_file in sorted(directory.glob("*.toml")):
        try:
            agent = _parse_agent_toml(toml_file)
            if agent.name in agents:
                logger.warning(
                    f"Duplicate agent name '{agent.name}' in {toml_file}, "
                    f"overwriting previous definition"
                )
            agents[agent.name] = agent
            logger.debug(f"Loaded agent definition: {agent.name} from {toml_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load agent from {toml_file}: {e}")

    if agents:
        logger.info(f"Loaded {len(agents)} agent definition(s): {', '.join(agents)}")
    return agents


def _parse_agent_toml(path: Path) -> AgentDef:
    """Parse a single TOML file into an AgentDef."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # Python < 3.11

    with open(path, "rb") as f:
        data = tomllib.load(f)

    name = data.get("name", path.stem)
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid agent name: {name!r}")

    llm_section = data.get("llm", {})

    return AgentDef(
        name=name,
        role=data.get("role", ""),
        personality=data.get("personality", ""),
        system_prompt=data.get("system_prompt", "").strip(),
        skills=data.get("skills"),
        exclude_skills=data.get("exclude_skills"),
        max_concurrent=data.get("max_concurrent", 1),
        llm_provider=llm_section.get("provider", ""),
        llm_model=llm_section.get("model", ""),
        llm_base_url=llm_section.get("base_url", ""),
        llm_api_key=llm_section.get("api_key", ""),
    )


def ensure_agents_dir() -> Path:
    """Create the agents directory if it doesn't exist. Returns the path."""
    _AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    return _AGENTS_DIR


def save_agent_def(agent: AgentDef, agents_dir: Path | None = None) -> Path:
    """Write an AgentDef to a TOML file. Returns the file path."""
    directory = agents_dir or ensure_agents_dir()
    path = directory / f"{agent.name}.toml"

    lines = [
        f'name = "{agent.name}"',
        f'role = "{agent.role}"',
    ]
    if agent.personality:
        lines.append(f'personality = "{agent.personality}"')
    lines.append(f"max_concurrent = {agent.max_concurrent}")

    if agent.system_prompt:
        lines.append(f'\nsystem_prompt = """\n{agent.system_prompt}\n"""')

    if agent.skills is not None:
        skill_list = ", ".join(f'"{s}"' for s in agent.skills)
        lines.append(f"skills = [{skill_list}]")
    if agent.exclude_skills is not None:
        ex_list = ", ".join(f'"{s}"' for s in agent.exclude_skills)
        lines.append(f"exclude_skills = [{ex_list}]")

    if agent.has_llm_override:
        lines.append("")
        lines.append("[llm]")
        lines.append(f'provider = "{agent.llm_provider}"')
        lines.append(f'model = "{agent.llm_model}"')
        if agent.llm_base_url:
            lines.append(f'base_url = "{agent.llm_base_url}"')
        if agent.llm_api_key:
            lines.append(f'api_key = "{agent.llm_api_key}"')

    lines.append("")  # trailing newline
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved agent definition: {agent.name} → {path}")
    return path
