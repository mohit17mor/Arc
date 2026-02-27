"""
Arc Configuration — loads and merges config from multiple sources.

Precedence (highest to lowest):
1. Explicit overrides (passed in code)
2. Environment variables (ARC_*)
3. Project config (./arc.toml)
4. User config (~/.arc/config.toml)
5. Defaults (hardcoded)

Environment variable mapping:
    ARC_LLM_PROVIDER → llm.default_provider
    ARC_LLM_MODEL → llm.default_model
    ARC_LLM_BASE_URL → llm.base_url
    ARC_LLM_API_KEY → llm.api_key
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from arc.core.errors import ConfigError

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config Sub-Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AgentConfig(BaseModel):
    """Agent behavior configuration."""

    name: str = "arc"
    max_iterations: int = 25
    tool_timeout: int = 120
    temperature: float = 0.7
    context_ratio: float = 0.75
    recent_window: int = 20


class SecurityConfig(BaseModel):
    """Security policy configuration."""

    auto_allow: list[str] = Field(default_factory=lambda: ["file:read"])
    always_ask: list[str] = Field(
        default_factory=lambda: ["file:write", "file:delete", "shell:exec"]
    )
    never_allow: list[str] = Field(default_factory=list)
    workspace: str = "."
    audit_enabled: bool = True


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    backend: str = "sqlite"
    path: str = "~/.arc/memory"
    enable_long_term: bool = True
    enable_episodic: bool = True
    embedding_provider: str = "local"


class CostConfig(BaseModel):
    """Cost tracking configuration."""

    enabled: bool = True
    session_limit_usd: float = 5.0
    daily_limit_usd: float = 50.0
    warn_at_percent: float = 0.8


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    default_provider: str = "ollama"
    default_model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class ShellConfig(BaseModel):
    """Shell provider configuration."""

    provider: str = "auto"
    default_shell: str | None = None
    timeout: int = 30


class IdentityConfig(BaseModel):
    """Identity and personality configuration."""

    path: str = "~/.arc/identity.md"
    personality: str = "helpful"
    user_name: str | None = None
    agent_name: str = "Arc"


class TelegramConfig(BaseModel):
    """Telegram bot notification configuration."""

    token: str = ""
    chat_id: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""

    enabled: bool = True
    db_path: str = "~/.arc/scheduler.db"
    poll_interval: int = 30  # seconds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ArcConfig(BaseModel):
    """Root configuration for Arc."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @staticmethod
    def load(
        overrides: dict[str, Any] | None = None,
        project_path: Path | None = None,
        user_path: Path | None = None,
    ) -> ArcConfig:
        """
        Load configuration from all sources and merge.

        Precedence: overrides > env vars > project toml > user toml > defaults
        """
        # Start with defaults
        merged: dict[str, Any] = {}

        # Layer 1: User config (~/.arc/config.toml)
        user_config_path = user_path or Path.home() / ".arc" / "config.toml"
        if user_config_path.exists():
            user_data = _load_toml(user_config_path)
            _deep_merge(merged, user_data)

        # Layer 2: Project config (./arc.toml)
        project_config_path = project_path or Path.cwd() / "arc.toml"
        if project_config_path.exists():
            project_data = _load_toml(project_config_path)
            _deep_merge(merged, project_data)

        # Layer 3: Environment variables
        env_data = _load_from_env()
        _deep_merge(merged, env_data)

        # Layer 4: Explicit overrides
        if overrides:
            _deep_merge(merged, overrides)

        # Substitute ${ENV_VAR} in string values
        _substitute_env_vars(merged)

        try:
            return ArcConfig(**merged)
        except Exception as e:
            raise ConfigError(f"Invalid configuration: {e}") from e

    def get_arc_home(self) -> Path:
        """Get the Arc home directory (~/.arc)."""
        return Path(self.identity.path).expanduser().parent

    def get_workspace(self) -> Path:
        """Get the resolved workspace path."""
        return Path(self.security.workspace).resolve()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        raise ConfigError(f"Failed to load config from {path}: {e}") from e


def _load_from_env() -> dict[str, Any]:
    """Load configuration from ARC_* environment variables."""
    result: dict[str, Any] = {}

    env_mapping = {
        "ARC_LLM_PROVIDER": ("llm", "default_provider"),
        "ARC_LLM_MODEL": ("llm", "default_model"),
        "ARC_LLM_BASE_URL": ("llm", "base_url"),
        "ARC_LLM_API_KEY": ("llm", "api_key"),
        "ARC_AGENT_NAME": ("agent", "name"),
        "ARC_AGENT_MAX_ITERATIONS": ("agent", "max_iterations"),
        "ARC_AGENT_TEMPERATURE": ("agent", "temperature"),
        "ARC_SHELL_PROVIDER": ("shell", "provider"),
        "ARC_SECURITY_WORKSPACE": ("security", "workspace"),
        "ARC_IDENTITY_USER_NAME": ("identity", "user_name"),
        "ARC_IDENTITY_AGENT_NAME": ("identity", "agent_name"),
        "ARC_IDENTITY_PERSONALITY": ("identity", "personality"),
    }

    for env_var, (section, key) in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in result:
                result[section] = {}
            # Try to convert numeric values
            result[section][key] = _convert_value(value)

    return result


def _convert_value(value: str) -> Any:
    """Convert string value to appropriate type."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # String
    return value


def _deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _substitute_env_vars(data: dict) -> None:
    """Recursively substitute ${ENV_VAR} patterns in string values."""
    pattern = re.compile(r"\$\{([^}]+)\}")

    for key, value in data.items():
        if isinstance(value, dict):
            _substitute_env_vars(value)
        elif isinstance(value, str):
            matches = pattern.findall(value)
            for var_name in matches:
                env_value = os.environ.get(var_name, "")
                value = value.replace(f"${{{var_name}}}", env_value)
            data[key] = value
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    matches = pattern.findall(item)
                    for var_name in matches:
                        env_value = os.environ.get(var_name, "")
                        item = item.replace(f"${{{var_name}}}", env_value)
                    value[i] = item