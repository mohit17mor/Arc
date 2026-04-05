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

    # Worker / sub-agent model (optional — falls back to main if empty)
    worker_provider: str = ""
    worker_model: str = ""
    worker_base_url: str = ""
    worker_api_key: str = ""

    @property
    def has_worker_override(self) -> bool:
        """True if a separate worker LLM is configured."""
        return bool(self.worker_provider and self.worker_model)


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
    """Telegram bot configuration (notifications + bidirectional chat)."""

    token: str = ""
    chat_id: str = ""  # for outbound notifications
    allowed_users: list[str] = Field(default_factory=list)  # chat_ids allowed to interact

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    @property
    def platform_configured(self) -> bool:
        """True if the bot is set up for bidirectional chat."""
        return bool(self.token)


class TavilyConfig(BaseModel):
    """Tavily search API configuration."""

    api_key: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class NgrokConfig(BaseModel):
    """Ngrok tunnel configuration for Liquid Web."""

    auth_token: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.auth_token)


class VoiceConfig(BaseModel):
    """Voice input (STT) and output (TTS) configuration."""

    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5
    whisper_model: str = "base.en"
    silence_duration: float = 1.5  # seconds of silence = end of speech
    listen_timeout: float = 30.0  # seconds before returning to sleep

    # TTS settings
    tts_provider: str = "auto"  # auto | kokoro | system
    tts_voice: str = "af_heart"  # kokoro voice name
    tts_speed: float = 1.0


class GatewayConfig(BaseModel):
    """Gateway (WebSocket + WebChat) configuration."""

    host: str = "127.0.0.1"
    port: int = 18789


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""

    enabled: bool = True
    db_path: str = "~/.arc/scheduler.db"
    poll_interval: int = 30  # seconds


class MCPServerDef(BaseModel):
    """Definition of a single MCP server."""

    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""  # for SSE/streamable-HTTP transport (alternative to command)


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration.

    Loaded from ~/.arc/mcp.json (Claude Desktop-compatible format).
    """

    servers: dict[str, MCPServerDef] = Field(default_factory=dict)

    @staticmethod
    def load_from_file(path: Path | None = None) -> "MCPConfig":
        """Load MCP server definitions from mcp.json."""
        import json as _json

        mcp_path = path or Path.home() / ".arc" / "mcp.json"
        if not mcp_path.exists():
            return MCPConfig()

        try:
            data = _json.loads(mcp_path.read_text(encoding="utf-8"))
        except Exception:
            return MCPConfig()

        # Claude Desktop format: {"mcpServers": {"name": {command, args, env}}}
        raw_servers = data.get("mcpServers", data.get("servers", {}))
        servers: dict[str, MCPServerDef] = {}
        for name, cfg in raw_servers.items():
            servers[name] = MCPServerDef(**cfg)
        return MCPConfig(servers=servers)


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
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    ngrok: NgrokConfig = Field(default_factory=NgrokConfig)

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
        "ARC_LLM_WORKER_PROVIDER": ("llm", "worker_provider"),
        "ARC_LLM_WORKER_MODEL": ("llm", "worker_model"),
        "ARC_LLM_WORKER_BASE_URL": ("llm", "worker_base_url"),
        "ARC_LLM_WORKER_API_KEY": ("llm", "worker_api_key"),
        "ARC_TAVILY_API_KEY": ("tavily", "api_key"),
        "ARC_NGROK_AUTH_TOKEN": ("ngrok", "auth_token"),
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