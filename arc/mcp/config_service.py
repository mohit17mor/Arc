"""
Shared MCP config editing, validation, and hot reload support.

This module is the common path used by:
- Web UI JSON editor
- Agent-side MCP config tools
- Manual file edits detected via watcher
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from arc.core.config import MCPServerDef

logger = logging.getLogger(__name__)


def _default_mcp_text() -> str:
    return '{\n  "mcpServers": {}\n}\n'


def _extract_server_names(data: dict[str, Any] | None) -> list[str]:
    if not data:
        return []
    raw_servers = data.get("mcpServers", data.get("servers", {}))
    if not isinstance(raw_servers, dict):
        return []
    return sorted(str(name) for name in raw_servers.keys())


@dataclass(slots=True)
class MCPConfigState:
    path: str
    text: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("data", None)
        return data


@dataclass(slots=True)
class MCPReloadResult:
    path: str
    text: str
    valid: bool
    applied: bool
    errors: list[str] = field(default_factory=list)
    active_server_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MCPConfigStore:
    """Read, validate, and write the shared MCP config file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".arc" / "mcp.json")

    @property
    def path(self) -> Path:
        return self._path

    def read_text(self) -> str:
        if not self._path.exists():
            return _default_mcp_text()
        return self._path.read_text(encoding="utf-8")

    def inspect(self) -> MCPConfigState:
        return self.validate_text(self.read_text())

    def validate_text(self, text: str) -> MCPConfigState:
        errors: list[str] = []
        data: dict[str, Any] | None = None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})")
            return MCPConfigState(
                path=str(self._path),
                text=text,
                valid=False,
                errors=errors,
                data=None,
            )

        if not isinstance(parsed, dict):
            errors.append("Top-level MCP config must be a JSON object.")
            return MCPConfigState(
                path=str(self._path),
                text=text,
                valid=False,
                errors=errors,
                data=None,
            )

        raw_servers = parsed.get("mcpServers", parsed.get("servers", {}))
        if raw_servers is None:
            raw_servers = {}
        if not isinstance(raw_servers, dict):
            errors.append("'mcpServers' must be an object mapping server names to configs.")
            return MCPConfigState(
                path=str(self._path),
                text=text,
                valid=False,
                errors=errors,
                data=None,
            )

        for name, cfg in raw_servers.items():
            if not isinstance(cfg, dict):
                errors.append(f"Server '{name}' must be a JSON object.")
                continue
            if not cfg.get("command") and not cfg.get("url"):
                errors.append(f"Server '{name}' must define 'command' or 'url'.")
                continue
            try:
                MCPServerDef(**cfg)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = ".".join(str(part) for part in err.get("loc", ()))
                    message = err.get("msg", "invalid value")
                    if loc:
                        errors.append(f"Server '{name}' field '{loc}': {message}.")
                    else:
                        errors.append(f"Server '{name}': {message}.")

        data = parsed
        return MCPConfigState(
            path=str(self._path),
            text=text,
            valid=not errors,
            errors=errors,
            data=data,
        )

    def write_text(self, text: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(text, encoding="utf-8")


class MCPReloadCoordinator:
    """Validate, apply, and watch the shared MCP config file."""

    def __init__(
        self,
        *,
        store: MCPConfigStore,
        apply_config: Callable[[str, dict[str, Any]], Awaitable[None]],
        poll_interval: float = 1.0,
        debounce_seconds: float = 0.1,
    ) -> None:
        self._store = store
        self._apply_config = apply_config
        self._poll_interval = poll_interval
        self._debounce_seconds = debounce_seconds
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_seen_text = store.read_text()
        self._last_applied_text: str | None = None
        self._last_error = ""
        self._active_server_names: list[str] = []

    @property
    def active_server_names(self) -> list[str]:
        return list(self._active_server_names)

    @property
    def last_error(self) -> str:
        return self._last_error

    def inspect(self) -> dict[str, Any]:
        state = self._store.inspect()
        payload = state.to_dict()
        payload["active_server_names"] = self.active_server_names
        payload["last_error"] = self._last_error
        return payload

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._last_seen_text = self._store.read_text()
        self._task = asyncio.create_task(self._watch_loop(), name="mcp-config-watch")

    async def stop(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    def mark_applied(self, text: str, active_server_names: list[str]) -> None:
        self._last_seen_text = text
        self._last_applied_text = text
        self._active_server_names = list(active_server_names)
        self._last_error = ""

    async def save_and_reload(self, text: str) -> MCPReloadResult:
        state = self._store.validate_text(text)
        if not state.valid or state.data is None:
            return MCPReloadResult(
                path=state.path,
                text=state.text,
                valid=False,
                applied=False,
                errors=state.errors,
                active_server_names=self.active_server_names,
            )

        async with self._lock:
            self._store.write_text(text)
            self._last_seen_text = text
            return await self._apply_valid_state(state)

    async def reload_from_disk(self, *, reason: str = "watch") -> MCPReloadResult:
        async with self._lock:
            state = self._store.inspect()
            self._last_seen_text = state.text
            if not state.valid or state.data is None:
                self._last_error = "; ".join(state.errors)
                logger.warning(f"MCP config reload skipped ({reason}): {self._last_error}")
                return MCPReloadResult(
                    path=state.path,
                    text=state.text,
                    valid=False,
                    applied=False,
                    errors=state.errors,
                    active_server_names=self.active_server_names,
                )

            if state.text == self._last_applied_text:
                return MCPReloadResult(
                    path=state.path,
                    text=state.text,
                    valid=True,
                    applied=False,
                    errors=[],
                    active_server_names=self.active_server_names,
                )

            return await self._apply_valid_state(state)

    async def _apply_valid_state(self, state: MCPConfigState) -> MCPReloadResult:
        assert state.data is not None
        try:
            await self._apply_config(state.text, state.data)
        except Exception as exc:
            self._last_error = str(exc)
            logger.error(f"MCP config apply failed: {exc}")
            return MCPReloadResult(
                path=state.path,
                text=state.text,
                valid=True,
                applied=False,
                errors=[str(exc)],
                active_server_names=self.active_server_names,
            )

        active_names = _extract_server_names(state.data)
        self._last_applied_text = state.text
        self._active_server_names = active_names
        self._last_error = ""
        return MCPReloadResult(
            path=state.path,
            text=state.text,
            valid=True,
            applied=True,
            errors=[],
            active_server_names=active_names,
        )

    async def _watch_loop(self) -> None:
        pending_change_at: float | None = None
        while not self._stop.is_set():
            current_text = self._store.read_text()
            if current_text != self._last_seen_text:
                self._last_seen_text = current_text
                pending_change_at = time.monotonic()

            if pending_change_at is not None:
                elapsed = time.monotonic() - pending_change_at
                if elapsed >= self._debounce_seconds:
                    await self.reload_from_disk(reason="watch")
                    pending_change_at = None

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                continue
