"""Tests for shared MCP config editing and hot reload."""

from __future__ import annotations

import asyncio
import json

import pytest

from arc.mcp.config_service import MCPConfigStore, MCPReloadCoordinator


def _valid_config(server_name: str = "github") -> str:
    return json.dumps(
        {
            "mcpServers": {
                server_name: {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                }
            }
        },
        indent=2,
    )


def test_config_store_defaults_to_empty_config_when_file_missing(tmp_path):
    store = MCPConfigStore(tmp_path / "mcp.json")

    state = store.inspect()

    assert state.valid is True
    assert state.errors == []
    assert '"mcpServers"' in state.text


def test_config_store_reports_invalid_json(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{not json", encoding="utf-8")
    store = MCPConfigStore(path)

    state = store.inspect()

    assert state.valid is False
    assert state.errors
    assert "json" in state.errors[0].lower()


def test_config_store_validates_server_shape(tmp_path):
    store = MCPConfigStore(tmp_path / "mcp.json")

    state = store.validate_text(
        json.dumps({"mcpServers": {"broken": {"env": {"TOKEN": "x"}}}})
    )

    assert state.valid is False
    assert any("broken" in err for err in state.errors)


@pytest.mark.asyncio
async def test_reload_coordinator_save_and_reload_applies_valid_config(tmp_path):
    applied: list[dict] = []
    store = MCPConfigStore(tmp_path / "mcp.json")

    async def apply_config(raw_text: str, data: dict) -> None:
        applied.append(data)

    coordinator = MCPReloadCoordinator(store=store, apply_config=apply_config, poll_interval=0.01)

    result = await coordinator.save_and_reload(_valid_config("github"))

    assert result.applied is True
    assert result.valid is True
    assert applied[-1]["mcpServers"]["github"]["command"] == "npx"
    assert "github" in coordinator.active_server_names


@pytest.mark.asyncio
async def test_reload_coordinator_rejects_invalid_save_and_keeps_last_good(tmp_path):
    applied: list[dict] = []
    path = tmp_path / "mcp.json"
    store = MCPConfigStore(path)

    async def apply_config(raw_text: str, data: dict) -> None:
        applied.append(data)

    coordinator = MCPReloadCoordinator(store=store, apply_config=apply_config, poll_interval=0.01)
    good = _valid_config("github")
    await coordinator.save_and_reload(good)

    result = await coordinator.save_and_reload('{"mcpServers":{"broken":{}}}')

    assert result.applied is False
    assert result.valid is False
    assert len(applied) == 1
    assert path.read_text(encoding="utf-8") == good
    assert coordinator.active_server_names == ["github"]


@pytest.mark.asyncio
async def test_reload_coordinator_manual_invalid_edit_keeps_last_good_active(tmp_path):
    applied: list[dict] = []
    path = tmp_path / "mcp.json"
    store = MCPConfigStore(path)

    async def apply_config(raw_text: str, data: dict) -> None:
        applied.append(data)

    coordinator = MCPReloadCoordinator(store=store, apply_config=apply_config, poll_interval=0.01)
    await coordinator.save_and_reload(_valid_config("github"))

    path.write_text('{"mcpServers":{"broken":{}}}', encoding="utf-8")
    result = await coordinator.reload_from_disk(reason="manual-edit")

    assert result.applied is False
    assert result.valid is False
    assert len(applied) == 1
    assert coordinator.active_server_names == ["github"]
    assert coordinator.last_error


@pytest.mark.asyncio
async def test_reload_coordinator_watcher_applies_latest_valid_file_contents(tmp_path):
    applied: list[dict] = []
    path = tmp_path / "mcp.json"
    store = MCPConfigStore(path)

    async def apply_config(raw_text: str, data: dict) -> None:
        applied.append(data)

    coordinator = MCPReloadCoordinator(
        store=store,
        apply_config=apply_config,
        poll_interval=0.01,
        debounce_seconds=0.03,
    )

    await coordinator.start()
    try:
        path.write_text(_valid_config("github"), encoding="utf-8")
        await asyncio.sleep(0.01)
        path.write_text(_valid_config("filesystem"), encoding="utf-8")
        await asyncio.sleep(0.12)
    finally:
        await coordinator.stop()

    assert applied
    assert "filesystem" in applied[-1]["mcpServers"]
    assert coordinator.active_server_names == ["filesystem"]


@pytest.mark.asyncio
async def test_reload_coordinator_serializes_concurrent_save_requests(tmp_path):
    apply_started = 0
    max_in_flight = 0
    in_flight = 0
    release_first = asyncio.Event()
    store = MCPConfigStore(tmp_path / "mcp.json")

    async def apply_config(raw_text: str, data: dict) -> None:
        nonlocal apply_started, max_in_flight, in_flight
        apply_started += 1
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if apply_started == 1:
            await release_first.wait()
        in_flight -= 1

    coordinator = MCPReloadCoordinator(store=store, apply_config=apply_config, poll_interval=0.01)

    first = asyncio.create_task(coordinator.save_and_reload(_valid_config("github")))
    await asyncio.sleep(0)
    second = asyncio.create_task(coordinator.save_and_reload(_valid_config("filesystem")))
    await asyncio.sleep(0.02)
    release_first.set()
    await asyncio.gather(first, second)

    assert max_in_flight == 1
    assert coordinator.active_server_names == ["filesystem"]
