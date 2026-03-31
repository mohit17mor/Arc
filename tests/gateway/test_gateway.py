"""Tests for the Gateway server."""

import asyncio
import json
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from arc.gateway.server import GatewayServer
from arc.core.run_control import RunControlManager, RunControlAction, RunStatus
from arc.tasks.store import TaskStore
from arc.tasks.types import Task, TaskStatus
from arc.tasks.agents import load_agent_defs


class _FakeTurnController:
    def __init__(self, *, accepted: bool = True, interrupted: bool = False, active: bool | None = None) -> None:
        self.accepted = accepted
        self.is_active = accepted if active is None else active
        self.active_turn = None
        self.last_outcome = type("Outcome", (), {"interrupted": interrupted, "reason": "cancel" if interrupted else None})()
        self.reasons: list[str] = []

    async def interrupt_current(self, *, reason: str) -> bool:
        self.reasons.append(reason)
        return self.accepted




# ━━━ Unit tests (no HTTP) ━━━


def test_gateway_name():
    """Gateway identifies itself as 'gateway'."""
    gw = GatewayServer()
    assert gw.name == "gateway"


def test_gateway_defaults():
    """Default host/port match expected values."""
    gw = GatewayServer()
    assert gw.url == "http://127.0.0.1:18789"
    assert gw.client_count == 0


def test_gateway_custom_port():
    """Custom host/port are respected."""
    gw = GatewayServer(host="0.0.0.0", port=9999)
    assert gw.url == "http://0.0.0.0:9999"


def test_fallback_html():
    """Fallback HTML is returned when template is missing."""
    html = GatewayServer._fallback_html()
    assert "Arc Gateway" in html
    assert "<html>" in html


def test_dashboard_template_has_multistep_task_controls():
    """Dashboard exposes task-step builder controls for chained tasks."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "Add Step" in template
    assert "Reviewer" in template





def test_dashboard_template_inline_script_is_valid_javascript(tmp_path):
    """Dashboard inline script should parse so Alpine can boot."""
    import shutil
    import subprocess

    node = shutil.which("node")
    assert node, "node is required for dashboard script syntax check"

    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    start = template.index("<script>") + len("<script>")
    end = template.rindex("</script>")
    script = template[start:end]

    script_path = tmp_path / "dashboard-inline.js"
    script_path.write_text(script, encoding="utf-8")

    proc = subprocess.run([node, "--check", str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr





def test_dashboard_template_limits_plan_panel_to_main_agent_and_supports_minimize():
    """Plan panel should only render main-agent plans and expose a minimize control."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "ed.source!=='main'" in template or 'ed.source !== "main"' in template
    assert "plan-min" in template
    assert "planMin" in template
    assert "@click=\"planMin=!planMin\"" in template
    assert ":class=\"{min:planMin}\"" in template or ":class=\"{min: planMin}\"" in template


def test_dashboard_template_shows_visible_interrupt_message_in_chat():
    """Interrupted runs should show a prominent chat notice, not only a toast."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "Processing interrupted" in template
    assert ".mg.interrupt" in template


def test_dashboard_template_supports_interrupted_plan_state():
    """Dashboard plan UI can render interrupted/stale plans."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "lifecycle_status" in template
    assert "Interrupted plan" in template


def test_dashboard_template_includes_workspace_surface():
    """Chat page should expose a dedicated workspace pane and renderer hooks."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "workspace-pane" in template
    assert "workspace-root" in template
    assert "workspace:update" in template
    assert "_wu(" in template or "_renderWorkspace" in template


def test_dashboard_template_renders_detail_panel_media():
    """Detail panels should render top-level and section-level media, including local images."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "_renderWorkspaceDetailMedia" in template
    assert "Array.isArray(data.media)" in template
    assert "Array.isArray(s.media)" in template
    assert "this._workspaceImageSrc" in template


def test_dashboard_template_renders_image_thumbnails_in_tables():
    """Tables should render image URLs as thumbnails instead of raw text only."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "_isWorkspaceImageValue" in template
    assert "workspace-table-image" in template
    assert "this._workspaceImageSrc(val)" in template


def test_dashboard_template_includes_mcp_json_editor():
    """Skills & MCP page should expose an MCP JSON editor with validation/save controls."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "MCP Config Editor" in template
    assert "/api/mcp/config" in template
    assert "Save & Apply" in template
    assert "Validate" in template


def test_dashboard_template_dedupes_remote_sync_user_from_sync():
    """Remote turns should not render the same user message twice when sync_user is followed by sync."""
    template = (Path(__file__).resolve().parents[2] / "arc/gateway/templates/dashboard.html").read_text(encoding="utf-8")
    assert "pendingSyncUsers" in template
    assert "message_id" in template
    assert "this.pendingSyncUsers.add(d.message_id)" in template
    assert "this.pendingSyncUsers.has(d.message_id)" in template
    assert "this.pendingSyncUsers.delete(d.message_id)" in template


# ━━━ Integration tests (with aiohttp test server) ━━━


@pytest.fixture
async def gateway_app(tmp_path):
    """Create a Gateway wired to a mock handler and return (app, gateway)."""

    async def mock_handler(user_input: str):
        """Simulates an agent that yields chunks."""
        for word in f"Hello from Arc: {user_input}".split():
            yield word + " "

    gw = GatewayServer(host="127.0.0.1", port=0)
    gw._handler = mock_handler
    store = TaskStore(db_path=tmp_path / "gateway_tasks.db")
    await store.initialize()
    gw.set_task_store(store)

    app = web.Application()
    app.router.add_get("/", gw._handle_index)
    app.router.add_get("/ws", gw._handle_ws)
    app.router.add_get("/health", gw._handle_health)
    app.router.add_get("/status", gw._handle_status)
    app.router.add_get("/api/tasks", gw._api_list_tasks)
    app.router.add_post("/api/tasks", gw._api_create_task)
    app.router.add_get("/api/tasks/{task_id}", gw._api_get_task)
    app.router.add_post("/api/tasks/clear", gw._api_clear_tasks)
    app.router.add_post("/api/tasks/{task_id}/cancel", gw._api_cancel_task)
    app.router.add_post("/api/tasks/{task_id}/reply", gw._api_reply_task)
    app.router.add_get("/api/runs", gw._api_list_runs)
    app.router.add_post("/api/runs/{run_id}/cancel", gw._api_cancel_run)
    app.router.add_get("/api/agents", gw._api_list_agents)
    app.router.add_post("/api/agents", gw._api_create_agent)
    app.router.add_delete("/api/agents/{name}", gw._api_delete_agent)
    app.router.add_get("/api/overview", gw._api_overview)
    app.router.add_get("/api/scheduler", gw._api_list_jobs)
    app.router.add_post("/api/scheduler/{job_id}/cancel", gw._api_cancel_job)
    app.router.add_get("/api/skills", gw._api_list_skills)
    app.router.add_get("/api/mcp", gw._api_list_mcp)
    app.router.add_get("/api/mcp/config", gw._api_get_mcp_config)
    app.router.add_put("/api/mcp/config", gw._api_put_mcp_config)
    app.router.add_get("/api/logs", gw._api_get_logs)
    app.router.add_get("/api/local-image", gw._api_local_image)
    gw._running = True
    gw._start_time = time.time()

    return app, gw, store


@pytest.fixture
async def client(gateway_app):
    """Create an aiohttp test client."""
    app, gw, store = gateway_app
    async with TestClient(TestServer(app)) as c:
        yield c, gw, store
    await store.close()


async def test_health_endpoint(client):
    """GET /health returns 200 with status ok."""
    c, gw, store = client
    resp = await c.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert "uptime_s" in data


async def test_status_endpoint(client):
    """GET /status returns running status."""
    c, gw, store = client
    resp = await c.get("/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "running"
    assert data["connected_clients"] == 0
    assert data["total_messages"] == 0


async def test_webchat_page(client):
    """GET / serves the WebChat HTML."""
    c, gw, store = client
    resp = await c.get("/")
    assert resp.status == 200
    text = await resp.text()
    assert "Arc" in text
    assert "<html" in text


async def test_local_image_endpoint_serves_image_file(client, tmp_path):
    """GET /api/local-image serves local image bytes for workspace cards."""
    c, gw, store = client
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    resp = await c.get("/api/local-image", params={"path": str(image_path)})
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "image/jpeg"
    assert await resp.read() == b"fake-jpeg-bytes"


async def test_local_image_endpoint_rejects_non_image_files(client, tmp_path):
    """GET /api/local-image only serves image-like file types."""
    c, gw, store = client
    text_path = tmp_path / "note.txt"
    text_path.write_text("hello", encoding="utf-8")

    resp = await c.get("/api/local-image", params={"path": str(text_path)})
    assert resp.status == 400


async def test_websocket_connect(client):
    """WebSocket connects and receives welcome message."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        msg = await ws.receive_json()
        assert msg["type"] == "connected"
        assert "Connected" in msg["message"]
        assert gw.client_count == 1
    # After disconnect
    await asyncio.sleep(0.05)
    assert gw.client_count == 0


async def test_workspace_event_replayed_to_new_websocket_clients(client):
    """Latest workspace state should be replayed when a fresh client connects."""
    c, gw, store = client
    payload = {
        "workspace_id": "main",
        "revision": 9,
        "mode": "replace",
        "intent": "news_results",
        "title": "Latest AI News",
        "layout": "stack",
        "blocks": [],
    }

    await gw.broadcast_event("workspace:update", {"payload": payload, "source": "main"})

    async with c.ws_connect("/ws") as ws:
        connected = await ws.receive_json()
        assert connected["type"] == "connected"

        replay = await ws.receive_json()
        assert replay["type"] == "event"
        assert replay["event"] == "workspace:update"
        assert replay["data"]["payload"]["title"] == "Latest AI News"


async def test_websocket_chat(client):
    """Send a message and receive streamed chunks + done."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        # Consume welcome
        await ws.receive_json()

        # Send chat message
        await ws.send_json({"type": "message", "content": "test"})

        # Should get: thinking, chunks, done
        thinking = await ws.receive_json()
        assert thinking["type"] == "thinking"

        chunks = []
        while True:
            msg = await ws.receive_json()
            if msg["type"] == "chunk":
                chunks.append(msg["content"])
            elif msg["type"] == "done":
                break

        full = "".join(chunks)
        assert "Hello" in full
        assert "test" in full
        assert gw._total_messages == 1


async def test_websocket_ping_pong(client):
    """Ping message gets a pong response."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "ping"})
        msg = await ws.receive_json()
        assert msg["type"] == "pong"


async def test_websocket_plain_text(client):
    """Plain text (non-JSON) is treated as a chat message."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_str("hello plain text")

        thinking = await ws.receive_json()
        assert thinking["type"] == "thinking"

        while True:
            msg = await ws.receive_json()
            if msg["type"] == "done":
                assert "hello plain text" in msg["full_content"]
                break


async def test_slash_help(client):
    """/help returns command list."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/help"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "/skills" in msg["content"]
        assert "/cost" in msg["content"]
        # Should NOT trigger the agent (no thinking/chunk/done)
        assert gw._total_messages == 0


async def test_slash_status(client):
    """/status returns gateway info."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/status"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "Gateway Status" in msg["content"]
        assert "Clients:" in msg["content"]


def test_attach_channel():
    """Channels can be attached and tracked."""
    from unittest.mock import AsyncMock, PropertyMock

    gw = GatewayServer()

    # Create a mock channel
    mock_channel = AsyncMock()
    type(mock_channel).name = PropertyMock(return_value="telegram")
    mock_channel.run = AsyncMock()
    mock_channel.stop = AsyncMock()

    gw.attach_channel(mock_channel)
    assert gw.active_channels == ["telegram"]

    # Attach another
    mock_channel2 = AsyncMock()
    type(mock_channel2).name = PropertyMock(return_value="discord")
    gw.attach_channel(mock_channel2)
    assert gw.active_channels == ["telegram", "discord"]


class _FakeWebSocket:
    def __init__(self, *, closed: bool = False, fail_on_send: bool = False, fail_on_close: bool = False):
        self.closed = closed
        self.fail_on_send = fail_on_send
        self.fail_on_close = fail_on_close
        self.messages: list[dict] = []
        self.close_calls: list[tuple[int, bytes]] = []

    async def send_json(self, payload: dict) -> None:
        if self.fail_on_send:
            raise RuntimeError("send failed")
        self.messages.append(payload)

    async def close(self, *, code: int, message: bytes) -> None:
        if self.fail_on_close:
            raise RuntimeError("close failed")
        self.close_calls.append((code, message))
        self.closed = True


async def test_gateway_run_starts_server_and_attached_channels():
    async def handler(user_input: str):
        yield f"echo:{user_input}"

    channel_started = asyncio.Event()

    class _Channel:
        name = "telegram"

        def __init__(self) -> None:
            self.stop = AsyncMock()

        async def run(self, synced_handler):
            channel_started.set()
            await asyncio.Event().wait()

    gw = GatewayServer(host="127.0.0.1", port=0)
    channel = _Channel()
    gw.attach_channel(channel)

    run_task = asyncio.create_task(gw.run(handler))
    await asyncio.wait_for(channel_started.wait(), timeout=1.0)

    assert gw._running is True
    assert gw._runner is not None
    assert gw._site is not None
    assert len(gw._channel_tasks) == 1

    await gw.stop()
    await asyncio.wait_for(run_task, timeout=1.0)

    channel.stop.assert_awaited_once()
    assert gw._runner is None
    assert gw._site is None
    assert gw._stop_event.is_set()


async def test_gateway_stop_cleans_up_even_when_components_fail():
    gw = GatewayServer()

    class _Channel:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail
            self.stop = AsyncMock(side_effect=RuntimeError("boom") if fail else None)

    good_channel = _Channel("telegram")
    bad_channel = _Channel("discord", fail=True)
    gw._channels = [good_channel, bad_channel]

    blocker = asyncio.create_task(asyncio.sleep(30))
    gw._channel_tasks = [blocker]

    ws_ok = _FakeWebSocket()
    ws_bad = _FakeWebSocket(fail_on_close=True)
    gw._clients = {ws_ok, ws_bad}
    gw._site = types.SimpleNamespace(stop=AsyncMock())
    gw._runner = types.SimpleNamespace(cleanup=AsyncMock())

    await gw.stop()

    good_channel.stop.assert_awaited_once()
    bad_channel.stop.assert_awaited_once()
    assert gw._channel_tasks == []
    assert gw._clients == set()
    assert ws_ok.close_calls == [(1001, b"Gateway shutting down")]
    assert blocker.cancelled()
    assert gw._site is None
    assert gw._runner is None
    assert gw._stop_event.is_set()


async def test_run_channel_syncs_history_and_drops_failed_clients():
    gw = GatewayServer()
    ws_ok = _FakeWebSocket()
    ws_bad = _FakeWebSocket(fail_on_send=True)
    gw._clients = {ws_ok, ws_bad}

    class _Channel:
        name = "telegram"
        streamed_chunks: list[str]

        async def run(self, synced_handler):
            self.streamed_chunks = []
            async for chunk in synced_handler("status update"):
                self.streamed_chunks.append(chunk)

    async def handler(user_input: str):
        yield "chunk-1 "
        yield user_input.upper()

    channel = _Channel()
    await gw._run_channel(channel, handler)

    assert channel.streamed_chunks == ["chunk-1 ", "STATUS UPDATE"]
    assert ws_ok.messages == [
        {
            "type": "sync",
            "source": "telegram",
            "user_input": "status update",
            "response": "chunk-1 STATUS UPDATE",
        }
    ]
    assert ws_bad not in gw._clients
    assert gw._history[-1] == {
        "source": "telegram",
        "user_input": "status update",
        "response": "chunk-1 STATUS UPDATE",
    }


async def test_run_channel_logs_channel_crashes(caplog):
    gw = GatewayServer()

    class _Channel:
        name = "telegram"

        async def run(self, synced_handler):
            raise RuntimeError("channel exploded")

    async def handler(_user_input: str):
        if False:  # pragma: no cover
            yield ""

    with caplog.at_level("ERROR"):
        await gw._run_channel(_Channel(), handler)

    assert "Channel telegram crashed: channel exploded" in caplog.text


async def test_status_shows_channels(client):
    """/status shows attached channel names."""
    c, gw, store = client

    # Attach a mock channel
    from unittest.mock import AsyncMock, PropertyMock
    mock_ch = AsyncMock()
    type(mock_ch).name = PropertyMock(return_value="telegram")
    gw._channels = [mock_ch]

    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/status"})
        msg = await ws.receive_json()
        assert "telegram" in msg["content"]


async def test_slash_unknown(client):
    """/unknown returns error message."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/foobar"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "Unknown command" in msg["content"]


async def test_slash_cost_no_tracker(client):
    """/cost without tracker shows unavailable."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/cost"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "not available" in msg["content"]


async def test_slash_cost_with_tracker(client):
    """/cost with tracker shows token counts."""
    c, gw, store = client
    gw.set_cost_tracker({
        "requests": 5,
        "input_tokens": 1000,
        "output_tokens": 200,
        "total_tokens": 1200,
        "cost_usd": 0.01,
        "worker_total_tokens": 0,
        "grand_total_tokens": 1200,
        "last_input_tokens": 3456,
        "last_cached_input_tokens": 2048,
        "cached_input_tokens": 2500,
        "uncached_input_tokens": 7500,
        "context_window": 128000,
        "turn_peak_input": 4000,
    })
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/cost"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "Current Context" in msg["content"]
        assert "3,456" in msg["content"]
        assert "2,048" in msg["content"]
        assert "128,000" in msg["content"]
        assert "Grand Total" not in msg["content"]
        assert "Session Cost" not in msg["content"]
        assert "Cost:" not in msg["content"]


async def test_slash_skills_no_manager(client):
    """/skills without manager shows unavailable."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/skills"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "no skill manager" in msg["content"].lower()


async def test_slash_commands_dont_count_as_messages(client):
    """Slash commands should not increment the message counter."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/help"})
        await ws.receive_json()
        await ws.send_json({"type": "message", "content": "/status"})
        await ws.receive_json()
        assert gw._total_messages == 0


async def test_broadcast_event(client):
    """broadcast_event sends to all connected clients."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws1:
        await ws1.receive_json()  # welcome
        async with c.ws_connect("/ws") as ws2:
            await ws2.receive_json()  # welcome

            await gw.broadcast_event("skill:tool_call", {"tool": "web_search"})

            for ws in (ws1, ws2):
                msg = await ws.receive_json()
                assert msg["type"] == "event"
                assert msg["event"] == "skill:tool_call"
                assert msg["data"]["tool"] == "web_search"


async def test_empty_message_ignored(client):
    """Empty content messages are silently ignored."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": ""})
        # No response should come — use a short timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.receive_json(), timeout=0.3)
        assert gw._total_messages == 0


async def test_api_clear_tasks_deletes_terminal_tasks(client):
    c, gw, store = client
    done_task = Task(title="Done", instruction="i", status=TaskStatus.DONE)
    queued_task = Task(title="Queued", instruction="i", status=TaskStatus.QUEUED)
    await store.save(done_task)
    await store.save(queued_task)

    resp = await c.post("/api/tasks/clear", json={"task_ids": [done_task.id, queued_task.id]})
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "cleared"
    assert data["deleted"] == 1
    assert await store.get_by_id(done_task.id) is None
    assert await store.get_by_id(queued_task.id) is not None


async def test_api_create_agent_persists_full_llm_config(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    monkeypatch.setattr("arc.tasks.agents._AGENTS_DIR", agents_dir)

    gw = GatewayServer(host="127.0.0.1", port=0)
    request = AsyncMock()
    request.json = AsyncMock(
        return_value={
            "name": "researcher",
            "role": "Deep research",
            "llm_provider": "codex",
            "llm_model": "codex-mini-latest",
            "llm_base_url": "http://internal.example/v1",
            "llm_api_key": "sk-secret",
        }
    )

    resp = await gw._api_create_agent(request)
    assert resp.status == 201

    agents = load_agent_defs(agents_dir)
    assert "researcher" in agents
    agent = agents["researcher"]
    assert agent.llm_provider == "codex"
    assert agent.llm_model == "codex-mini-latest"
    assert agent.llm_base_url == "http://internal.example/v1"
    assert agent.llm_api_key == "sk-secret"


async def test_api_llm_providers_lists_all_presets():
    gw = GatewayServer(host="127.0.0.1", port=0)
    request = AsyncMock()
    resp = await gw._api_list_llm_providers(request)
    assert resp.status == 200
    data = json.loads(resp.text)

    names = [item["name"] for item in data]
    assert "codex" in names
    assert "responses" in names
    assert "openai" in names


# ━━━ record_event and event ring buffer ━━━


def test_record_event_stores_entry():
    """record_event appends to the event log."""
    gw = GatewayServer()
    gw.record_event("test:event", "unit_test", {"key": "value"})
    assert len(gw._event_log) == 1
    entry = gw._event_log[0]
    assert entry["type"] == "test:event"
    assert entry["source"] == "unit_test"
    assert entry["data"]["key"] == "value"
    assert "timestamp" in entry


def test_record_event_caps_at_max():
    """Event log caps at _max_events."""
    gw = GatewayServer()
    gw._max_events = 5
    for i in range(10):
        gw.record_event("e", "s", {"i": i})
    assert len(gw._event_log) == 5
    # Should keep the latest events
    assert gw._event_log[0]["data"]["i"] == 5
    assert gw._event_log[-1]["data"]["i"] == 9


def test_record_event_filters_large_data():
    """record_event filters out oversized or non-primitive values."""
    gw = GatewayServer()
    gw.record_event("e", "s", {
        "ok": "small",
        "big_str": "x" * 600,  # string itself is fine (it's primitive)
        "func": lambda: None,  # non-primitive, should be filtered
    })
    entry = gw._event_log[0]
    assert "ok" in entry["data"]
    # lambda is not a primitive type and should be excluded
    assert "func" not in entry["data"]


# ━━━ _record_history ━━━


def test_record_history_caps_at_max():
    """History buffer caps at _max_history."""
    gw = GatewayServer()
    gw._max_history = 3
    for i in range(5):
        gw._record_history("src", f"q{i}", f"a{i}")
    assert len(gw._history) == 3
    assert gw._history[0]["user_input"] == "q2"
    assert gw._history[-1]["user_input"] == "q4"


# ━━━ WebSocket: status message type ━━━


async def test_websocket_status_message(client):
    """Sending type=status returns current status."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "status"})
        msg = await ws.receive_json()
        assert msg["type"] == "status"
        assert "connected_clients" in msg
        assert "total_messages" in msg


# ━━━ WebSocket: history replay on connect ━━━


async def test_websocket_history_replay(client):
    """New connections receive chat history."""
    c, gw, store = client
    # Pre-populate history
    gw._record_history("webchat", "hi", "hello back")
    async with c.ws_connect("/ws") as ws:
        welcome = await ws.receive_json()
        assert welcome["type"] == "connected"
        history = await ws.receive_json()
        assert history["type"] == "history"
        assert len(history["messages"]) == 1
        assert history["messages"][0]["user_input"] == "hi"


# ━━━ WebSocket: agent error during chat ━━━


async def test_websocket_chat_agent_error(gateway_app):
    """Agent errors are sent inline as chunks and a done is still emitted."""
    app, gw, store = gateway_app

    async def failing_handler(user_input: str):
        yield "start "
        raise RuntimeError("boom")

    gw._handler = failing_handler

    async with TestClient(TestServer(app)) as c:
        async with c.ws_connect("/ws") as ws:
            await ws.receive_json()  # welcome
            await ws.send_json({"type": "message", "content": "trigger error"})
            thinking = await ws.receive_json()
            assert thinking["type"] == "thinking"

            chunks = []
            done_msg = None
            while True:
                msg = await ws.receive_json()
                if msg["type"] == "chunk":
                    chunks.append(msg["content"])
                elif msg["type"] == "done":
                    done_msg = msg
                    break

            full = "".join(chunks)
            assert "start" in full
            assert "boom" in full
            assert done_msg is not None
    await store.close()


# ━━━ REST API: GET /api/tasks ━━━


async def test_api_list_tasks_empty(client):
    """GET /api/tasks returns empty list when no tasks exist."""
    c, gw, store = client
    resp = await c.get("/api/tasks")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


async def test_api_list_tasks_with_tasks(client):
    """GET /api/tasks returns saved tasks."""
    c, gw, store = client
    t = Task(title="Test", instruction="Do it", assigned_agent="worker")
    await store.save(t)
    resp = await c.get("/api/tasks")
    assert resp.status == 200
    data = await resp.json()
    assert len(data) == 1
    assert data[0]["title"] == "Test"


async def test_api_list_tasks_filter_by_status(client):
    """GET /api/tasks?status=done filters correctly."""
    c, gw, store = client
    t1 = Task(title="Done", instruction="i", status=TaskStatus.DONE, assigned_agent="w")
    t2 = Task(title="Queued", instruction="i", status=TaskStatus.QUEUED, assigned_agent="w")
    await store.save(t1)
    await store.save(t2)
    resp = await c.get("/api/tasks?status=done")
    data = await resp.json()
    assert len(data) == 1
    assert data[0]["title"] == "Done"


# ━━━ REST API: GET /api/tasks/{task_id} ━━━


async def test_api_get_task_found(client):
    """GET /api/tasks/{id} returns task with comments."""
    c, gw, store = client
    t = Task(title="Detail", instruction="check", assigned_agent="w")
    await store.save(t)
    await store.add_comment(t.id, "worker", "Started work")
    resp = await c.get(f"/api/tasks/{t.id}")
    assert resp.status == 200
    data = await resp.json()
    assert data["title"] == "Detail"
    assert len(data["comments"]) == 1
    assert data["comments"][0]["content"] == "Started work"


async def test_api_get_task_not_found(client):
    """GET /api/tasks/{id} returns 404 for missing task."""
    c, gw, store = client
    resp = await c.get("/api/tasks/nonexistent")
    assert resp.status == 404


# ━━━ REST API: POST /api/tasks (create) ━━━


async def test_api_create_task_with_agent(client):
    """POST /api/tasks creates a task with assigned_agent."""
    c, gw, store = client
    resp = await c.post("/api/tasks", json={
        "title": "New task",
        "instruction": "Do something",
        "assigned_agent": "researcher",
    })
    assert resp.status == 201
    data = await resp.json()
    assert data["title"] == "New task"
    assert data["assigned_agent"] == "researcher"


async def test_api_create_task_with_steps(client):
    """POST /api/tasks creates multi-step task."""
    c, gw, store = client
    resp = await c.post("/api/tasks", json={
        "title": "Multi-step",
        "steps": [
            {"agent": "writer"},
            {"agent": "reviewer", "review_by": "human"},
        ],
    })
    assert resp.status == 201
    data = await resp.json()
    assert len(data["steps"]) == 2
    assert data["steps"][1]["review_by"] == "human"


async def test_api_create_task_missing_title(client):
    """POST /api/tasks rejects empty title."""
    c, gw, store = client
    resp = await c.post("/api/tasks", json={"instruction": "no title"})
    assert resp.status == 400
    data = await resp.json()
    assert "title" in data["error"]


async def test_api_create_task_missing_agent_and_steps(client):
    """POST /api/tasks rejects when neither agent nor steps given."""
    c, gw, store = client
    resp = await c.post("/api/tasks", json={"title": "No agent"})
    assert resp.status == 400
    data = await resp.json()
    assert "agent" in data["error"].lower() or "steps" in data["error"].lower()


async def test_api_create_task_invalid_json(client):
    """POST /api/tasks rejects non-JSON body."""
    c, gw, store = client
    resp = await c.post("/api/tasks", data=b"not json",
                        headers={"Content-Type": "application/json"})
    assert resp.status == 400


# ━━━ REST API: POST /api/tasks/{id}/cancel ━━━


async def test_api_cancel_task(client):
    """POST /api/tasks/{id}/cancel sets status to cancelled."""
    c, gw, store = client
    t = Task(title="Cancel me", instruction="i", assigned_agent="w")
    await store.save(t)
    resp = await c.post(f"/api/tasks/{t.id}/cancel")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "cancelled"
    # Verify in store
    updated = await store.get_by_id(t.id)
    assert updated.status == TaskStatus.CANCELLED


async def test_api_cancel_task_uses_processor_when_available(tmp_path):
    """POST /api/tasks/{id}/cancel delegates to task processor when available."""
    gw = GatewayServer()
    store = TaskStore(db_path=tmp_path / "gateway_tasks.db")
    await store.initialize()
    gw.set_task_store(store)

    t = Task(title="Cancel via processor", instruction="i", assigned_agent="w")
    await store.save(t)

    mock_proc = AsyncMock()
    mock_proc.cancel_task = AsyncMock(return_value=True)
    gw.set_task_processor(mock_proc)

    request = AsyncMock()
    request.match_info = {"task_id": t.id}
    resp = await gw._api_cancel_task(request)
    assert resp.status == 200
    data = json.loads(resp.text)
    assert data["status"] == "cancelled"
    mock_proc.cancel_task.assert_called_once_with(t.id)

    await store.close()


async def test_api_cancel_task_not_found(client):
    """POST /api/tasks/{id}/cancel returns 404 for unknown task."""
    c, gw, store = client
    resp = await c.post("/api/tasks/fake-id/cancel")
    assert resp.status == 404


async def test_api_cancel_already_done_task(client):
    """POST /api/tasks/{id}/cancel returns 404 for already-done task."""
    c, gw, store = client
    t = Task(title="Done", instruction="i", status=TaskStatus.DONE, assigned_agent="w")
    await store.save(t)
    resp = await c.post(f"/api/tasks/{t.id}/cancel")
    assert resp.status == 404


# ━━━ REST API: POST /api/tasks/{id}/reply ━━━


async def test_api_reply_task_no_processor(client):
    """POST /api/tasks/{id}/reply returns 503 without task processor."""
    c, gw, store = client
    resp = await c.post("/api/tasks/any-id/reply", json={"reply": "ok"})
    assert resp.status == 503


async def test_api_reply_task_missing_reply(client):
    """POST /api/tasks/{id}/reply rejects empty reply."""
    c, gw, store = client
    mock_proc = AsyncMock()
    gw.set_task_processor(mock_proc)
    resp = await c.post("/api/tasks/any-id/reply", json={"reply": ""})
    assert resp.status == 400


async def test_api_reply_task_success(client):
    """POST /api/tasks/{id}/reply forwards to task processor."""
    c, gw, store = client
    mock_proc = AsyncMock()
    mock_proc.handle_human_reply = AsyncMock(return_value="Reply accepted")
    gw.set_task_processor(mock_proc)
    resp = await c.post("/api/tasks/t-123/reply", json={"reply": "looks good", "action": "approve"})
    assert resp.status == 200
    data = await resp.json()
    assert data["message"] == "Reply accepted"
    mock_proc.handle_human_reply.assert_called_once_with("t-123", "looks good", "approve")


# ━━━ REST API: GET/POST /api/runs* ━━━


async def test_ws_message_rejected_while_turn_active():
    """WebSocket chat messages are rejected while a foreground turn is active."""
    gw = GatewayServer()
    gw.set_turn_controller(_FakeTurnController(active=True, accepted=True))
    ws = AsyncMock()
    gw._ws_sources[ws] = "webchat"

    await gw._process_ws_message(ws, json.dumps({"type": "message", "content": "hello"}))

    ws.send_json.assert_awaited_with({
        "type": "busy",
        "message": "A foreground turn is already active. Stop it first.",
    })


async def test_ws_interrupt_requests_active_turn():
    """WebSocket interrupt messages request foreground interruption."""
    gw = GatewayServer()
    controller = _FakeTurnController(accepted=True)
    gw.set_turn_controller(controller)
    ws = AsyncMock()
    gw._ws_sources[ws] = "webchat"

    await gw._process_ws_message(ws, json.dumps({"type": "interrupt"}))

    assert controller.reasons == ["gateway:webchat"]
    ws.send_json.assert_awaited_with({"type": "interrupt_ack", "accepted": True})


async def test_handle_chat_marks_interrupted_done_payload():
    """Gateway done payload includes interrupted status for the active turn."""
    gw = GatewayServer()
    gw.set_turn_controller(_FakeTurnController(interrupted=True))

    async def handler(user_input: str):
        yield "partial"

    gw._handler = handler
    ws = AsyncMock()

    await gw._handle_chat(ws, "hello", "webchat")

    done_payload = ws.send_json.await_args_list[-1].args[0]
    assert done_payload["type"] == "done"
    assert done_payload["interrupted"] is True
    assert done_payload["reason"] == "cancel"


async def test_api_list_runs():
    """GET /api/runs returns currently known runtime executions."""
    gw = GatewayServer()
    run_control = RunControlManager()
    gw.set_run_control(run_control)
    handle = run_control.start_run(
        kind="agent",
        source="task:t-123:researcher",
        metadata={"agent_id": "task:t-123:researcher"},
    )

    request = AsyncMock()
    request.query = {}
    resp = await gw._api_list_runs(request)
    assert resp.status == 200
    data = json.loads(resp.text)
    assert len(data) == 1
    assert data[0]["run_id"] == handle.run_id
    assert data[0]["kind"] == "agent"
    assert data[0]["status"] == "running"


async def test_api_cancel_run():
    """POST /api/runs/{id}/cancel requests cancellation for a live run."""
    gw = GatewayServer()
    run_control = RunControlManager()
    gw.set_run_control(run_control)
    handle = run_control.start_run(kind="agent", source="main")

    request = AsyncMock()
    request.match_info = {"run_id": handle.run_id}
    resp = await gw._api_cancel_run(request)
    assert resp.status == 200
    data = json.loads(resp.text)
    assert data["status"] == "cancelling"

    snapshot = run_control.get_run(handle.run_id)
    assert snapshot is not None
    assert snapshot.status == RunStatus.CANCELLING
    assert snapshot.requested_action == RunControlAction.CANCEL


# ━━━ REST API: GET /api/overview ━━━


async def test_api_overview(client):
    """GET /api/overview returns system overview."""
    c, gw, store = client
    t = Task(title="T", instruction="i", status=TaskStatus.DONE, assigned_agent="w")
    await store.save(t)
    gw.set_cost_tracker({"total_tokens": 500})
    resp = await c.get("/api/overview")
    assert resp.status == 200
    data = await resp.json()
    assert "uptime_s" in data
    assert data["connected_clients"] == 0
    assert data["tasks"]["total"] == 1
    assert data["tasks"]["by_status"]["done"] == 1
    assert data["cost"]["total_tokens"] == 500


async def test_api_overview_without_store():
    """GET /api/overview works without task store."""
    gw = GatewayServer()
    gw._start_time = time.time()
    gw._running = True
    request = AsyncMock()
    resp = await gw._api_overview(request)
    assert resp.status == 200
    data = json.loads(resp.text)
    assert "tasks" not in data


# ━━━ REST API: GET /api/agents ━━━


async def test_api_list_agents(client):
    """GET /api/agents returns configured agents."""
    c, gw, store = client
    from arc.tasks.types import AgentDef
    gw.set_agent_defs({
        "writer": AgentDef(name="writer", role="Write content"),
    })
    resp = await c.get("/api/agents")
    assert resp.status == 200
    data = await resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "writer"
    assert data[0]["role"] == "Write content"


async def test_api_list_agents_empty(client):
    """GET /api/agents returns empty when no agents defined."""
    c, gw, store = client
    resp = await c.get("/api/agents")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


# ━━━ REST API: DELETE /api/agents/{name} ━━━


async def test_api_delete_agent(client, tmp_path, monkeypatch):
    """DELETE /api/agents/{name} removes agent file."""
    c, gw, store = client
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "test_agent.toml").write_text('[agent]\nname = "test_agent"\nrole = "test"\n')
    monkeypatch.setattr("arc.tasks.agents._AGENTS_DIR", agents_dir)
    resp = await c.delete("/api/agents/test_agent")
    assert resp.status == 200
    assert not (agents_dir / "test_agent.toml").exists()


async def test_api_delete_agent_not_found(client, tmp_path, monkeypatch):
    """DELETE /api/agents/{name} returns 404 for missing agent."""
    c, gw, store = client
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    monkeypatch.setattr("arc.tasks.agents._AGENTS_DIR", agents_dir)
    resp = await c.delete("/api/agents/nonexistent")
    assert resp.status == 404


# ━━━ REST API: GET /api/skills ━━━


async def test_api_list_skills_no_manager(client):
    """GET /api/skills returns empty when no skill manager."""
    c, gw, store = client
    resp = await c.get("/api/skills")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


async def test_api_list_skills_with_manager(client):
    """GET /api/skills returns skill info."""
    c, gw, store = client
    mock_mgr = AsyncMock()
    mock_mgr.skill_names = ["browsing"]

    mock_tool = AsyncMock()
    mock_tool.name = "web_search"
    mock_tool.description = "Search the web"

    mock_manifest = AsyncMock()
    mock_manifest.name = "browsing"
    mock_manifest.version = "1.0"
    mock_manifest.description = "Web browsing"
    mock_manifest.always_available = False
    mock_manifest.tools = [mock_tool]

    mock_mgr.get_manifest = lambda name: mock_manifest if name == "browsing" else None
    gw.set_skill_manager(mock_mgr)

    resp = await c.get("/api/skills")
    data = await resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "browsing"
    assert data[0]["tools"][0]["name"] == "web_search"


# ━━━ REST API: GET /api/mcp ━━━


async def test_api_list_mcp_no_manager(client):
    """GET /api/mcp returns empty when no MCP manager."""
    c, gw, store = client
    resp = await c.get("/api/mcp")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


async def test_api_list_mcp_with_manager(client):
    """GET /api/mcp returns server info."""
    from unittest.mock import MagicMock
    c, gw, store = client
    mock_mgr = MagicMock()
    mock_mgr.server_info.return_value = [
        {"name": "github", "transport": "stdio", "connected": False, "tools": 5, "hint": ""},
    ]
    gw.set_mcp_manager(mock_mgr)
    resp = await c.get("/api/mcp")
    data = await resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "github"
    assert data[0]["tools"] == 5


async def test_api_get_mcp_config_without_service(client):
    c, gw, store = client
    resp = await c.get("/api/mcp/config")
    assert resp.status == 503


async def test_api_get_mcp_config_returns_editor_state(client):
    from unittest.mock import MagicMock

    c, gw, store = client
    mock_service = MagicMock()
    mock_service.inspect.return_value = {
        "path": "/tmp/mcp.json",
        "text": '{"mcpServers":{}}',
        "valid": True,
        "errors": [],
    }
    gw.set_mcp_config_service(mock_service)

    resp = await c.get("/api/mcp/config")

    assert resp.status == 200
    data = await resp.json()
    assert data["valid"] is True
    assert '"mcpServers"' in data["text"]


async def test_api_put_mcp_config_validates_and_applies_changes(client):
    c, gw, store = client

    class _Service:
        async def save_and_reload(self, text: str):
            assert '"github"' in text
            return {
                "path": "/tmp/mcp.json",
                "text": text,
                "valid": True,
                "applied": True,
                "errors": [],
                "active_server_names": ["github"],
            }

    gw.set_mcp_config_service(_Service())

    resp = await c.put(
        "/api/mcp/config",
        json={"text": '{"mcpServers":{"github":{"command":"npx"}}}'},
    )

    assert resp.status == 200
    data = await resp.json()
    assert data["applied"] is True
    assert data["active_server_names"] == ["github"]


async def test_api_put_mcp_config_rejects_invalid_body(client):
    c, gw, store = client

    class _Service:
        async def save_and_reload(self, text: str):
            raise AssertionError("service should not be called")

    gw.set_mcp_config_service(_Service())

    resp = await c.put("/api/mcp/config", json={"text": 123})

    assert resp.status == 400


# ━━━ REST API: GET /api/logs ━━━


async def test_api_get_logs_empty(client):
    """GET /api/logs returns empty array by default."""
    c, gw, store = client
    resp = await c.get("/api/logs")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


async def test_api_get_logs_returns_events(client):
    """GET /api/logs returns recorded events."""
    c, gw, store = client
    gw.record_event("skill:call", "agent", {"tool": "web_search"})
    gw.record_event("memory:store", "memory", {"count": 1})
    resp = await c.get("/api/logs")
    data = await resp.json()
    assert len(data) == 2


async def test_api_get_logs_filter_by_source(client):
    """GET /api/logs?source=memory filters by source."""
    c, gw, store = client
    gw.record_event("skill:call", "agent", {})
    gw.record_event("memory:store", "memory", {})
    resp = await c.get("/api/logs?source=memory")
    data = await resp.json()
    assert len(data) == 1
    assert data[0]["source"] == "memory"


async def test_api_get_logs_respects_limit(client):
    """GET /api/logs?limit=2 caps results."""
    c, gw, store = client
    for i in range(5):
        gw.record_event("e", "s", {"i": i})
    resp = await c.get("/api/logs?limit=2")
    data = await resp.json()
    assert len(data) == 2


# ━━━ REST API: GET /api/scheduler ━━━


async def test_api_list_jobs_no_store(client):
    """GET /api/scheduler returns empty when no scheduler store."""
    c, gw, store = client
    resp = await c.get("/api/scheduler")
    assert resp.status == 200
    data = await resp.json()
    assert data == []


# ━━━ REST API: POST /api/scheduler/{id}/cancel ━━━


async def test_api_cancel_job_no_store(client):
    """POST /api/scheduler/{id}/cancel returns 503 without store."""
    c, gw, store = client
    resp = await c.post("/api/scheduler/job-1/cancel")
    assert resp.status == 503


# ━━━ REST API: 503 when stores are missing ━━━


async def test_api_tasks_503_without_store(gateway_app):
    """Task APIs return 503 when task store not set."""
    app, gw, store = gateway_app
    gw._task_store = None  # Remove store

    async with TestClient(TestServer(app)) as c:
        for endpoint in ["/api/tasks", f"/api/tasks/x"]:
            resp = await c.get(endpoint)
            assert resp.status == 503

        resp = await c.post("/api/tasks", json={"title": "t", "assigned_agent": "a"})
        assert resp.status == 503

        resp = await c.post("/api/tasks/x/cancel")
        assert resp.status == 503

        resp = await c.post("/api/tasks/clear", json={"task_ids": ["x"]})
        assert resp.status == 503
    await store.close()


# ━━━ broadcast_notification ━━━


async def test_broadcast_notification(client):
    """broadcast_notification sends to all WS clients."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome

        mock_notif = AsyncMock()
        mock_notif.job_name = "daily_report"
        mock_notif.content = "Report generated"
        mock_notif.fired_at = 1234567890

        await gw.broadcast_notification(mock_notif)
        msg = await ws.receive_json()
        assert msg["type"] == "notification"
        assert msg["job_name"] == "daily_report"
        assert msg["content"] == "Report generated"


# ━━━ Slash commands: /clear ━━━


async def test_slash_clear(client):
    """/clear clears session memory if available."""
    from unittest.mock import MagicMock
    c, gw, store = client
    mock_session = MagicMock()
    gw.set_session_memory(mock_session)
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/clear"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "cleared" in msg["content"].lower()
        mock_session.clear.assert_called_once()


# ━━━ Slash commands: /memory without manager ━━━


async def test_slash_memory_no_manager(client):
    """/memory without memory manager shows unavailable."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/memory"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "not available" in msg["content"].lower()


# ━━━ Slash commands: /jobs without store ━━━


async def test_slash_jobs_no_store(client):
    """/jobs without scheduler store shows unavailable."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/jobs"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "not available" in msg["content"].lower()


# ━━━ Slash commands: /mcp without manager ━━━


async def test_slash_mcp_no_manager(client):
    """/mcp without MCP manager shows not configured."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/mcp"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "not configured" in msg["content"].lower() or "no mcp" in msg["content"].lower()


# ━━━ Slash commands: /workflow without skill ━━━


async def test_slash_workflow_no_skill(client):
    """/workflow without workflow skill shows unavailable."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/workflow"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "not available" in msg["content"].lower()


# ━━━ Dependency setters ━━━


def test_dependency_setters():
    """All dependency setters store their values."""
    gw = GatewayServer()
    mock = AsyncMock()

    gw.set_skill_manager(mock)
    assert gw._skill_manager is mock

    gw.set_memory_manager(mock)
    assert gw._memory_manager is mock

    gw.set_scheduler_store(mock)
    assert gw._scheduler_store is mock

    gw.set_mcp_manager(mock)
    assert gw._mcp_manager is mock

    gw.set_session_memory(mock)
    assert gw._session_memory is mock

    gw.set_workflow_skill(mock)
    assert gw._workflow_skill is mock

    gw.set_kernel(mock)
    assert gw._kernel is mock

    gw.set_task_processor(mock)
    assert gw._task_processor is mock

    gw.set_agent_defs({"a": mock})
    assert gw._agent_defs == {"a": mock}


# ━━━ Multiple WebSocket clients ━━━


async def test_multiple_clients_sync(client):
    """Messages from one client are synced to others."""
    c, gw, store = client
    async with c.ws_connect("/ws") as ws1:
        await ws1.receive_json()  # welcome
        async with c.ws_connect("/ws") as ws2:
            await ws2.receive_json()  # welcome
            assert gw.client_count == 2

            # ws1 sends a message
            await ws1.send_json({"type": "message", "content": "hello sync"})

            # ws2 should receive the broadcast user message
            sync_msg = await ws2.receive_json()
            assert sync_msg["type"] == "sync_user"
            assert sync_msg["user_input"] == "hello sync"

            # ws1 gets thinking + chunks + done
            thinking = await ws1.receive_json()
            assert thinking["type"] == "thinking"
            while True:
                msg = await ws1.receive_json()
                if msg["type"] == "done":
                    break

            # ws2 gets the sync response
            sync_resp = await ws2.receive_json()
            assert sync_resp["type"] == "sync"
            assert "hello sync" in sync_resp["user_input"]
