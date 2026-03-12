"""Tests for the Gateway server."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock
import time

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from arc.gateway.server import GatewayServer
from arc.tasks.store import TaskStore
from arc.tasks.types import Task, TaskStatus


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
    app.router.add_post("/api/tasks/clear", gw._api_clear_tasks)
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

        # Collect chunks
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

        # Drain chunks until done
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
    gw.set_cost_tracker({"requests": 5, "input_tokens": 1000, "output_tokens": 200,
                          "total_tokens": 1200, "cost_usd": 0.01,
                          "worker_total_tokens": 0, "grand_total_tokens": 1200})
    async with c.ws_connect("/ws") as ws:
        await ws.receive_json()  # welcome
        await ws.send_json({"type": "message", "content": "/cost"})
        msg = await ws.receive_json()
        assert msg["type"] == "command_result"
        assert "1,200" in msg["content"]
        assert "$0.01" in msg["content"]


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
