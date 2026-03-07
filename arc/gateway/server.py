"""
Gateway Server — the central control plane for Arc.

This is the ONE process that runs everything.  All channels (WebChat,
Telegram, future Discord/Slack/WhatsApp) plug into it and share the
same AgentLoop, memory, and session.

    arc gateway

starts:
  - WebSocket + WebChat UI on localhost:18789
  - Telegram bot (if configured)
  - Scheduler engine
  - All skills, memory, security

Every channel talks to the same agent.  Conversations are in sync.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from aiohttp import web, WSMsgType

from arc.platforms.base import Platform, MessageHandler

logger = logging.getLogger(__name__)

# Where the WebChat template lives
_TEMPLATE_DIR = Path(__file__).parent / "templates"


class GatewayServer(Platform):
    """
    WebSocket + HTTP server that exposes Arc as a persistent service.

    Usage::

        gateway = GatewayServer(host="127.0.0.1", port=18789)
        await gateway.run(handle_message)   # blocks until stop()
        await gateway.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18789,
    ) -> None:
        self._host = host
        self._port = port
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._handler: MessageHandler | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._running = False
        self._stop_event = asyncio.Event()

        # Cached WebChat HTML (loaded once on first request)
        self._webchat_html: str | None = None

        # Status tracking
        self._start_time: float = 0.0
        self._total_messages: int = 0

        # Optional dependencies for slash commands
        self._skill_manager: Any = None
        self._cost_tracker: dict = {}
        self._memory_manager: Any = None
        self._scheduler_store: Any = None
        self._mcp_manager: Any = None
        self._session_memory: Any = None

        # Channels that plug into this gateway (Telegram, etc.)
        self._channels: list[Platform] = []
        self._channel_tasks: list[asyncio.Task] = []

        # Chat history — stores recent exchanges for replay on new connections
        # Each entry: {"source": "webchat"|"telegram"|..., "user_input": str, "response": str}
        self._history: list[dict[str, str]] = []
        self._max_history: int = 50  # keep last 50 exchanges

    @property
    def name(self) -> str:
        return "gateway"

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ━━━ Dependency setters ━━━

    def set_skill_manager(self, mgr: Any) -> None:
        self._skill_manager = mgr

    def set_cost_tracker(self, tracker: dict) -> None:
        self._cost_tracker = tracker

    def set_memory_manager(self, mgr: Any) -> None:
        self._memory_manager = mgr

    def set_scheduler_store(self, store: Any) -> None:
        self._scheduler_store = store

    def set_mcp_manager(self, mgr: Any) -> None:
        self._mcp_manager = mgr

    def set_session_memory(self, mem: Any) -> None:
        self._session_memory = mem

    def attach_channel(self, channel: Platform) -> None:
        """
        Attach a channel platform to this Gateway.

        When run() is called, all attached channels start concurrently
        and share the same message handler (same agent, same memory).

        Usage::

            gw.attach_channel(telegram_platform)
            gw.attach_channel(discord_platform)   # future
            await gw.run(handler)
        """
        self._channels.append(channel)
        logger.info(f"Channel attached: {channel.name}")

    @property
    def active_channels(self) -> list[str]:
        """Names of all attached channels."""
        return [ch.name for ch in self._channels]

    # ━━━ Platform interface ━━━

    async def run(self, handler: MessageHandler) -> None:
        """
        Start the Gateway and ALL attached channels, then block.

        The WebSocket/WebChat server always runs.  Attached channels
        (Telegram, etc.) run concurrently in the same process, sharing
        the same handler → same agent → same memory.
        """
        self._handler = handler
        self._start_time = time.time()

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/status", self._handle_status)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        self._running = True
        logger.info(f"Gateway started at {self.url}")

        # Start all attached channels concurrently (same handler = same agent)
        for channel in self._channels:
            task = asyncio.create_task(
                self._run_channel(channel, handler),
                name=f"channel-{channel.name}",
            )
            self._channel_tasks.append(task)
            logger.info(f"Channel started: {channel.name}")

        # Block until stop() is called
        await self._stop_event.wait()

    async def stop(self) -> None:
        """Shut down the Gateway and all attached channels gracefully."""
        self._running = False

        # Stop all attached channels
        for channel in self._channels:
            try:
                await channel.stop()
                logger.info(f"Channel stopped: {channel.name}")
            except Exception as e:
                logger.warning(f"Error stopping channel {channel.name}: {e}")

        # Cancel channel tasks
        for task in self._channel_tasks:
            task.cancel()
        for task in self._channel_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._channel_tasks.clear()

        # Close all WebSocket connections
        for ws in list(self._clients):
            try:
                await ws.close(code=1001, message=b"Gateway shutting down")
            except Exception:
                pass
        self._clients.clear()

        if self._site:
            await self._site.stop()
            self._site = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._stop_event.set()
        logger.info("Gateway stopped")

    async def _run_channel(self, channel: Platform, handler: MessageHandler) -> None:
        """Run a single channel platform with cross-platform sync."""

        async def synced_handler(user_input: str):
            """Wraps the real handler — yields chunks to the channel,
            then broadcasts the full exchange to WebChat clients."""
            full_response = ""
            async for chunk in handler(user_input):
                full_response += chunk
                yield chunk

            # Broadcast to all WebChat clients so they see cross-platform messages
            for ws in list(self._clients):
                if not ws.closed:
                    try:
                        await ws.send_json({
                            "type": "sync",
                            "source": channel.name,
                            "user_input": user_input,
                            "response": full_response,
                        })
                    except Exception:
                        self._clients.discard(ws)

            # Record in history so new connections see it
            self._record_history(channel.name, user_input, full_response)

        try:
            await channel.run(synced_handler)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Channel {channel.name} crashed: {e}", exc_info=True)

    # ━━━ HTTP handlers ━━━

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the WebChat UI."""
        if self._webchat_html is None:
            template_path = _TEMPLATE_DIR / "webchat.html"
            if template_path.exists():
                self._webchat_html = template_path.read_text(encoding="utf-8")
            else:
                self._webchat_html = self._fallback_html()

        return web.Response(
            text=self._webchat_html,
            content_type="text/html",
        )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check — returns 200 OK."""
        return web.json_response({"status": "ok", "uptime_s": int(time.time() - self._start_time)})

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Status endpoint — returns session info."""
        return web.json_response({
            "status": "running" if self._running else "stopped",
            "uptime_s": int(time.time() - self._start_time),
            "connected_clients": len(self._clients),
            "total_messages": self._total_messages,
        })

    # ━━━ WebSocket handler ━━━

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection — one per client."""
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)

        self._clients.add(ws)
        logger.info(f"WebSocket client connected ({len(self._clients)} total)")

        # Send welcome
        await ws.send_json({
            "type": "connected",
            "message": "Connected to Arc Gateway",
        })

        # Send chat history so new tabs see previous conversation
        if self._history:
            await ws.send_json({
                "type": "history",
                "messages": self._history,
            })

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._process_ws_message(ws, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"WebSocket error: {ws.exception()}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"WebSocket handler error: {e}")
        finally:
            self._clients.discard(ws)
            logger.info(f"WebSocket client disconnected ({len(self._clients)} total)")

        return ws

    async def _process_ws_message(
        self,
        ws: web.WebSocketResponse,
        raw: str,
    ) -> None:
        """Parse and route a single WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Treat plain text as a chat message
            data = {"type": "message", "content": raw}

        msg_type = data.get("type", "message")

        if msg_type == "message":
            content = data.get("content", "").strip()
            if not content:
                return

            # Handle slash commands
            if content.startswith("/"):
                await self._handle_command(ws, content)
                return

            await self._handle_chat(ws, content)

        elif msg_type == "ping":
            await ws.send_json({"type": "pong"})

        elif msg_type == "status":
            await ws.send_json({
                "type": "status",
                "connected_clients": len(self._clients),
                "total_messages": self._total_messages,
            })

    async def _handle_chat(
        self,
        ws: web.WebSocketResponse,
        user_input: str,
    ) -> None:
        """Stream an agent response back to the WebSocket client."""
        if not self._handler:
            await ws.send_json({"type": "error", "message": "Agent not ready"})
            return

        self._total_messages += 1

        # Signal that we're thinking
        await ws.send_json({"type": "thinking"})

        # Collect full response while streaming chunks
        full_response = ""
        try:
            async for chunk in self._handler(user_input):
                full_response += chunk
                await ws.send_json({
                    "type": "chunk",
                    "content": chunk,
                })
        except Exception as e:
            logger.error(f"Agent error: {e}")
            await ws.send_json({
                "type": "error",
                "message": f"Agent error: {str(e)}",
            })
            return

        # Signal completion
        await ws.send_json({
            "type": "done",
            "full_content": full_response,
        })

        # Record in history so new connections see it
        self._record_history("webchat", user_input, full_response)

        # Broadcast to other connected clients (sync)
        for other_ws in self._clients:
            if other_ws is not ws and not other_ws.closed:
                try:
                    await other_ws.send_json({
                        "type": "sync",
                        "user_input": user_input,
                        "response": full_response,
                    })
                except Exception:
                    pass  # Client may have disconnected

    # ━━━ Event forwarding ━━━

    async def broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Broadcast an event to all connected WebSocket clients.

        Called from the kernel event bus — lets WebChat show tool calls,
        thinking status, worker updates, etc.
        """
        message = {"type": "event", "event": event_type, "data": data}
        for ws in list(self._clients):
            if not ws.closed:
                try:
                    await ws.send_json(message)
                except Exception:
                    self._clients.discard(ws)

    # ━━━ Slash commands ━━━

    async def _handle_command(self, ws: web.WebSocketResponse, command: str) -> None:
        """Handle a /command from the WebChat client."""
        cmd = command.lower().strip()
        parts = command.strip().split(maxsplit=2)

        if cmd in ("/help", "/h", "/?"):
            await self._send_system(ws,
                "Commands:\n"
                "  /help     — Show this help\n"
                "  /skills   — List available skills and tools\n"
                "  /cost     — Show token usage and cost\n"
                "  /memory   — Show core facts (long-term memory)\n"
                "  /jobs     — List scheduled jobs\n"
                "  /mcp      — Show MCP server status\n"
                "  /clear    — Clear conversation history\n"
                "  /status   — Connection and session info"
            )

        elif cmd == "/cost":
            if not self._cost_tracker:
                await self._send_system(ws, "Cost tracking not available")
            else:
                s = self._cost_tracker
                text = (
                    f"Session Cost\n"
                    f"  Requests:      {s.get('requests', 0)}\n"
                    f"  Input tokens:  {s.get('input_tokens', 0):,}\n"
                    f"  Output tokens: {s.get('output_tokens', 0):,}\n"
                    f"  Total tokens:  {s.get('total_tokens', 0):,}\n"
                    f"  Cost:          ${s.get('cost_usd', 0):.4f}"
                )
                worker_total = s.get("worker_total_tokens", 0)
                if worker_total > 0:
                    text += (
                        f"\n\nWorkers\n"
                        f"  Requests:      {s.get('worker_requests', 0)}\n"
                        f"  Total tokens:  {worker_total:,}"
                    )
                text += f"\n\nGrand Total:   {s.get('grand_total_tokens', 0):,} tokens"
                await self._send_system(ws, text)

        elif cmd in ("/skills", "/skill"):
            if not self._skill_manager:
                await self._send_system(ws, "No skill manager available")
            else:
                lines = ["Skills:\n"]
                for name in sorted(self._skill_manager.skill_names):
                    manifest = self._skill_manager.get_manifest(name)
                    if manifest is None:
                        continue
                    tier = "always-on" if manifest.always_available else "on-demand"
                    tool_names = ", ".join(t.name for t in manifest.tools)
                    lines.append(f"  [{tier}] {manifest.name} v{manifest.version}")
                    lines.append(f"    Tools: {tool_names}")
                await self._send_system(ws, "\n".join(lines))

        elif cmd.startswith("/memory"):
            if not self._memory_manager:
                await self._send_system(ws, "Long-term memory not available")
            else:
                sub = parts[1].lower() if len(parts) > 1 else ""
                if sub == "forget" and len(parts) > 2:
                    fact_id = parts[2].strip()
                    try:
                        await self._memory_manager.delete_core(fact_id)
                        await self._send_system(ws, f"Deleted memory: {fact_id}")
                    except Exception as e:
                        await self._send_system(ws, f"Error: {e}")
                elif sub == "episodic":
                    try:
                        entries = await self._memory_manager.get_recent_episodic(limit=10)
                        if not entries:
                            await self._send_system(ws, "No episodic memories yet")
                        else:
                            lines = ["Recent episodic memories:\n"]
                            for entry in entries:
                                content = entry.get("content", str(entry))
                                if len(content) > 100:
                                    content = content[:97] + "..."
                                lines.append(f"  • {content}")
                            await self._send_system(ws, "\n".join(lines))
                    except Exception as e:
                        await self._send_system(ws, f"Error reading episodic: {e}")
                else:
                    try:
                        facts = await self._memory_manager.get_all_core()
                        if not facts:
                            await self._send_system(ws, "No core memories yet. Chat more to build them!")
                        else:
                            lines = ["Core facts (long-term memory):\n"]
                            for fact in facts:
                                fid = fact.get("id", "?")
                                content = fact.get("content", str(fact))
                                lines.append(f"  [{fid}] {content}")
                            lines.append(f"\n  ({len(facts)} facts total)")
                            lines.append("  Use /memory forget <id> to delete one")
                            await self._send_system(ws, "\n".join(lines))
                    except Exception as e:
                        await self._send_system(ws, f"Error reading memory: {e}")

        elif cmd.startswith("/jobs"):
            if not self._scheduler_store:
                await self._send_system(ws, "Scheduler not available")
            else:
                sub = parts[1].lower() if len(parts) > 1 else ""
                if sub == "cancel" and len(parts) > 2:
                    name_or_id = parts[2].strip()
                    try:
                        job = await self._scheduler_store.get_by_name(name_or_id)
                        if job is None:
                            all_jobs = await self._scheduler_store.get_all()
                            job = next((j for j in all_jobs if j.id == name_or_id), None)
                        if job is None:
                            await self._send_system(ws, f"No job found: {name_or_id}")
                        else:
                            await self._scheduler_store.delete(job.id)
                            await self._send_system(ws, f"Cancelled job: {job.name}")
                    except Exception as e:
                        await self._send_system(ws, f"Error: {e}")
                else:
                    try:
                        jobs = await self._scheduler_store.get_all()
                        if not jobs:
                            await self._send_system(ws, "No scheduled jobs")
                        else:
                            lines = ["Scheduled jobs:\n"]
                            for job in jobs:
                                lines.append(f"  {job.name} — {job.trigger_type}")
                            await self._send_system(ws, "\n".join(lines))
                    except Exception as e:
                        await self._send_system(ws, f"Error: {e}")

        elif cmd == "/mcp":
            if not self._mcp_manager:
                await self._send_system(ws, "No MCP servers configured")
            else:
                lines = ["MCP Servers:\n"]
                for info in self._mcp_manager.server_info():
                    status = "connected" if info["connected"] else "not connected (lazy)"
                    lines.append(f"  {info['name']} — {status} — {info['tools']} tools")
                await self._send_system(ws, "\n".join(lines))

        elif cmd == "/clear":
            if self._session_memory:
                self._session_memory.clear()
            await self._send_system(ws, "Conversation cleared")

        elif cmd == "/status":
            uptime = int(time.time() - self._start_time)
            channels = ", ".join(self.active_channels) if self.active_channels else "none"
            await self._send_system(ws,
                f"Gateway Status\n"
                f"  Uptime:     {uptime // 60}m {uptime % 60}s\n"
                f"  Clients:    {len(self._clients)} WebSocket\n"
                f"  Channels:   {channels}\n"
                f"  Messages:   {self._total_messages}"
            )

        else:
            await self._send_system(ws, f"Unknown command: {command}\nType /help for available commands")

    async def _send_system(self, ws: web.WebSocketResponse, text: str) -> None:
        """Send a system message to a single client."""
        await ws.send_json({"type": "command_result", "content": text})

    def _record_history(self, source: str, user_input: str, response: str) -> None:
        """Append an exchange to the history buffer (capped at max_history)."""
        self._history.append({
            "source": source,
            "user_input": user_input,
            "response": response,
        })
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    # ━━━ Fallback HTML ━━━

    @staticmethod
    def _fallback_html() -> str:
        """Minimal WebChat if the template file is missing."""
        return (
            "<!DOCTYPE html><html><head><title>Arc Gateway</title></head>"
            "<body><h1>Arc Gateway</h1><p>WebChat template not found. "
            "Check arc/gateway/templates/webchat.html</p></body></html>"
        )
