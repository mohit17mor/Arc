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
import uuid
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
        self._ws_sources: dict[web.WebSocketResponse, str] = {}
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
        self._workflow_skill: Any = None

        # Kernel reference for event subscriptions
        self._kernel: Any = None

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

    def set_workflow_skill(self, skill: Any) -> None:
        self._workflow_skill = skill

    def set_kernel(self, kernel: Any) -> None:
        self._kernel = kernel

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

        # Clean up any stale/closed connections before adding new one
        stale = {c for c in self._clients if c.closed}
        if stale:
            self._clients -= stale
            logger.debug(f"Cleaned {len(stale)} stale WebSocket connection(s)")

        self._clients.add(ws)
        logger.info(f"WebSocket client connected ({len(self._clients)} total)")
        self._ws_sources[ws] = "webchat"

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
            self._ws_sources.pop(ws, None)
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
        source = data.get("source") or self._ws_sources.get(ws, "webchat")
        if data.get("source"):
            self._ws_sources[ws] = source

        if msg_type == "message":
            content = data.get("content", "").strip()
            if not content:
                return

            # Handle slash commands (quick — OK to await)
            if content.startswith("/"):
                await self._handle_command(ws, content)
                return

            # Run chat in background so WS message loop stays alive
            # for heartbeats during long agent runs
            asyncio.create_task(
                self._handle_chat(ws, content, source),
                name="chat-message",
            )

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
        source: str,
    ) -> None:
        """Stream an agent response back to the WebSocket client."""
        if not self._handler:
            await ws.send_json({"type": "error", "message": "Agent not ready"})
            return

        self._total_messages += 1

        message_id = uuid.uuid4().hex

        await self._broadcast_user_message(
            source=source,
            user_input=user_input,
            message_id=message_id,
            exclude_ws=ws,
        )

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
            error_text = f"\n\nSorry, something went wrong: {e}"
            full_response += error_text
            # Send the error as a chunk so the user sees it inline
            try:
                await ws.send_json({
                    "type": "chunk",
                    "content": error_text,
                })
            except Exception:
                pass  # WebSocket may have closed

        # Always signal completion — even after errors.
        # This ensures WebChat exits "thinking" state and the exchange
        # is recorded in history so the conversation stays coherent.
        await ws.send_json({
            "type": "done",
            "full_content": full_response,
        })

        # Record in history so new connections see it
        self._record_history("webchat", user_input, full_response)

        # Broadcast to other connected clients (sync)
        await self._broadcast_response(
            source=source,
            user_input=user_input,
            response=full_response,
            message_id=message_id,
            exclude_ws=ws,
        )

    async def _broadcast_payload(
        self,
        payload: dict[str, Any],
        exclude_ws: web.WebSocketResponse | None = None,
    ) -> None:
        for other_ws in list(self._clients):
            if other_ws is exclude_ws:
                continue
            if other_ws.closed:
                self._clients.discard(other_ws)
                self._ws_sources.pop(other_ws, None)
                continue
            try:
                await other_ws.send_json(payload)
            except Exception:
                self._clients.discard(other_ws)
                self._ws_sources.pop(other_ws, None)

    async def _broadcast_user_message(
        self,
        source: str,
        user_input: str,
        message_id: str,
        exclude_ws: web.WebSocketResponse | None = None,
    ) -> None:
        payload = {
            "type": "sync_user",
            "source": source,
            "user_input": user_input,
            "message_id": message_id,
        }
        await self._broadcast_payload(payload, exclude_ws)

    async def _broadcast_response(
        self,
        source: str,
        user_input: str,
        response: str,
        message_id: str,
        exclude_ws: web.WebSocketResponse | None = None,
    ) -> None:
        payload = {
            "type": "sync",
            "source": source,
            "user_input": user_input,
            "response": response,
            "message_id": message_id,
        }
        await self._broadcast_payload(payload, exclude_ws)

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

    async def broadcast_notification(self, notification: Any) -> None:
        """
        Broadcast a job/worker completion notification to all WebSocket clients.

        Called by GatewayChannel when a scheduled job or worker finishes.
        Appears as a system message in WebChat.
        """
        message = {
            "type": "notification",
            "job_name": notification.job_name,
            "content": notification.content,
            "fired_at": notification.fired_at,
        }
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
                "  /workflow — List or run workflows\n"
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
                        entries = await self._memory_manager.list_episodic(limit=10)
                        if not entries:
                            await self._send_system(ws, "No episodic memories yet")
                        else:
                            import datetime
                            lines = ["Recent episodic memories:\n"]
                            for entry in entries:
                                ts = datetime.datetime.fromtimestamp(entry.created_at).strftime("%Y-%m-%d %H:%M")
                                content = entry.content
                                if len(content) > 100:
                                    content = content[:97] + "..."
                                lines.append(f"  [{ts}] {content}")
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
                                conf = getattr(fact, "confidence", 1.0)
                                lines.append(f"  [{fact.id}] (conf={conf:.2f}) {fact.content}")
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
                        from arc.scheduler.triggers import make_trigger
                        import datetime
                        jobs = await self._scheduler_store.get_all()
                        if not jobs:
                            await self._send_system(ws, "No scheduled jobs")
                        else:
                            lines = ["Scheduled jobs:\n"]
                            for job in jobs:
                                status = "active" if job.active else "inactive"
                                trigger = make_trigger(job.trigger)
                                next_dt = (
                                    datetime.datetime.fromtimestamp(job.next_run).strftime("%Y-%m-%d %H:%M")
                                    if job.next_run > 0 else "—"
                                )
                                lines.append(f"  {job.name} ({status}) — {trigger.description} — next: {next_dt}")
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

        elif cmd.startswith("/workflow"):
            if not self._workflow_skill:
                await self._send_system(ws, "Workflow engine not available")
            else:
                sub = parts[1] if len(parts) > 1 else ""
                if not sub or sub == "list":
                    # List workflows
                    result = await self._workflow_skill.execute_tool("list_workflows", {})
                    await self._send_system(ws, result.output or "No workflows found")
                else:
                    # Run workflow as background task so the WS message loop
                    # stays alive for heartbeats (prevents mid-workflow disconnects)
                    wf_name = sub
                    context_str = parts[2] if len(parts) > 2 else ""
                    asyncio.create_task(
                        self._run_workflow_streaming(ws, wf_name, context_str),
                        name=f"workflow-{wf_name}",
                    )

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

    async def _run_workflow_streaming(
        self,
        ws: web.WebSocketResponse,
        wf_name: str,
        context: str,
    ) -> None:
        """Run a workflow and send events + agent text to client in order.

        Subscribes to kernel workflow events so they arrive at the client
        in the correct sequence relative to the agent's streamed text.
        Events are sent as ``workflow_event`` messages, agent text as
        ``workflow_progress`` — all on the same WebSocket, in order.
        """
        from arc.core.events import Event, EventType

        # Temporary event handler — sends workflow events to this specific client
        async def _on_workflow_event(event: Event) -> None:
            if ws.closed:
                return
            try:
                await ws.send_json({
                    "type": "workflow_event",
                    "event": event.type,
                    "data": event.data,
                })
            except Exception:
                pass

        # Subscribe to workflow events for the duration of this run
        kernel = getattr(self, "_kernel", None)
        subscribed = False
        if kernel:
            kernel.on(EventType.WORKFLOW_START, _on_workflow_event)
            kernel.on(EventType.WORKFLOW_STEP_START, _on_workflow_event)
            kernel.on(EventType.WORKFLOW_STEP_COMPLETE, _on_workflow_event)
            kernel.on(EventType.WORKFLOW_STEP_FAILED, _on_workflow_event)
            kernel.on(EventType.WORKFLOW_COMPLETE, _on_workflow_event)
            kernel.on(EventType.WORKFLOW_PAUSED, _on_workflow_event)
            kernel.on(EventType.WORKFLOW_WAITING_INPUT, _on_workflow_event)
            subscribed = True

        try:
            async for chunk in self._workflow_skill.stream_workflow(wf_name, context):
                if ws.closed:
                    logger.info("WebSocket closed mid-workflow, stopping output")
                    break
                try:
                    await ws.send_json({
                        "type": "workflow_progress",
                        "content": chunk,
                    })
                except (ConnectionResetError, Exception):
                    logger.info("WebSocket send failed mid-workflow, stopping output")
                    break
        except Exception as e:
            logger.error(f"Workflow streaming error: {e}", exc_info=True)
            if not ws.closed:
                try:
                    await self._send_system(ws, f"Workflow error: {e}")
                except Exception:
                    pass
        finally:
            # Unsubscribe
            if subscribed and kernel:
                kernel.off(EventType.WORKFLOW_START, _on_workflow_event)
                kernel.off(EventType.WORKFLOW_STEP_START, _on_workflow_event)
                kernel.off(EventType.WORKFLOW_STEP_COMPLETE, _on_workflow_event)
                kernel.off(EventType.WORKFLOW_STEP_FAILED, _on_workflow_event)
                kernel.off(EventType.WORKFLOW_COMPLETE, _on_workflow_event)
                kernel.off(EventType.WORKFLOW_PAUSED, _on_workflow_event)
                kernel.off(EventType.WORKFLOW_WAITING_INPUT, _on_workflow_event)

    # ━━━ Fallback HTML ━━━

    @staticmethod
    def _fallback_html() -> str:
        """Minimal WebChat if the template file is missing."""
        return (
            "<!DOCTYPE html><html><head><title>Arc Gateway</title></head>"
            "<body><h1>Arc Gateway</h1><p>WebChat template not found. "
            "Check arc/gateway/templates/webchat.html</p></body></html>"
        )
