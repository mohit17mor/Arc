"""
Arc CLI entry point.

Commands:
    arc init  — First-time setup
    arc chat  — Interactive chat
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import typer
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="arc",
    help="Arc — Micro-agents you can teach, share, and compose.",
    add_completion=False,
)

console = Console()


def get_arc_home() -> Path:
    """Get the Arc home directory."""
    return Path.home() / ".arc"


def get_config_path() -> Path:
    """Get the config file path."""
    return get_arc_home() / "config.toml"


def get_identity_path() -> Path:
    """Get the identity file path."""
    return get_arc_home() / "identity.md"


@app.command()
def doctor() -> None:
    """Check whether Arc's managed runtime install looks healthy."""
    from arc.install.health import evaluate_install_health

    report = evaluate_install_health(get_arc_home())
    status = "OK" if report.ok else "Needs Attention"
    lines = [f"[bold]Status:[/bold] {status}"]

    if report.blocking_issues:
        lines.append("\n[bold red]Blocking Issues[/bold red]")
        lines.extend(f"- {issue}" for issue in report.blocking_issues)

    if report.optional_issues:
        lines.append("\n[bold yellow]Optional Issues[/bold yellow]")
        lines.extend(f"- {issue}" for issue in report.optional_issues)

    lines.append("\n[bold]Checked Paths[/bold]")
    lines.extend(f"- {name}: {path}" for name, path in report.checked_paths.items())

    console.print(Panel("\n".join(lines), title="Install Health", border_style="cyan" if report.ok else "yellow"))
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def init() -> None:
    """Initialize or reconfigure Arc — interactive setup wizard."""
    from arc.identity.setup import run_first_time_setup
    from arc.core.config import ArcConfig

    config_path = get_config_path()
    identity_path = get_identity_path()

    # Load existing config as defaults (empty dict on first run)
    existing: dict = {}
    if config_path.exists():
        try:
            cfg = ArcConfig.load(user_path=config_path)
            existing = {
                "user_name": cfg.identity.user_name or "User",
                "agent_name": cfg.identity.agent_name,
                "personality": cfg.identity.personality,
                "provider": cfg.llm.default_provider,
                "model": cfg.llm.default_model,
                "base_url": cfg.llm.base_url,
                "api_key": cfg.llm.api_key,
                "tavily_api_key": cfg.tavily.api_key,
                "ngrok_auth_token": cfg.ngrok.auth_token,
                "telegram_token": cfg.telegram.token,
                "telegram_chat_id": cfg.telegram.chat_id,
                "telegram_allowed_users": cfg.telegram.allowed_users,
            }
            if cfg.llm.has_worker_override:
                existing["worker_provider"] = cfg.llm.worker_provider
                existing["worker_model"] = cfg.llm.worker_model
                existing["worker_base_url"] = cfg.llm.worker_base_url
                existing["worker_api_key"] = cfg.llm.worker_api_key
        except Exception:
            pass  # corrupt config — treat as fresh setup

    # Run setup
    run_first_time_setup(config_path, identity_path, console, existing=existing)


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output"),
) -> None:
    """Start an interactive chat session."""
    asyncio.run(_run_chat(model, verbose))


async def _run_chat(model_override: str | None, verbose: bool = False) -> None:
    """Run the chat session."""
    from arc.cli.bootstrap import bootstrap, ArcRuntime
    from arc.core.events import Event, EventType
    from arc.platforms.cli.app import CLIPlatform
    from arc.notifications.channels.cli import CLIChannel

    config_path = get_config_path()
    if not config_path.exists():
        console.print(
            "[yellow]Arc is not configured yet.[/yellow]\n"
            "[dim]Run [bold]arc init[/bold] first.[/dim]"
        )
        raise typer.Exit(1)

    rt = await bootstrap(
        log_level=logging.DEBUG if verbose else logging.WARNING,
        model_override=model_override,
        platform_name="CLI",
        interactive_security=True,
    )

    # Create CLI platform
    cli = CLIPlatform(
        console=console,
        agent_name=rt.identity["agent_name"],
        user_name=rt.identity["user_name"],
    )

    cli.set_approval_flow(rt.agent.security.approval_flow)
    cli.set_escalation_bus(rt.escalation_bus)
    cli.set_skill_manager(rt.skill_manager)
    cli.set_skill_router(rt.skill_router)
    cli.set_mcp_manager(rt.mcp_manager)
    cli.set_turn_controller(rt.turn_controller)
    if rt.memory_manager is not None:
        cli.set_memory_manager(rt.memory_manager)

    # Wire workflow skill for /workflow command
    from arc.workflow.skill import WorkflowSkill as _WFSkillCLI
    wf_skill_cli = rt.skill_manager.get_skill("workflow")
    if wf_skill_cli and isinstance(wf_skill_cli, _WFSkillCLI):
        cli.set_workflow_skill(wf_skill_cli)

    # Queue for scheduler/worker results → CLI injection
    from arc.notifications.base import Notification as _Notification
    pending_queue: asyncio.Queue[_Notification] = asyncio.Queue()
    cli_channel = CLIChannel(pending_queue)
    rt.notification_router.register(cli_channel)
    cli.set_pending_queue(pending_queue)

    if rt.config.scheduler.enabled:
        cli.set_scheduler_store(rt.sched_store)

    # Worker-internal events to suppress from main chat
    _WORKER_INTERNAL: frozenset[str] = frozenset({
        EventType.AGENT_THINKING,
        EventType.SKILL_TOOL_CALL,
        EventType.SKILL_TOOL_RESULT,
        EventType.LLM_REQUEST,
        EventType.LLM_CHUNK,
        EventType.LLM_RESPONSE,
    })

    async def forward_to_cli(event: Event) -> None:
        if event.source != "main" and event.type in _WORKER_INTERNAL:
            return
        cli.on_event(event)

    rt.kernel.on(EventType.AGENT_THINKING, forward_to_cli)
    rt.kernel.on(EventType.SKILL_TOOL_CALL, forward_to_cli)
    rt.kernel.on(EventType.SKILL_TOOL_RESULT, forward_to_cli)
    rt.kernel.on(EventType.SECURITY_APPROVAL, forward_to_cli)
    rt.kernel.on(EventType.AGENT_ESCALATION, forward_to_cli)
    rt.kernel.on(EventType.AGENT_SPAWNED, forward_to_cli)
    rt.kernel.on(EventType.AGENT_TASK_COMPLETE, forward_to_cli)
    rt.kernel.on(EventType.AGENT_PLAN_UPDATE, forward_to_cli)
    rt.kernel.on(EventType.USER_INTERRUPT, forward_to_cli)

    # Workflow events
    rt.kernel.on(EventType.WORKFLOW_START, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_STEP_START, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_STEP_COMPLETE, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_STEP_FAILED, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_COMPLETE, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_PAUSED, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_WAITING_INPUT, forward_to_cli)

    # Message handler
    async def handle_message(user_input: str):
        rt.cost_tracker.start_turn()
        cli.set_cost_tracker(rt.cost_tracker.summary())
        async for chunk in rt.turn_controller.stream_message(user_input, source="cli"):
            yield chunk
        cli.set_cost_tracker(rt.cost_tracker.summary())

    try:
        await rt.start()
        cli_channel.set_active(True)
        await cli.run(handle_message)
    except Exception as e:
        logging.getLogger("arc").exception(f"Error in chat session: {e}")
        raise
    finally:
        cli_channel.set_active(False)
        await rt.shutdown()


@app.command()
def workers(
    follow: bool = typer.Option(False, "--follow", "-f", help="Keep watching for new activity"),
    lines: int = typer.Option(40, "--lines", "-n", help="Number of recent lines to show"),
) -> None:
    """Watch background worker activity in real time.

    Run this in a second terminal alongside 'arc chat' to see what
    workers are doing without cluttering the main chat window.

        arc workers          # show last 40 lines then exit
        arc workers --follow # live-tail, updates as workers run
    """
    import time as _time
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel

    log_path = get_arc_home() / "worker_activity.log"
    if not log_path.exists():
        console.print(
            "[dim]No worker activity log yet.  "
            "Start a chat session and delegate a task first.[/dim]"
        )
        raise typer.Exit(0)

    def _colourise(line: str) -> Text:
        """Apply Rich colours to a log line."""
        t = Text()
        # separator lines
        if line.startswith("\u2500") or line.startswith(" ") and "\u2014" in line:
            t.append(line, style="dim")
            return t
        parts = line.split(" | ", 3)
        if len(parts) < 3:
            t.append(line, style="dim")
            return t
        ts, worker, event = parts[0], parts[1], parts[2]
        detail = parts[3].rstrip() if len(parts) == 4 else ""
        event = event.strip()
        t.append(ts, style="dim")
        t.append(" | ", style="dim")
        t.append(f"{worker}", style="cyan")
        t.append(" | ", style="dim")
        if "SPAWNED" in event:
            t.append(f"{event:<10}", style="bold green")
        elif "COMPLETE" in event:
            t.append(f"{event:<10}", style="bold green" if "\u2713" in detail else "bold red")
        elif "TOOL CALL" in event:
            t.append(f"{event:<10}", style="yellow")
        elif "TOOL DONE" in event:
            icon_style = "green" if detail.startswith("\u2713") else "red"
            t.append(f"{event:<10}", style=icon_style)
        elif "ERROR" in event:
            t.append(f"{event:<10}", style="bold red")
        elif "THINKING" in event:
            t.append(f"{event:<10}", style="dim")
        else:
            t.append(f"{event:<10}", style="dim")
        if detail:
            t.append(" | ", style="dim")
            t.append(detail, style="dim" if "THINKING" in event else "")
        return t

    def _read_tail(n: int) -> list[str]:
        try:
            all_lines = log_path.read_text(encoding="utf-8").splitlines()
            return all_lines[-n:]
        except Exception:
            return []

    def _build_panel(tail_lines: list[str]) -> Panel:
        text = Text()
        for i, line in enumerate(tail_lines):
            if i:
                text.append("\n")
            text.append_text(_colourise(line))
        return Panel(
            text,
            title="[bold cyan]Worker Activity[/bold cyan]",
            subtitle="[dim]arc workers --follow  to live-tail[/dim]" if not follow else "[dim]Ctrl-C to exit[/dim]",
            border_style="cyan",
        )

    if not follow:
        console.print(_build_panel(_read_tail(lines)))
        raise typer.Exit(0)

    # --follow: live-tail with Rich Live, refresh every 0.1 s so THINKING
    # and TOOL CALL events are visible in real-time before COMPLETE lands.
    last_size = 0
    try:
        with Live(_build_panel(_read_tail(lines)), console=console, refresh_per_second=10) as live:
            while True:
                current_size = log_path.stat().st_size
                if current_size != last_size:
                    last_size = current_size
                    live.update(_build_panel(_read_tail(lines)))
                _time.sleep(0.1)
    except KeyboardInterrupt:
        pass


@app.command()
def version() -> None:
    """Show Arc version."""
    from arc import __version__
    console.print(f"Arc v{__version__}")


@app.command()
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    events: bool = typer.Option(False, "--events", "-e", help="Show events log instead"),
) -> None:
    """Show recent logs."""
    from datetime import datetime
    
    log_dir = get_arc_home() / "logs"
    
    if not log_dir.exists():
        console.print("[dim]No logs found.[/dim]")
        raise typer.Exit(0)
    
    # Find today's log file
    date_str = datetime.now().strftime('%Y%m%d')
    
    if events:
        log_file = log_dir / f"events_{date_str}.jsonl"
    else:
        log_file = log_dir / f"arc_{date_str}.log"
    
    if not log_file.exists():
        console.print(f"[dim]No log file for today: {log_file}[/dim]")
        raise typer.Exit(0)
    
    # Read and display last N lines
    with open(log_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        
    for line in all_lines[-lines:]:
        console.print(line.rstrip())


@app.command()
def telegram(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output"),
) -> None:
    """Run Arc as a Telegram bot (bidirectional chat)."""
    asyncio.run(_run_telegram(verbose))


async def _run_telegram(verbose: bool = False) -> None:
    """Run the Telegram bot platform."""
    from arc.cli.bootstrap import bootstrap
    from arc.core.config import ArcConfig
    from arc.platforms.telegram.app import TelegramPlatform

    config_path = get_config_path()
    if not config_path.exists():
        console.print(
            "[yellow]Arc is not configured yet.[/yellow]\n"
            "[dim]Run [bold]arc init[/bold] first.[/dim]"
        )
        raise typer.Exit(1)

    # Pre-check Telegram config before full bootstrap
    pre_config = ArcConfig.load()
    if not pre_config.telegram.platform_configured:
        console.print(
            "[yellow]Telegram bot is not configured.[/yellow]\n"
            "[dim]Run [bold]arc init[/bold] and set up a Telegram bot token,\n"
            "or add it manually to ~/.arc/config.toml:\n\n"
            "  [telegram]\n"
            '  token = "YOUR_BOT_TOKEN"\n'
            '  allowed_users = ["YOUR_CHAT_ID"][/dim]'
        )
        raise typer.Exit(1)

    rt = await bootstrap(
        log_level=logging.DEBUG if verbose else logging.INFO,
        platform_name="Telegram bot",
        interactive_security=False,
    )

    # Disable auto-open for Liquid Web (no desktop browser on Telegram)
    from arc.skills.builtin.liquid_web import LiquidWebSkill
    lw_skill = rt.skill_manager.get_skill("liquid_web")
    if lw_skill and isinstance(lw_skill, LiquidWebSkill):
        lw_skill._auto_open = False

    # Create Telegram platform
    allowed = set(rt.config.telegram.allowed_users) if rt.config.telegram.allowed_users else None
    tg_platform = TelegramPlatform(
        token=rt.config.telegram.token,
        allowed_chat_ids=allowed,
        agent_name=rt.identity["agent_name"],
    )
    tg_platform.set_cost_tracker(rt.cost_tracker)

    # Message handler
    async def handle_message(user_input: str):
        rt.cost_tracker.start_turn()
        async for chunk in rt.agent.run(user_input):
            yield chunk

    console.print(
        Panel(
            f"[bold green]Telegram bot starting[/bold green]\n\n"
            f"Bot: {rt.identity['agent_name']}\n"
            f"Allowed users: {rt.config.telegram.allowed_users or 'everyone'}\n\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        )
    )

    try:
        await rt.start()
        await tg_platform.run(handle_message)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.getLogger("arc").exception(f"Error in Telegram bot: {e}")
        raise
    finally:
        await tg_platform.stop()
        await rt.shutdown()


@app.command()
def config() -> None:
    """Show current configuration."""
    config_path = get_config_path()
    identity_path = get_identity_path()
    
    console.print(Panel("[bold]Arc Configuration[/bold]", border_style="cyan"))
    console.print()
    
    # Show config file location and content
    console.print(f"[bold]Config file:[/bold] {config_path}")
    if config_path.exists():
        console.print(Panel(config_path.read_text(), title="config.toml", border_style="dim"))
    else:
        console.print("[dim]Not found. Run 'arc init'[/dim]")
    
    console.print()
    
    # Show identity file location
    console.print(f"[bold]Identity file:[/bold] {identity_path}")
    if identity_path.exists():
        content = identity_path.read_text()
        # Show just first part
        preview = "\n".join(content.split("\n")[:20])
        if len(content.split("\n")) > 20:
            preview += "\n..."
        console.print(Panel(preview, title="identity.md", border_style="dim"))
    else:
        console.print("[dim]Not found. Run 'arc init'[/dim]")


@app.command()
def gateway(
    port: int = typer.Option(18789, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output"),
) -> None:
    """Run Arc Gateway — WebSocket server + WebChat UI.

    Open http://localhost:18789 in a browser for WebChat.
    Connect any WebSocket client to ws://localhost:18789/ws.

    The Gateway shares the same agent, memory, and session as
    CLI and Telegram — conversations stay in sync.
    """
    asyncio.run(_run_gateway(host, port, verbose))


async def _run_gateway(host: str, port: int, verbose: bool = False) -> None:
    """Bootstrap and run the Gateway server."""
    from arc.cli.bootstrap import bootstrap
    from arc.core.events import Event, EventType
    from arc.gateway.server import GatewayServer

    config_path = get_config_path()
    if not config_path.exists():
        console.print(
            "[yellow]Arc is not configured yet.[/yellow]\n"
            "[dim]Run [bold]arc init[/bold] first.[/dim]"
        )
        raise typer.Exit(1)

    rt = await bootstrap(
        log_level=logging.DEBUG if verbose else logging.INFO,
        platform_name="Gateway (WebSocket + WebChat)",
        interactive_security=False,
    )

    # Disable auto-open for Liquid Web when running as gateway
    from arc.skills.builtin.liquid_web import LiquidWebSkill
    lw_skill = rt.skill_manager.get_skill("liquid_web")
    if lw_skill and isinstance(lw_skill, LiquidWebSkill):
        lw_skill._auto_open = False

    # Create Gateway server
    gw = GatewayServer(host=host, port=port)

    # Wire dependencies for slash commands
    gw.set_skill_manager(rt.skill_manager)
    gw.set_cost_tracker(rt.cost_tracker.summary())
    if rt.memory_manager is not None:
        gw.set_memory_manager(rt.memory_manager)
    if rt.config.scheduler.enabled:
        gw.set_scheduler_store(rt.sched_store)
    gw.set_mcp_manager(rt.mcp_manager)
    if rt.mcp_config_service is not None:
        gw.set_mcp_config_service(rt.mcp_config_service)
    gw.set_session_memory(rt.agent._memory)
    gw.set_run_control(rt.run_control)
    gw.set_turn_controller(rt.turn_controller)

    # Wire task board dependencies
    if rt.task_store:
        gw.set_task_store(rt.task_store)
    if rt.task_processor:
        gw.set_task_processor(rt.task_processor)
    if rt.agent_defs:
        gw.set_agent_defs(rt.agent_defs)

    # Wire workflow skill for /workflow command
    from arc.workflow.skill import WorkflowSkill as _WFSkill
    wf_skill = rt.skill_manager.get_skill("workflow")
    if wf_skill and isinstance(wf_skill, _WFSkill):
        gw.set_workflow_skill(wf_skill)
    gw.set_kernel(rt.kernel)

    # Attach Telegram as a channel (if configured)
    if rt.config.telegram.platform_configured:
        from arc.platforms.telegram.app import TelegramPlatform
        allowed = set(rt.config.telegram.allowed_users) if rt.config.telegram.allowed_users else None
        tg_platform = TelegramPlatform(
            token=rt.config.telegram.token,
            allowed_chat_ids=allowed,
            agent_name=rt.identity["agent_name"],
        )
        tg_platform.set_cost_tracker(rt.cost_tracker)
        gw.attach_channel(tg_platform)

    # Wire kernel events → Gateway broadcast
    _WORKER_INTERNAL: frozenset[str] = frozenset({
        EventType.AGENT_THINKING,
        EventType.LLM_REQUEST,
        EventType.LLM_CHUNK,
        EventType.LLM_RESPONSE,
    })

    async def forward_to_gateway(event: Event) -> None:
        if event.source != "main" and event.type in _WORKER_INTERNAL:
            return
        await gw.broadcast_event(event.type, {**event.data, "source": event.source})

    rt.kernel.on(EventType.AGENT_THINKING, forward_to_gateway)
    rt.kernel.on(EventType.SKILL_TOOL_CALL, forward_to_gateway)
    rt.kernel.on(EventType.SKILL_TOOL_RESULT, forward_to_gateway)
    rt.kernel.on(EventType.AGENT_SPAWNED, forward_to_gateway)
    rt.kernel.on(EventType.AGENT_TASK_COMPLETE, forward_to_gateway)
    rt.kernel.on(EventType.AGENT_PLAN_UPDATE, forward_to_gateway)
    rt.kernel.on(EventType.WORKSPACE_UPDATE, forward_to_gateway)

    # Record ALL events into the gateway's ring buffer for the Logs tab
    async def record_event_for_logs(event: Event) -> None:
        gw.record_event(event.type, event.source, event.data)

    rt.kernel.on("*", record_event_for_logs)

    # Workflow events are NOT broadcast here — the /workflow handler
    # and the WorkflowSkill manage their own display to avoid
    # duplicate/out-of-order messages with the agent text stream.

    # ── Notification channel for Gateway ──
    # Workers and scheduled jobs deliver results through this channel.
    # The notification appears in WebChat AND is queued for the agent
    # to summarise on the next user turn.
    from arc.notifications.base import Notification as _GWNotification
    from arc.notifications.channels.gateway import GatewayChannel

    gw_pending_queue: asyncio.Queue[_GWNotification] = asyncio.Queue()
    gw_channel = GatewayChannel(
        broadcast_fn=gw.broadcast_notification,
        queue=gw_pending_queue,
    )
    rt.notification_router.register(gw_channel)

    # Message handler
    async def handle_message(user_input: str):
        rt.cost_tracker.start_turn()

        # ── Workflow input intercept ──
        # If a workflow is paused waiting for user input, route
        # the message to the workflow instead of the agent.
        wf_skill = rt.skill_manager.get_skill("workflow")
        if wf_skill and hasattr(wf_skill, "is_waiting_for_input") and wf_skill.is_waiting_for_input:
            wf_skill.provide_input(user_input)
            yield f"[Input received — workflow resuming]\n"
            return

        # Inject any pending job/worker results into the prompt
        pending_results: list[str] = []
        while not gw_pending_queue.empty():
            try:
                notif = gw_pending_queue.get_nowait()
                pending_results.append(
                    f"[Background task '{notif.job_name}' completed]\n{notif.content}"
                )
            except asyncio.QueueEmpty:
                break

        actual_input = user_input
        if pending_results:
            injected = "\n\n".join(pending_results)
            actual_input = (
                f"The following background task(s) completed since the last message. "
                f"Summarise the results for the user, then address their new message "
                f"(if any).\n\n{injected}\n\nUser message: {user_input}"
            )

        async for chunk in rt.turn_controller.stream_message(actual_input, source="gateway"):
            yield chunk
        gw.set_cost_tracker(rt.cost_tracker.summary())

    # Build startup info
    channels_info = ""
    if rt.config.telegram.platform_configured:
        channels_info += f"Telegram: [bold green]connected[/bold green]\n"
    else:
        channels_info += f"Telegram: [dim]not configured (run arc init)[/dim]\n"

    console.print(
        Panel(
            f"[bold green]Arc Gateway starting[/bold green]\n\n"
            f"Agent: {rt.identity['agent_name']}\n"
            f"WebChat: [bold]http://{host}:{port}[/bold]\n"
            f"WebSocket: [bold]ws://{host}:{port}/ws[/bold]\n"
            f"Health: [bold]http://{host}:{port}/health[/bold]\n\n"
            f"[bold]Channels:[/bold]\n"
            f"WebChat: [bold green]always on[/bold green]\n"
            f"{channels_info}\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        )
    )

    try:
        await rt.start()
        gw_channel.set_active(True)
        await gw.run(handle_message)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.getLogger("arc").exception(f"Error in Gateway: {e}")
        raise
    finally:
        gw_channel.set_active(False)
        await gw.stop()
        await rt.shutdown()


@app.command()
def listen(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output"),
    host: str = typer.Option("127.0.0.1", "--host", help="Gateway host"),
    port: int = typer.Option(18789, "--port", "-p", help="Gateway port"),
    no_overlay: bool = typer.Option(False, "--no-overlay", help="Disable screen overlay, use terminal bar"),
) -> None:
    """Start voice input — talk to Arc hands-free.

    Requires 'arc gateway' to be running in another terminal.
    Connects as a WebSocket client — your speech appears in WebChat.

    Install voice dependencies first:
        pip install sounddevice faster-whisper openwakeword

    For the ambient screen glow (optional):
        pip install PyQt6

    Say the wake word (default: "Hey Jarvis") to activate,
    then speak your request. Arc listens, transcribes, and responds.
    """
    # Check for voice dependencies before doing anything
    missing: list[str] = []
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster-whisper")
    try:
        import openwakeword  # noqa: F401
    except ImportError:
        missing.append("openwakeword")

    if missing:
        console.print(
            f"[red]Missing voice dependencies:[/red] {', '.join(missing)}\n\n"
            "[dim]Install them with:\n"
            "  pip install sounddevice faster-whisper openwakeword\n\n"
            "On Linux you also need:\n"
            "  sudo apt install libportaudio2[/dim]"
        )
        raise typer.Exit(1)

    from arc.voice.overlay import is_available as _overlay_available

    use_overlay = not no_overlay and _overlay_available()

    if use_overlay:
        _run_listen_with_overlay(host, port, verbose)
    else:
        if not no_overlay and not _overlay_available():
            console.print(
                "[dim]PyQt6 not installed — using terminal indicator. "
                "Install with: pip install PyQt6[/dim]\n"
            )
        asyncio.run(_run_listen_terminal(host, port, verbose))


def _run_listen_with_overlay(host: str, port: int, verbose: bool) -> None:
    """Run voice daemon with PyQt6 ambient edge glow overlay.

    Qt owns the main thread; asyncio daemon runs in a QThread.
    """
    import sys
    import threading

    from PyQt6.QtCore import QThread, QTimer
    from PyQt6.QtWidgets import QApplication

    from arc.voice.overlay import create_overlay
    from arc.voice.listener import VoiceState
    from arc.middleware.logging import setup_logging
    from arc.core.config import ArcConfig

    arc_home = get_arc_home()
    setup_logging(
        log_dir=arc_home / "logs",
        console_level=logging.DEBUG if verbose else logging.WARNING,
    )
    config = ArcConfig.load()
    gateway_url = f"ws://{host}:{port}/ws"

    # Qt app — must be created on main thread before any widgets
    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)  # no visible windows to close

    result = create_overlay()
    if result is None:
        # Shouldn't happen (we checked is_available), but fallback
        asyncio.run(_run_listen_terminal(host, port, verbose))
        return

    glow_bar, bridge = result

    # Map VoiceState → overlay state string
    _state_map = {
        VoiceState.SLEEPING: "sleeping",
        VoiceState.ACTIVE: "active",
        VoiceState.PROCESSING: "processing",
        VoiceState.LISTENING: "listening",
    }

    def status_callback(state: VoiceState, event: str) -> None:
        """Called from asyncio thread — emits Qt signal (thread-safe)."""
        overlay_state = _state_map.get(state, "sleeping")
        bridge.state_changed.emit(overlay_state)

    # Print startup info to terminal (before Qt takes over)
    wake_display = config.voice.wake_model.replace("_", " ").title()
    console.print(
        Panel(
            f"[bold green]Arc Voice starting (overlay mode)[/bold green]\n\n"
            f"Gateway: [bold]{gateway_url}[/bold]\n"
            f"Whisper model: [bold]{config.voice.whisper_model}[/bold]\n"
            f"Wake word: [bold]{wake_display}[/bold]\n\n"
            f"Say [bold cyan]{wake_display}[/bold cyan] to start talking.\n"
            "[dim]The screen edge will glow when Arc is listening.\n"
            "Press Ctrl+C to stop.[/dim]",
            border_style="cyan",
        )
    )

    # Run the daemon in a background thread with its own event loop
    daemon_error: list[str] = []
    daemon_loop: asyncio.AbstractEventLoop | None = None
    daemon_ref: list[Any] = []  # holds daemon instance for shutdown

    def _daemon_thread() -> None:
        nonlocal daemon_loop
        from arc.voice.daemon import VoiceDaemon

        daemon = VoiceDaemon(
            gateway_url=gateway_url,
            whisper_model=config.voice.whisper_model,
            wake_model=config.voice.wake_model,
            wake_threshold=config.voice.wake_threshold,
            silence_duration=config.voice.silence_duration,
            listen_timeout=config.voice.listen_timeout,
            status_callback=status_callback,
            tts_provider=config.voice.tts_provider,
            tts_voice=config.voice.tts_voice,
            tts_speed=config.voice.tts_speed,
        )
        daemon_ref.append(daemon)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        daemon_loop = loop
        try:
            loop.run_until_complete(daemon.run())
        except ConnectionError as e:
            daemon_error.append(str(e))
        except Exception as e:
            if not isinstance(e, SystemExit):
                daemon_error.append(str(e))
        finally:
            # Graceful cleanup — run stop() to close WebSocket, aiohttp, mic
            try:
                if not loop.is_closed():
                    loop.run_until_complete(daemon.stop())
            except Exception:
                pass
            # Cancel any remaining tasks
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending and not loop.is_closed():
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            if not loop.is_closed():
                loop.close()
            daemon_loop = None
            # Tell Qt to quit from the daemon thread
            QTimer.singleShot(0, qt_app.quit)

    thread = threading.Thread(target=_daemon_thread, name="voice-daemon", daemon=True)
    thread.start()

    # Handle Ctrl+C: signal daemon to stop gracefully, then quit Qt
    import signal

    def _sigint_handler(*_: Any) -> None:
        # Schedule daemon.stop() on the event loop (graceful shutdown)
        # instead of loop.stop() (brutal — leaves resources unclosed)
        if daemon_loop is not None and daemon_loop.is_running() and daemon_ref:
            daemon_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(daemon_ref[0].stop())
            )
        # Give the daemon a moment to clean up, then force Qt quit
        QTimer.singleShot(500, qt_app.quit)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Qt event loop on main thread — blocks until qt_app.quit()
    qt_app.exec()

    # Give the daemon thread a moment to finish cleanly
    thread.join(timeout=3.0)
    if thread.is_alive():
        # Force-stop the loop if still running
        if daemon_loop is not None and daemon_loop.is_running():
            daemon_loop.call_soon_threadsafe(daemon_loop.stop)
        thread.join(timeout=2.0)

    if daemon_error:
        console.print(f"[red]{daemon_error[0]}[/red]")


async def _run_listen_terminal(
    host: str,
    port: int,
    verbose: bool = False,
) -> None:
    """Run the voice daemon with terminal-based Rich indicator (fallback)."""
    from arc.middleware.logging import setup_logging

    arc_home = get_arc_home()
    setup_logging(
        log_dir=arc_home / "logs",
        console_level=logging.DEBUG if verbose else logging.WARNING,
    )

    # Load config for voice settings
    from arc.core.config import ArcConfig

    config = ArcConfig.load()
    gateway_url = f"ws://{host}:{port}/ws"

    from arc.voice.daemon import VoiceDaemon
    from arc.voice.listener import VoiceState
    from rich.align import Align
    from rich.console import Group
    from rich.live import Live
    from rich.text import Text

    state_info: dict[str, Any] = {
        "state": VoiceState.SLEEPING,
        "event": "sleep",
    }
    update_event = asyncio.Event()
    stop_indicator = asyncio.Event()

    def status_callback(state: VoiceState, event: str) -> None:
        state_info["state"] = state
        state_info["event"] = event
        update_event.set()

    def _render_bar(phase: int) -> Panel:
        state = state_info["state"]
        event = state_info["event"]
        console_width = max(10, console.size.width - 4)
        bar_chars = "▁▂▃▄▅▆▇█"
        animate_states = {VoiceState.ACTIVE, VoiceState.PROCESSING}
        labels = {
            VoiceState.SLEEPING: ("Sleeping", "grey35"),
            VoiceState.ACTIVE: ("Listening…", "cyan"),
            VoiceState.PROCESSING: ("Processing…", "yellow"),
            VoiceState.LISTENING: ("Awaiting follow-up", "green"),
        }
        label, colour = labels.get(state, ("Listening…", "cyan"))
        if state in animate_states:
            char = bar_chars[phase % len(bar_chars)]
            bar_body = char * console_width
        elif state == VoiceState.SLEEPING:
            bar_body = "░" * console_width
        else:
            bar_body = "█" * console_width
        bar = Text(bar_body, style=f"bold {colour}")
        status_line = Text(label.upper(), style=f"bold {colour}")
        event_line = Text(f"event: {event}", style="dim")
        content = Group(bar, status_line, event_line)
        return Panel(
            Align.center(content, vertical="middle"),
            title="Voice Status",
            border_style=colour,
            padding=(0, 1),
            expand=True,
        )

    async def indicator_loop() -> None:
        phase = 0
        anim_steps = 8
        update_event.set()
        with Live(
            _render_bar(phase),
            console=console,
            refresh_per_second=12,
            transient=False,
        ) as live:
            while not stop_indicator.is_set():
                try:
                    await asyncio.wait_for(update_event.wait(), timeout=0.25)
                    update_event.clear()
                except asyncio.TimeoutError:
                    pass
                if stop_indicator.is_set():
                    break
                if state_info["state"] in {VoiceState.ACTIVE, VoiceState.PROCESSING}:
                    phase = (phase + 1) % anim_steps
                else:
                    phase = 0
                live.update(_render_bar(phase))
            live.update(_render_bar(0))
    daemon = VoiceDaemon(
        gateway_url=gateway_url,
        whisper_model=config.voice.whisper_model,
        wake_model=config.voice.wake_model,
        wake_threshold=config.voice.wake_threshold,
        silence_duration=config.voice.silence_duration,
        listen_timeout=config.voice.listen_timeout,
        status_callback=status_callback,
        tts_provider=config.voice.tts_provider,
        tts_voice=config.voice.tts_voice,
        tts_speed=config.voice.tts_speed,
    )

    wake_display = config.voice.wake_model.replace("_", " ").title()
    console.print(
        Panel(
            f"[bold green]Arc Voice starting[/bold green]\n\n"
            f"Gateway: [bold]{gateway_url}[/bold]\n"
            f"Whisper model: [bold]{config.voice.whisper_model}[/bold]\n"
            f"Wake word: [bold]{wake_display}[/bold]\n"
            f"Silence timeout: {config.voice.silence_duration}s\n"
            f"Listen window: {config.voice.listen_timeout}s\n\n"
            f"Say [bold cyan]{wake_display}[/bold cyan] to start talking.\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        )
    )

    indicator_task = asyncio.create_task(indicator_loop())

    try:
        await daemon.run()
    except ConnectionError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_indicator.set()
        update_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await indicator_task
        with contextlib.suppress(Exception):
            await daemon.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task Board CLI commands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

task_app = typer.Typer(name="task", help="Manage the persistent task queue.")
app.add_typer(task_app)


@task_app.command("add")
def task_add(
    title: str = typer.Argument(..., help="Short title for the task"),
    instruction: str = typer.Option("", "--instruction", "-i", help="Full instructions (defaults to title if empty)"),
    assign: str = typer.Option("", "--assign", "-a", help="Agent name to assign to"),
    priority: int = typer.Option(1, "--priority", "-p", help="Priority (1=highest)"),
    max_bounces: int = typer.Option(3, "--max-bounces", help="Max review iterations"),
    depends_on: str = typer.Option("", "--after", help="Task ID that must complete first"),
) -> None:
    """Add a task to the queue."""
    asyncio.run(_task_add(title, instruction, assign, priority, max_bounces, depends_on))


async def _task_add(
    title: str, instruction: str, assign: str,
    priority: int, max_bounces: int, depends_on: str,
) -> None:
    from arc.tasks.store import TaskStore
    from arc.tasks.types import Task, TaskStep
    from arc.tasks.agents import load_agent_defs

    agents = load_agent_defs()
    if assign and assign not in agents:
        console.print(f"[red]Unknown agent '{assign}'.[/red] Available: {', '.join(agents) or 'none'}")
        raise typer.Exit(1)

    if not assign:
        if not agents:
            console.print("[red]No agents defined. Create one first:[/red] arc agent create <name>")
            raise typer.Exit(1)
        assign = list(agents.keys())[0]
        console.print(f"[dim]No agent specified, using '{assign}'[/dim]")

    task = Task(
        title=title,
        instruction=instruction or title,
        steps=[TaskStep(step_index=0, agent_name=assign)],
        assigned_agent=assign,
        priority=max(1, min(priority, 10)),
        max_bounces=max(1, min(max_bounces, 10)),
        depends_on=depends_on or None,
    )

    store = TaskStore()
    await store.initialize()
    await store.save(task)
    await store.close()

    console.print(f"[green]✓[/green] Task queued: {task.title} (id: {task.id}, agent: {assign})")


@task_app.command("list")
def task_list(
    status: str = typer.Option("", "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(30, "--limit", "-n", help="Max results"),
) -> None:
    """List tasks in the queue."""
    asyncio.run(_task_list(status, limit))


async def _task_list(status: str, limit: int) -> None:
    from arc.tasks.store import TaskStore
    from rich.table import Table

    store = TaskStore()
    await store.initialize()
    tasks = await store.get_all(status=status or None, limit=limit)
    await store.close()

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(title="Task Queue", border_style="dim")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Title")
    table.add_column("Agent", style="green")
    table.add_column("Step")
    table.add_column("Bounces")

    status_styles = {
        "queued": "yellow",
        "in_progress": "bold cyan",
        "in_review": "magenta",
        "revision_needed": "bold yellow",
        "awaiting_human": "bold red",
        "blocked": "bold red",
        "done": "green",
        "failed": "red",
        "cancelled": "dim",
    }

    for t in tasks:
        style = status_styles.get(t.status.value, "")
        step_info = f"{t.current_step + 1}/{len(t.steps) or 1}"
        table.add_row(
            t.id,
            f"[{style}]{t.status.value}[/{style}]",
            t.title[:50],
            t.current_agent,
            step_info,
            f"{t.bounce_count}/{t.max_bounces}",
        )

    console.print(table)


@task_app.command("show")
def task_show(
    task_id: str = typer.Argument(..., help="Task ID"),
) -> None:
    """Show full detail for a task including comments."""
    asyncio.run(_task_show(task_id))


async def _task_show(task_id: str) -> None:
    from arc.tasks.store import TaskStore

    store = TaskStore()
    await store.initialize()
    task = await store.get_by_id(task_id)
    if not task:
        console.print(f"[red]Task '{task_id}' not found.[/red]")
        await store.close()
        return

    comments = await store.get_comments(task_id)
    await store.close()

    console.print(Panel(
        f"[bold]{task.title}[/bold]\n"
        f"ID: {task.id}\n"
        f"Status: {task.status.value}\n"
        f"Agent: {task.current_agent}\n"
        f"Step: {task.current_step + 1}/{len(task.steps) or 1}\n"
        f"Bounces: {task.bounce_count}/{task.max_bounces}\n"
        f"Priority: {task.priority}\n"
        f"{'Depends on: ' + task.depends_on if task.depends_on else ''}\n\n"
        f"[bold]Instruction:[/bold]\n{task.instruction}",
        title="Task Detail",
        border_style="cyan",
    ))

    if task.result:
        console.print(Panel(task.result[:2000], title="Final Result", border_style="green"))

    if comments:
        console.print(f"\n[bold]Comments ({len(comments)}):[/bold]")
        for c in comments:
            style = "green" if c.agent_name == "human" else "cyan" if c.agent_name != "system" else "dim"
            console.print(f"  [{style}][{c.agent_name}][/{style}] {c.content[:300]}")


@task_app.command("cancel")
def task_cancel(
    task_id: str = typer.Argument(..., help="Task ID to cancel"),
) -> None:
    """Cancel a task."""
    asyncio.run(_task_cancel(task_id))


async def _task_cancel(task_id: str) -> None:
    from arc.tasks.store import TaskStore

    store = TaskStore()
    await store.initialize()
    ok = await store.cancel(task_id)
    await store.close()

    if ok:
        console.print(f"[green]✓[/green] Task {task_id} cancelled.")
    else:
        console.print(f"[red]Task {task_id} not found or already completed.[/red]")


@task_app.command("reply")
def task_reply(
    task_id: str = typer.Argument(..., help="Task ID to reply to"),
    reply: str = typer.Argument(..., help="Your response"),
    action: str = typer.Option("approve", "--action", "-a", help="'approve' or 'revise'"),
) -> None:
    """Reply to a blocked or awaiting-human task."""
    asyncio.run(_task_reply(task_id, reply, action))


async def _task_reply(task_id: str, reply: str, action: str) -> None:
    from arc.tasks.store import TaskStore
    from arc.tasks.types import TaskStatus

    store = TaskStore()
    await store.initialize()
    task = await store.get_blocked_task(task_id)
    if not task:
        console.print(f"[red]Task {task_id} is not waiting for human input.[/red]")
        await store.close()
        return

    if task.status == TaskStatus.AWAITING_HUMAN:
        if action == "approve":
            await store.update_status_with_comment(
                task_id, TaskStatus.QUEUED, "human",
                f"APPROVED: {reply}", task.current_step,
                extra_updates={"current_step": task.current_step + 1, "bounce_count": 0},
            )
            console.print(f"[green]✓[/green] Task {task_id} approved. Moving to next step.")
        else:
            await store.update_status_with_comment(
                task_id, TaskStatus.REVISION_NEEDED, "human",
                f"Revision requested: {reply}", task.current_step,
                extra_updates={"bounce_count": task.bounce_count + 1},
            )
            console.print(f"[green]✓[/green] Task {task_id} sent back for revision.")
    elif task.status == TaskStatus.BLOCKED:
        await store.update_status_with_comment(
            task_id, TaskStatus.QUEUED, "human",
            reply, task.current_step,
        )
        console.print(f"[green]✓[/green] Answer delivered to task {task_id}.")

    await store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent management CLI commands
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

agent_app = typer.Typer(name="agent", help="Manage named agents.")
app.add_typer(agent_app)


@agent_app.command("create")
def agent_create(
    name: str = typer.Argument(..., help="Agent name (alphanumeric + underscore)"),
    role: str = typer.Option("", "--role", "-r", help="Agent's role description"),
    personality: str = typer.Option("", "--personality", "-p", help="Personality traits"),
    model: str = typer.Option("", "--model", "-m", help="LLM model (e.g. 'ollama/llama3.2', 'openai/gpt-4o')"),
    max_concurrent: int = typer.Option(1, "--max-concurrent", help="Max parallel tasks"),
) -> None:
    """Create a new named agent.

    After creating, edit ~/.arc/agents/<name>.toml to add a detailed
    system_prompt using triple-quoted strings.
    """
    from arc.tasks.types import AgentDef
    from arc.tasks.agents import save_agent_def

    # Parse model string
    llm_provider = ""
    llm_model = ""
    if model and "/" in model:
        llm_provider, llm_model = model.split("/", 1)
    elif model:
        llm_model = model

    agent = AgentDef(
        name=name,
        role=role,
        personality=personality,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_concurrent=max_concurrent,
    )

    path = save_agent_def(agent)
    console.print(f"[green]✓[/green] Agent '{name}' created at {path}")
    if role:
        console.print(f"  Role: {role}")
    if model:
        console.print(f"  Model: {model}")
    console.print(f"  [dim]Edit {path} to add a detailed system_prompt[/dim]")


@agent_app.command("list")
def agent_list() -> None:
    """List all configured agents."""
    from arc.tasks.agents import load_agent_defs
    from rich.table import Table

    agents = load_agent_defs()
    if not agents:
        console.print(
            "[dim]No agents configured. Create one:[/dim]\n"
            "  arc agent create researcher --role 'Web research' --model ollama/llama3.2"
        )
        return

    table = Table(title="Named Agents", border_style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Role")
    table.add_column("LLM", style="green")
    table.add_column("Concurrent", justify="center")

    for a in agents.values():
        llm = f"{a.llm_provider}/{a.llm_model}" if a.has_llm_override else "default"
        table.add_row(a.name, a.role, llm, str(a.max_concurrent))

    console.print(table)


@agent_app.command("remove")
def agent_remove(
    name: str = typer.Argument(..., help="Agent name to remove"),
) -> None:
    """Remove an agent definition."""
    from arc.tasks.agents import _AGENTS_DIR

    path = _AGENTS_DIR / f"{name}.toml"
    if not path.exists():
        console.print(f"[red]Agent '{name}' not found.[/red]")
        raise typer.Exit(1)

    path.unlink()
    console.print(f"[green]✓[/green] Agent '{name}' removed.")


if __name__ == "__main__":
    app()
