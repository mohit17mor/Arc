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
    if rt.mcp_manager.has_servers:
        cli.set_mcp_manager(rt.mcp_manager)
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

    # Workflow events
    rt.kernel.on(EventType.WORKFLOW_START, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_STEP_START, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_STEP_COMPLETE, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_STEP_FAILED, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_COMPLETE, forward_to_cli)
    rt.kernel.on(EventType.WORKFLOW_PAUSED, forward_to_cli)

    # Message handler
    async def handle_message(user_input: str):
        rt.cost_tracker.start_turn()
        cli.set_cost_tracker(rt.cost_tracker.summary())
        async for chunk in rt.agent.run(user_input):
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
    if rt.mcp_manager.has_servers:
        gw.set_mcp_manager(rt.mcp_manager)
    gw.set_session_memory(rt.agent._memory)

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
        await gw.broadcast_event(event.type, event.data)

    rt.kernel.on(EventType.AGENT_THINKING, forward_to_gateway)
    rt.kernel.on(EventType.SKILL_TOOL_CALL, forward_to_gateway)
    rt.kernel.on(EventType.SKILL_TOOL_RESULT, forward_to_gateway)
    rt.kernel.on(EventType.AGENT_SPAWNED, forward_to_gateway)
    rt.kernel.on(EventType.AGENT_TASK_COMPLETE, forward_to_gateway)

    # Workflow events are NOT broadcast here — the /workflow handler
    # and the WorkflowSkill manage their own display to avoid
    # duplicate/out-of-order messages with the agent text stream.

    # Message handler
    async def handle_message(user_input: str):
        rt.cost_tracker.start_turn()
        async for chunk in rt.agent.run(user_input):
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
        await gw.run(handle_message)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.getLogger("arc").exception(f"Error in Gateway: {e}")
        raise
    finally:
        await gw.stop()
        await rt.shutdown()


@app.command()
def listen(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output"),
    host: str = typer.Option("127.0.0.1", "--host", help="Gateway host"),
    port: int = typer.Option(18789, "--port", "-p", help="Gateway port"),
) -> None:
    """Start voice input — talk to Arc hands-free.

    Requires 'arc gateway' to be running in another terminal.
    Connects as a WebSocket client — your speech appears in WebChat.

    Install voice dependencies first:
        pip install sounddevice faster-whisper openwakeword

    Say the wake word (default: "Hey Jarvis") to activate,
    then speak your request. Arc listens, transcribes, and responds.
    """
    asyncio.run(_run_listen(host, port, verbose))


async def _run_listen(
    host: str,
    port: int,
    verbose: bool = False,
) -> None:
    """Run the voice daemon."""
    from arc.middleware.logging import setup_logging

    arc_home = get_arc_home()
    setup_logging(
        log_dir=arc_home / "logs",
        console_level=logging.DEBUG if verbose else logging.WARNING,
    )

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
    except KeyboardInterrupt:
        pass
    finally:
        stop_indicator.set()
        update_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await indicator_task
        await daemon.stop()


if __name__ == "__main__":
    app()
