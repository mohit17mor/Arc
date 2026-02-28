"""
Arc CLI entry point.

Commands:
    arc init  — First-time setup
    arc chat  — Interactive chat
"""

from __future__ import annotations

import asyncio
import logging
import typer
from rich.console import Console
from pathlib import Path
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
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
) -> None:
    """Initialize Arc — first-time setup wizard."""
    from arc.identity.setup import run_first_time_setup

    config_path = get_config_path()
    identity_path = get_identity_path()

    # Check if already configured
    if config_path.exists() and not force:
        console.print(
            f"[yellow]Arc is already configured at {config_path}[/yellow]\n"
            f"[dim]Use --force to reconfigure[/dim]"
        )
        raise typer.Exit(0)

    # Run setup
    run_first_time_setup(config_path, identity_path, console)


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Override model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug output"),
) -> None:
    """Start an interactive chat session."""
    asyncio.run(_run_chat(model, verbose))


async def _run_chat(model_override: str | None, verbose: bool = False) -> None:
    """Run the chat session."""
    import platform as plat
    
    from arc.core.kernel import Kernel
    from arc.core.config import ArcConfig
    from arc.core.events import Event, EventType
    from arc.llm.ollama import OllamaProvider
    from arc.skills.manager import SkillManager
    from arc.skills.loader import discover_skills, discover_soft_skills
    from arc.security.engine import SecurityEngine
    from arc.agent.loop import AgentLoop, AgentConfig
    from arc.identity.soul import SoulManager
    from arc.platforms.cli.app import CLIPlatform
    from arc.middleware.cost import CostTracker
    from arc.middleware.logging import setup_logging, EventLogger
    from arc.memory.manager import MemoryManager
    from arc.notifications.router import NotificationRouter
    from arc.notifications.channels.cli import CLIChannel
    from arc.notifications.channels.file import FileChannel
    from arc.notifications.channels.telegram import TelegramChannel
    from arc.scheduler.store import SchedulerStore
    from arc.scheduler.engine import SchedulerEngine
    from arc.skills.builtin.scheduler import SchedulerSkill
    from arc.skills.builtin.worker import WorkerSkill
    from arc.agent.registry import AgentRegistry
    from arc.core.escalation import EscalationBus
    from arc.agent.worker_log import WorkerActivityLog

    config_path = get_config_path()
    identity_path = get_identity_path()

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    setup_logging(
        log_dir=get_arc_home() / "logs",
        console_level=log_level,
    )
    logger = logging.getLogger("arc")
    logger.info("Starting Arc chat session")

    # Check if configured
    if not config_path.exists():
        console.print(
            "[yellow]Arc is not configured yet.[/yellow]\n"
            "[dim]Run [bold]arc init[/bold] first.[/dim]"
        )
        raise typer.Exit(1)

    # Load config
    config = ArcConfig.load()
    logger.debug(f"Config loaded: model={config.llm.default_model}")

    # Override model if specified
    if model_override:
        config.llm.default_model = model_override
        logger.info(f"Model overridden to: {model_override}")

    # Load identity
    soul = SoulManager(identity_path)
    identity = soul.load()
    logger.debug(f"Identity loaded: agent={identity['agent_name']}")

    # Create kernel
    kernel = Kernel(config=config)

    # Setup logging middleware
    event_logger = EventLogger(log_dir=get_arc_home() / "logs")
    kernel.use(event_logger.middleware)

    # Setup cost tracking
    cost_tracker = CostTracker()
    kernel.use(cost_tracker.middleware)

    # Setup LLM
    llm = OllamaProvider(
        base_url=config.llm.base_url,
        model=config.llm.default_model,
    )
    logger.info(f"LLM: {config.llm.default_model} at {config.llm.base_url}")

    # Setup skills — auto-discovered from builtins + ~/.arc/skills/*.py
    skill_manager = SkillManager(kernel)
    for skill in discover_skills():
        await skill_manager.register(skill)
    logger.debug(f"Skills registered: {skill_manager.skill_names}")

    # Inject scheduler store into SchedulerSkill (discovered automatically)
    sched_store = SchedulerStore(db_path=Path(config.scheduler.db_path).expanduser())
    if config.scheduler.enabled:
        await sched_store.initialize()
        sched_skill = skill_manager.get_skill("scheduler")
        if sched_skill and isinstance(sched_skill, SchedulerSkill):
            sched_skill.set_store(sched_store)
        logger.info("Scheduler store initialised")

    # Setup security
    security = SecurityEngine(config.security, kernel)

    # Build system prompt with environment info
    env_info = (
        f"\n\nEnvironment:\n"
        f"- OS: {plat.system()} {plat.release()}\n"
        f"- Working directory: {Path.cwd()}\n"
        f"- Shell: {'PowerShell' if plat.system() == 'Windows' else 'Bash'}\n"
    )

    research_strategy = (
        "\n\nWeb Research Strategy:\n"
        "- For any question that needs web data: run ONE web_search, "
        "then read at most 2-3 of the most relevant URLs with web_read, "
        "then synthesize everything and give your answer. Stop there.\n"
        "- Do NOT loop: search → read → search → read. One search is almost always enough.\n"
        "- For live data (prices, rates, weather): prefer http_get against a known API URL "
        "instead of going through a search + read cycle.\n"
        "- Once you have enough information to answer, stop calling tools and respond."
    )

    delegation_strategy = (
        "\n\nDelegation Strategy (delegate_task vs. doing it yourself):\n"
        "Do it YOURSELF (inline) when:\n"
        "- The task needs ONE web search + 2-3 reads and a quick answer.\n"
        "- The user wants an instant response and the work takes <30 seconds.\n"
        "- It is a simple lookup, calculation, or short file read.\n\n"
        "DELEGATE to a background worker when:\n"
        "- The task requires many tool calls or multiple rounds of searching/reading.\n"
        "- The user asks to research multiple things in parallel (start one worker per topic).\n"
        "- The task is explicitly long-running: 'analyse this whole codebase', "
        "'monitor X for the next hour', 'compile a detailed report'.\n"
        "- The user wants to keep chatting while something runs in the background.\n\n"
        "When you delegate: call delegate_task ONCE, then immediately reply to the user "
        "in plain text confirming what you delegated. Do NOT call any other tools "
        "after delegating — especially not list_workers. The result arrives automatically."
    )

    # Soft skills — content of ~/.arc/skills/*.md injected as extra instructions
    soft_skill_text = discover_soft_skills()

    system_prompt = identity["system_prompt"] + env_info + research_strategy + delegation_strategy + soft_skill_text

    # Setup long-term memory
    mem_db_path = get_arc_home() / "memory" / "memory.db"
    memory_manager = MemoryManager(db_path=str(mem_db_path))
    try:
        await memory_manager.initialize()
        logger.info("Long-term memory initialized")
    except Exception as e:
        logger.warning(f"Long-term memory init failed (running without it): {e}")
        memory_manager = None  # type: ignore[assignment]

    # Create agent
    agent = AgentLoop(
        kernel=kernel,
        llm=llm,
        skill_manager=skill_manager,
        security=security,
        system_prompt=system_prompt,
        config=AgentConfig(
            max_iterations=config.agent.max_iterations,
            temperature=config.agent.temperature,
        ),
        memory_manager=memory_manager,
    )

    # Sub-agent factory for the scheduler.
    # Each scheduled job gets an independent AgentLoop (clean session,
    # no shared conversation history) but shares the same Kernel, LLM,
    # SkillManager and SecurityEngine — so all registered tools are
    # available to the job.  This is the extension point for multi-agent:
    # different job types can use different factory functions.
    sub_agent_system_prompt = (
        "You are a proactive background assistant completing a scheduled task. "
        "Use tools as needed to fulfil the task fully and accurately. "
        "Return a concise, well-structured answer — do not ask follow-up questions."
        + env_info
    )

    def make_sub_agent(agent_id: str = "scheduler") -> AgentLoop:
        return AgentLoop(
            kernel=kernel,
            llm=llm,
            skill_manager=skill_manager,
            security=SecurityEngine.make_permissive(kernel),
            system_prompt=sub_agent_system_prompt,
            config=AgentConfig(
                max_iterations=config.agent.max_iterations,
                temperature=0.5,
                excluded_skills=frozenset({"scheduler"}),
            ),
            memory_manager=None,  # sub-agents have no long-term memory
            agent_id=agent_id,  # distinct source so events are filtered from main chat
        )

    # Create CLI platform
    cli = CLIPlatform(
        console=console,
        agent_name=identity["agent_name"],
        user_name=identity["user_name"],
    )

    # Multi-agent infrastructure
    agent_registry = AgentRegistry()
    escalation_bus = EscalationBus(kernel)

    cli.set_approval_flow(agent.security.approval_flow)
    cli.set_escalation_bus(escalation_bus)
    cli.set_skill_manager(skill_manager)
    if memory_manager is not None:
        cli.set_memory_manager(memory_manager)

    # Queue that bridges scheduler → CLI injection (no interleaving).
    # CLIChannel puts completed job results here; CLIPlatform drains it
    # before each user turn so the main agent sees them as context.
    from arc.notifications.base import Notification as _Notification
    pending_queue: asyncio.Queue[_Notification] = asyncio.Queue()

    # Setup notification router
    cli_channel = CLIChannel(pending_queue)
    notification_router = NotificationRouter()
    if config.telegram.configured:
        notification_router.register(TelegramChannel(config.telegram.token, config.telegram.chat_id))
        logger.info("Telegram notification channel registered")
    notification_router.register(cli_channel)
    notification_router.register(FileChannel(get_arc_home() / "notifications.log"))

    # Always wire the pending queue — workers AND scheduler both deliver here.
    # Must happen before scheduler setup so the reference is consistent.
    cli.set_pending_queue(pending_queue)

    # Setup scheduler engine
    scheduler_engine: SchedulerEngine | None = None
    if config.scheduler.enabled:
        scheduler_engine = SchedulerEngine(
            store=sched_store,
            llm=llm,
            agent_factory=make_sub_agent,
            router=notification_router,
            kernel=kernel,
            agent_registry=agent_registry,
        )
        cli.set_scheduler_store(sched_store)

    # Inject WorkerSkill dependencies — must happen after notification_router is built
    worker_skill = skill_manager.get_skill("worker")
    if worker_skill and isinstance(worker_skill, WorkerSkill):
        worker_skill.set_dependencies(
            llm=llm,
            skill_manager=skill_manager,
            escalation_bus=escalation_bus,
            notification_router=notification_router,
            agent_registry=agent_registry,
        )
        logger.info("WorkerSkill dependencies injected")

    # Worker activity logger — writes to ~/.arc/worker_activity.log
    # Subscribe BEFORE forward_to_cli so worker events are logged even though
    # they are filtered out from the main chat window.
    worker_log = WorkerActivityLog(get_arc_home() / "worker_activity.log")
    worker_log.open()

    # Worker-internal event types that must NOT bleed into the main chat.
    # Any agent that isn't the main agent (workers, scheduler sub-agents) tags
    # events with source != "main".  Only coordination events (spawned /
    # complete / escalation) from those agents pass through to the CLI.
    _WORKER_INTERNAL: frozenset[str] = frozenset({
        EventType.AGENT_THINKING,
        EventType.SKILL_TOOL_CALL,
        EventType.SKILL_TOOL_RESULT,
        EventType.LLM_REQUEST,
        EventType.LLM_CHUNK,
        EventType.LLM_RESPONSE,
    })

    # Connect events to CLI for status display
    async def forward_to_cli(event: Event) -> None:
        if event.source != "main" and event.type in _WORKER_INTERNAL:
            return  # suppress — sub-agent internals never reach the main chat window
        cli.on_event(event)

    kernel.on(EventType.AGENT_THINKING, forward_to_cli)
    kernel.on(EventType.SKILL_TOOL_CALL, forward_to_cli)
    kernel.on(EventType.SKILL_TOOL_RESULT, forward_to_cli)
    kernel.on(EventType.SECURITY_APPROVAL, forward_to_cli)
    kernel.on(EventType.AGENT_ESCALATION, forward_to_cli)
    kernel.on(EventType.AGENT_SPAWNED, forward_to_cli)
    kernel.on(EventType.AGENT_TASK_COMPLETE, forward_to_cli)

    # Worker activity log — captures everything workers do silently
    kernel.on(EventType.AGENT_SPAWNED,      worker_log.handle)
    kernel.on(EventType.AGENT_THINKING,     worker_log.handle)
    kernel.on(EventType.SKILL_TOOL_CALL,    worker_log.handle)
    kernel.on(EventType.SKILL_TOOL_RESULT,  worker_log.handle)
    kernel.on(EventType.AGENT_TASK_COMPLETE, worker_log.handle)
    kernel.on(EventType.AGENT_ERROR,        worker_log.handle)

    # Message handler
    async def handle_message(user_input: str):
        # Update cost tracker reference before each message
        cli.set_cost_tracker(cost_tracker.summary())
        logger.info(f"User input: {user_input[:100]}...")

        async for chunk in agent.run(user_input):
            yield chunk
        
        # Update cost tracker after message completes
        cli.set_cost_tracker(cost_tracker.summary())
        logger.info(f"Response complete. Tokens: {cost_tracker.total_tokens}")

    # Run
    try:
        await kernel.start()
        if scheduler_engine:
            await scheduler_engine.start()
        cli_channel.set_active(True)
        await cli.run(handle_message)
    except Exception as e:
        logger.exception(f"Error in chat session: {e}")
        raise
    finally:
        logger.info("Shutting down")
        cli_channel.set_active(False)
        worker_log.close()
        # Cancel all background agents before anything else closes
        await agent_registry.shutdown_all()
        if scheduler_engine:
            await scheduler_engine.stop()
        if config.scheduler.enabled:
            await sched_store.close()
        await skill_manager.shutdown_all()
        await kernel.stop()
        await llm.close()
        if memory_manager is not None:
            await memory_manager.close()


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


if __name__ == "__main__":
    app()