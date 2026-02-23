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
    from arc.skills.builtin.filesystem import FilesystemSkill
    from arc.skills.builtin.terminal import TerminalSkill
    from arc.security.engine import SecurityEngine
    from arc.agent.loop import AgentLoop, AgentConfig
    from arc.identity.soul import SoulManager
    from arc.platforms.cli.app import CLIPlatform
    from arc.middleware.cost import CostTracker
    from arc.middleware.logging import setup_logging, EventLogger

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

    # Setup skills
    skill_manager = SkillManager(kernel)
    await skill_manager.register(FilesystemSkill())
    await skill_manager.register(TerminalSkill())
    logger.debug(f"Skills registered: {skill_manager.skill_names}")

    # Setup security
    security = SecurityEngine(config.security, kernel)

    # Build system prompt with environment info
    env_info = (
        f"\n\nEnvironment:\n"
        f"- OS: {plat.system()} {plat.release()}\n"
        f"- Working directory: {Path.cwd()}\n"
        f"- Shell: {'PowerShell' if plat.system() == 'Windows' else 'Bash'}\n"
    )

    system_prompt = identity["system_prompt"] + env_info

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
    )

    # Create CLI platform
    cli = CLIPlatform(
        console=console,
        agent_name=identity["agent_name"],
        user_name=identity["user_name"],
    )

    cli.set_approval_flow(agent.security.approval_flow)

    # Connect events to CLI for status display
    async def forward_to_cli(event: Event) -> None:
        cli.on_event(event)

    kernel.on(EventType.AGENT_THINKING, forward_to_cli)
    kernel.on(EventType.SKILL_TOOL_CALL, forward_to_cli)
    kernel.on(EventType.SKILL_TOOL_RESULT, forward_to_cli)
    kernel.on(EventType.SECURITY_APPROVAL, forward_to_cli)

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
        await cli.run(handle_message)
    except Exception as e:
        logger.exception(f"Error in chat session: {e}")
        raise
    finally:
        logger.info("Shutting down")
        await skill_manager.shutdown_all()
        await kernel.stop()
        await llm.close()


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