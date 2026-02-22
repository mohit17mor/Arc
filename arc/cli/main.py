"""
Arc CLI entry point.

Commands:
    arc init  — First-time setup
    arc chat  — Interactive chat
"""

from __future__ import annotations

import asyncio
import typer
from rich.console import Console
from pathlib import Path

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
) -> None:
    """Start an interactive chat session."""
    asyncio.run(_run_chat(model))


async def _run_chat(model_override: str | None) -> None:
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

    config_path = get_config_path()
    identity_path = get_identity_path()

    # Check if configured
    if not config_path.exists():
        console.print(
            "[yellow]Arc is not configured yet.[/yellow]\n"
            "[dim]Run [bold]arc init[/bold] first.[/dim]"
        )
        raise typer.Exit(1)

    # Load config
    config = ArcConfig.load()

    # Override model if specified
    if model_override:
        config.llm.default_model = model_override

    # Load identity
    soul = SoulManager(identity_path)
    identity = soul.load()

    # Create kernel
    kernel = Kernel(config=config)

    # Setup cost tracking
    cost_tracker = CostTracker()
    kernel.use(cost_tracker.middleware)

    # Setup LLM
    llm = OllamaProvider(
        base_url=config.llm.base_url,
        model=config.llm.default_model,
    )

    # Setup skills
    skill_manager = SkillManager(kernel)
    await skill_manager.register(FilesystemSkill())
    await skill_manager.register(TerminalSkill())

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

    # Connect events to CLI for status display
    async def forward_to_cli(event: Event) -> None:
        cli.on_event(event)

    kernel.on(EventType.AGENT_THINKING, forward_to_cli)
    kernel.on(EventType.SKILL_TOOL_CALL, forward_to_cli)
    kernel.on(EventType.SKILL_TOOL_RESULT, forward_to_cli)

    # Message handler
    async def handle_message(user_input: str):
        # Update cost tracker reference before each message
        cli.set_cost_tracker(cost_tracker.summary())

        async for chunk in agent.run(user_input):
            yield chunk
        
        # Update cost tracker after message completes
        cli.set_cost_tracker(cost_tracker.summary())

    # Run
    try:
        await kernel.start()
        await cli.run(handle_message)
    finally:
        await skill_manager.shutdown_all()
        await kernel.stop()
        await llm.close()


@app.command()
def version() -> None:
    """Show Arc version."""
    from arc import __version__
    console.print(f"Arc v{__version__}")


if __name__ == "__main__":
    app()