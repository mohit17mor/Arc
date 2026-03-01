"""
First-run setup — interactive onboarding experience.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from arc.identity.personality import list_personalities, get_personality
from arc.identity.soul import SoulManager


def _pick_provider(console: Console) -> dict:
    """Interactive provider selection menu. Returns dict with provider config."""
    from arc.llm.factory import get_presets

    presets = get_presets()
    providers = list(presets.items())

    console.print("[bold]Which LLM provider do you want to use?[/bold]")
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, (key, preset) in enumerate(providers, 1):
        needs_key = " [dim](API key required)[/dim]" if preset["needs_key"] else ""
        table.add_row(
            f"[bold cyan]{i}[/bold cyan]",
            f"[bold]{preset['label']}[/bold]{needs_key}",
        )
    console.print(table)
    console.print()

    choice = Prompt.ask(
        "[bold]Pick a number[/bold]",
        choices=[str(i) for i in range(1, len(providers) + 1)],
        default="1",
    )
    provider_name, preset = providers[int(choice) - 1]
    console.print(f"[green]✓[/green] {preset['label']}")
    console.print()

    result: dict = {"provider": provider_name}

    # Base URL — editable for local/custom, shown as default for cloud
    if provider_name in ("ollama", "lmstudio", "custom"):
        default_url = preset["base_url"] or "http://localhost:8000/v1"
        result["base_url"] = Prompt.ask(
            "[bold]Base URL[/bold]",
            default=default_url,
        )
        console.print()
    else:
        result["base_url"] = preset["base_url"]

    # API key
    if preset["needs_key"]:
        result["api_key"] = Prompt.ask(
            f"[bold]API key for {preset['label']}[/bold]",
            default="",
        )
        console.print()
    else:
        result["api_key"] = ""

    # Model name
    result["model"] = Prompt.ask(
        "[bold]Which model should I use?[/bold]",
        default=preset["default_model"],
    )
    console.print()

    return result


def _pick_worker_model(console: Console) -> dict | None:
    """Optionally configure a separate model for background workers."""
    console.print(
        "[bold]Worker model[/bold]\n"
        "[dim]Background workers (delegated tasks) can use a cheaper / faster "
        "model to save costs. Leave this blank to use the same model as above.[/dim]"
    )
    console.print()

    use_separate = Confirm.ask(
        "Configure a separate worker model?",
        default=False,
    )
    console.print()

    if not use_separate:
        return None

    return _pick_provider(console)


def run_first_time_setup(
    config_path: Path,
    identity_path: Path,
    console: Console | None = None,
) -> dict:
    """
    Run the first-time setup wizard.

    Returns dict with configuration values.
    """
    console = console or Console()

    # Banner
    console.print()
    console.print(
        Panel(
            "[bold cyan]"
            "   █████╗ ██████╗  ██████╗\n"
            "  ██╔══██╗██╔══██╗██╔════╝\n"
            "  ███████║██████╔╝██║     \n"
            "  ██╔══██║██╔══██╗██║     \n"
            "  ██║  ██║██║  ██║╚██████╗\n"
            "  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝\n"
            "[/bold cyan]\n"
            "[dim]Welcome, human. Let's get acquainted.[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # ── Identity ──────────────────────────────────────────────
    user_name = Prompt.ask(
        "[bold]What should I call you?[/bold]",
        default="User",
    )
    console.print()

    agent_name = Prompt.ask(
        f"[bold]Nice to meet you, {user_name}! Now give me a name[/bold]",
        default="Arc",
    )
    console.print()

    # ── Personality ───────────────────────────────────────────
    console.print(f"[bold]Alright, I'm {agent_name}. What kind of AI am I?[/bold]")
    console.print()

    personalities = list_personalities()
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, p in enumerate(personalities, 1):
        table.add_row(
            f"[bold cyan]{i}[/bold cyan]",
            f"{p.emoji} [bold]{p.name}[/bold]",
        )
        table.add_row("", f"[dim]{p.description}[/dim]")
        table.add_row("", "")
    console.print(table)

    choice = Prompt.ask(
        "[bold]Pick a number[/bold]",
        choices=[str(i) for i in range(1, len(personalities) + 1)],
        default="1",
    )
    personality = personalities[int(choice) - 1]
    console.print()
    console.print(f"[green]✓[/green] {personality.emoji} {personality.name} it is!")
    console.print()

    # ── LLM Provider ─────────────────────────────────────────
    provider_cfg = _pick_provider(console)

    # ── Worker Model (optional) ──────────────────────────────
    worker_cfg = _pick_worker_model(console)

    # ── Connection test ──────────────────────────────────────
    console.print("[dim]Testing connection...[/dim]")
    try:
        from arc.llm.factory import create_llm
        test_llm = create_llm(
            provider_cfg["provider"],
            model=provider_cfg["model"],
            base_url=provider_cfg["base_url"],
            api_key=provider_cfg["api_key"],
        )
        info = test_llm.get_model_info()
        console.print(
            f"[green]✓[/green] Connected to {info.provider}/{info.model}"
        )
    except Exception as e:
        console.print(
            f"[yellow]⚠ Could not verify connection: {e}[/yellow]\n"
            "[dim]Config saved anyway — you can fix it later in ~/.arc/config.toml[/dim]"
        )
    console.print()

    # ── Create identity file ─────────────────────────────────
    soul = SoulManager(identity_path)
    soul.create(agent_name, user_name, personality.id)

    # ── Create config file ───────────────────────────────────
    config_lines = [
        "# Arc Configuration",
        "# Generated by arc init",
        "",
        "[llm]",
        f'default_provider = "{provider_cfg["provider"]}"',
        f'default_model = "{provider_cfg["model"]}"',
        f'base_url = "{provider_cfg["base_url"]}"',
    ]
    if provider_cfg["api_key"]:
        config_lines.append(f'api_key = "{provider_cfg["api_key"]}"')

    if worker_cfg:
        config_lines += [
            "",
            "# Worker model (used by background workers / sub-agents)",
            f'worker_provider = "{worker_cfg["provider"]}"',
            f'worker_model = "{worker_cfg["model"]}"',
            f'worker_base_url = "{worker_cfg["base_url"]}"',
        ]
        if worker_cfg["api_key"]:
            config_lines.append(f'worker_api_key = "{worker_cfg["api_key"]}"')

    config_lines += [
        "",
        "[identity]",
        f'user_name = "{user_name}"',
        f'agent_name = "{agent_name}"',
        f'personality = "{personality.id}"',
        "",
        "[security]",
        'auto_allow = ["file:read"]',
        'always_ask = ["file:write", "file:delete", "shell:exec"]',
        "",
    ]

    config_content = "\n".join(config_lines)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content, encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────
    model_info = f"{provider_cfg['provider']}/{provider_cfg['model']}"
    if worker_cfg:
        model_info += f"\n[green]✓[/green] Worker: {worker_cfg['provider']}/{worker_cfg['model']}"

    console.print(
        Panel(
            f"[green]✓[/green] Identity saved to [cyan]{identity_path}[/cyan]\n"
            f"[green]✓[/green] Config saved to [cyan]{config_path}[/cyan]\n"
            f"[green]✓[/green] Model: {model_info}\n"
            "\n"
            "[bold]You're all set! Here's what you can do:[/bold]\n"
            "\n"
            "  [cyan]arc chat[/cyan]          — Talk to me\n"
            "  [cyan]arc run <recipe>[/cyan]  — Run a micro-agent\n"
            "  [cyan]arc teach <name>[/cyan]  — Teach me a new task\n"
            "\n"
            f"[dim]Try: [bold]arc chat[/bold][/dim]",
            title="[bold green]Setup Complete[/bold green]",
            border_style="green",
        )
    )

    result = {
        "user_name": user_name,
        "agent_name": agent_name,
        "personality": personality.id,
        "provider": provider_cfg["provider"],
        "model": provider_cfg["model"],
        "base_url": provider_cfg["base_url"],
        "api_key": provider_cfg["api_key"],
    }
    if worker_cfg:
        result["worker_provider"] = worker_cfg["provider"]
        result["worker_model"] = worker_cfg["model"]
        result["worker_base_url"] = worker_cfg["base_url"]
        result["worker_api_key"] = worker_cfg["api_key"]
    return result