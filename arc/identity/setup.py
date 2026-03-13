"""
First-run setup — interactive onboarding experience.

Uses questionary for arrow-key navigable menus (like OpenClaw).
Falls back to plain Rich prompts when running in non-interactive
contexts (CI, piped stdin — e.g. tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from arc.identity.personality import list_personalities, get_personality
from arc.identity.soul import SoulManager


# ── Model catalogs per provider ──────────────────────────────────
# Popular models available on each provider.  The first model in each
# list is the recommended default.

MODEL_CATALOG: dict[str, list[dict[str, str]]] = {
    "ollama": [
        {"value": "llama3.2",              "name": "Llama 3.2 (3B) — fast, good for most tasks"},
        {"value": "llama3.1",              "name": "Llama 3.1 (8B) — well-rounded"},
        {"value": "llama3.3",              "name": "Llama 3.3 (70B) — most capable, slower"},
        {"value": "qwen2.5-coder:7b",     "name": "Qwen 2.5 Coder (7B) — optimised for code"},
        {"value": "qwen2.5-coder:32b",    "name": "Qwen 2.5 Coder (32B) — best open code model"},
        {"value": "deepseek-r1:8b",        "name": "DeepSeek R1 (8B) — strong reasoning"},
        {"value": "deepseek-r1:70b",       "name": "DeepSeek R1 (70B) — top-tier reasoning"},
        {"value": "mistral",               "name": "Mistral (7B) — fast, lightweight"},
        {"value": "gemma2",                "name": "Gemma 2 (9B) — Google's open model"},
        {"value": "phi3",                  "name": "Phi-3 (3.8B) — Microsoft, very small"},
    ],
    "openai": [
        {"value": "gpt-4o",               "name": "GPT-4o — best overall"},
        {"value": "gpt-4o-mini",           "name": "GPT-4o Mini — fast & cheap"},
        {"value": "gpt-4.1",              "name": "GPT-4.1 — latest flagship"},
        {"value": "gpt-4.1-mini",         "name": "GPT-4.1 Mini — latest efficient"},
        {"value": "gpt-4.1-nano",         "name": "GPT-4.1 Nano — ultra-fast"},
        {"value": "o1",                    "name": "o1 — advanced reasoning"},
        {"value": "o1-mini",               "name": "o1-mini — fast reasoning"},
        {"value": "o3-mini",               "name": "o3-mini — latest reasoning"},
    ],
    "openrouter": [
        {"value": "anthropic/claude-sonnet-4-20250514",    "name": "Claude Sonnet 4 — best coding model"},
        {"value": "anthropic/claude-3.5-sonnet",           "name": "Claude 3.5 Sonnet — fast & capable"},
        {"value": "anthropic/claude-3.5-haiku",            "name": "Claude 3.5 Haiku — ultra-fast, cheap"},
        {"value": "google/gemini-2.0-flash-001",           "name": "Gemini 2.0 Flash — fast, free tier"},
        {"value": "google/gemini-2.5-pro-preview",         "name": "Gemini 2.5 Pro — most capable"},
        {"value": "openai/gpt-4o",                         "name": "GPT-4o (via OpenRouter)"},
        {"value": "openai/gpt-4o-mini",                    "name": "GPT-4o Mini (via OpenRouter)"},
        {"value": "deepseek/deepseek-r1",                  "name": "DeepSeek R1 — top reasoning, cheap"},
        {"value": "meta-llama/llama-3.3-70b-instruct",     "name": "Llama 3.3 70B — open, strong"},
        {"value": "qwen/qwen-2.5-coder-32b-instruct",     "name": "Qwen 2.5 Coder 32B — best open code"},
    ],
    "groq": [
        {"value": "llama-3.3-70b-versatile",     "name": "Llama 3.3 70B — most capable"},
        {"value": "llama-3.1-8b-instant",         "name": "Llama 3.1 8B Instant — ultra-fast"},
        {"value": "llama-3.2-3b-preview",         "name": "Llama 3.2 3B — smallest, fastest"},
        {"value": "llama-3.2-11b-vision-preview",  "name": "Llama 3.2 11B Vision — multimodal"},
        {"value": "mixtral-8x7b-32768",            "name": "Mixtral 8x7B — 32K context"},
        {"value": "gemma2-9b-it",                  "name": "Gemma 2 9B — Google's model on Groq"},
        {"value": "deepseek-r1-distill-llama-70b", "name": "DeepSeek R1 Distill 70B — reasoning"},
    ],
    "together": [
        {"value": "meta-llama/Llama-3.3-70B-Instruct-Turbo",     "name": "Llama 3.3 70B Turbo — fast & capable"},
        {"value": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  "name": "Llama 3.1 8B Turbo — cheap"},
        {"value": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo","name": "Llama 3.1 405B Turbo — largest open model"},
        {"value": "Qwen/Qwen2.5-Coder-32B-Instruct",              "name": "Qwen 2.5 Coder 32B — code specialist"},
        {"value": "deepseek-ai/DeepSeek-R1",                       "name": "DeepSeek R1 — reasoning"},
        {"value": "google/gemma-2-27b-it",                         "name": "Gemma 2 27B — Google"},
        {"value": "mistralai/Mixtral-8x22B-Instruct-v0.1",        "name": "Mixtral 8x22B — large MoE"},
    ],
    "lmstudio": [
        {"value": "default",  "name": "Default loaded model"},
    ],
    "custom": [],
}


def _is_interactive() -> bool:
    """Check if we're in an interactive terminal (not piped stdin)."""
    import sys
    return sys.stdin.isatty()


def _q_select(message: str, choices: list[dict[str, str]], default: str | None = None) -> str:
    """
    Arrow-key select using questionary.

    Each choice is {"value": "...", "name": "display text"}.
    Returns the value string.
    """
    import questionary
    from questionary import Choice, Style

    style = Style([
        ("qmark",       "fg:cyan bold"),
        ("question",    "bold"),
        ("pointer",     "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected",    "fg:green"),
        ("answer",      "fg:green bold"),
    ])

    q_choices = [
        Choice(title=c["name"], value=c["value"])
        for c in choices
    ]

    # If the default isn't among the choices, clear it to avoid ValueError
    choice_values = {c["value"] for c in choices}
    safe_default = default if default in choice_values else None

    result = questionary.select(
        message,
        choices=q_choices,
        default=safe_default,
        style=style,
        use_arrow_keys=True,
        use_shortcuts=False,
    ).ask()

    if result is None:
        raise KeyboardInterrupt("Setup cancelled.")
    return result


def _q_text(message: str, default: str = "") -> str:
    """Text input using questionary."""
    import questionary
    from questionary import Style

    style = Style([
        ("qmark",    "fg:cyan bold"),
        ("question", "bold"),
        ("answer",   "fg:green bold"),
    ])

    result = questionary.text(
        message,
        default=default,
        style=style,
    ).ask()

    if result is None:
        raise KeyboardInterrupt("Setup cancelled.")
    return result


def _q_multiline_text(
    console: Console,
    message: str,
    default: str = "",
    terminator: str = "END",
) -> str:
    """
    Collect multiline text until a terminator line is entered.

    If no lines are provided and default is non-empty, the default is returned.
    """
    console.print(f"[bold]{message}[/bold]")
    console.print(
        f"[dim]Enter your prompt. Type '{terminator}' on a new line when done.[/dim]"
    )
    if default.strip():
        console.print(
            f"[dim]Tip: type '{terminator}' immediately to keep the current prompt.[/dim]"
        )

    lines: list[str] = []
    while True:
        line = console.input("[cyan]> [/cyan]")
        if line.strip() == terminator:
            break
        lines.append(line)

    prompt_text = "\n".join(lines).strip()
    if not prompt_text and default.strip():
        return default.strip()
    return prompt_text


def _q_password(message: str) -> str:
    """Password input using questionary (masked)."""
    import questionary
    from questionary import Style

    style = Style([
        ("qmark",    "fg:cyan bold"),
        ("question", "bold"),
        ("answer",   "fg:green bold"),
    ])

    result = questionary.password(
        message,
        style=style,
    ).ask()

    if result is None:
        raise KeyboardInterrupt("Setup cancelled.")
    return result


def _q_confirm(message: str, default: bool = False) -> bool:
    """Yes/No confirm using questionary."""
    import questionary
    from questionary import Style

    style = Style([
        ("qmark",    "fg:cyan bold"),
        ("question", "bold"),
        ("answer",   "fg:green bold"),
    ])

    result = questionary.confirm(
        message,
        default=default,
        style=style,
    ).ask()

    if result is None:
        raise KeyboardInterrupt("Setup cancelled.")
    return result


# ── Provider & model pickers ─────────────────────────────────────


def _pick_provider(
    console: Console,
    interactive: bool = True,
    defaults: dict | None = None,
) -> dict:
    """Interactive provider selection menu. Returns dict with provider config.

    Args:
        defaults: existing values to pre-populate (provider, model, base_url, api_key).
    """
    from arc.llm.factory import get_presets

    presets = get_presets()
    providers = list(presets.items())
    d = defaults or {}
    default_provider = d.get("provider", "ollama")

    if interactive:
        # Arrow-key selection
        provider_choices = []
        for key, preset in providers:
            tag = " 🔑" if preset["needs_key"] else ""
            provider_choices.append({
                "value": key,
                "name": f"{preset['label']}{tag}",
            })

        provider_name = _q_select(
            "Which LLM provider?",
            provider_choices,
            default=default_provider,
        )
        preset = presets[provider_name]
        console.print(f"  [green]✓[/green] {preset['label']}")
        console.print()
    else:
        # Fallback for non-interactive (tests, CI)
        from rich.table import Table
        console.print("[bold]Which LLM provider do you want to use?[/bold]")
        console.print()
        table = Table(show_header=False, box=None, padding=(0, 2))
        for i, (key, preset) in enumerate(providers, 1):
            needs_key = " (API key required)" if preset["needs_key"] else ""
            table.add_row(f"[bold cyan]{i}[/bold cyan]", f"[bold]{preset['label']}[/bold]{needs_key}")
        console.print(table)
        console.print()
        # Pre-select the right number if there's an existing provider
        fallback_idx = "1"
        for i, (key, _) in enumerate(providers, 1):
            if key == default_provider:
                fallback_idx = str(i)
                break
        choice = Prompt.ask("[bold]Pick a number[/bold]",
                            choices=[str(i) for i in range(1, len(providers) + 1)],
                            default=fallback_idx)
        provider_name, preset = providers[int(choice) - 1]
        console.print(f"  [green]✓[/green] {preset['label']}")
        console.print()

    result: dict = {"provider": provider_name}

    # Use existing values as defaults when same provider is re-selected
    same_provider = (provider_name == default_provider)

    # Base URL
    needs_base_url_prompt = provider_name in ("ollama", "lmstudio", "custom") or (
        preset.get("class") == "responses"
    )
    if needs_base_url_prompt:
        fallback_url = preset["base_url"] or "http://localhost:8000/v1"
        if provider_name == "codex" and not preset["base_url"]:
            fallback_url = "https://api.openai.com/v1"
        default_url = (
            d.get("base_url") if d.get("base_url")
            else fallback_url
        )
        if interactive:
            result["base_url"] = _q_text("Base URL:", default=default_url)
        else:
            result["base_url"] = Prompt.ask("[bold]Base URL[/bold]", default=default_url)
        console.print()
    else:
        result["base_url"] = preset["base_url"]

    # API key
    if preset["needs_key"]:
        existing_key = d.get("api_key", "") if same_provider else ""
        if existing_key:
            masked = existing_key[:4] + "•" * (len(existing_key) - 8) + existing_key[-4:] if len(existing_key) > 8 else "••••"
            console.print(f"  [dim]Current key: {masked}[/dim]")
        if interactive:
            new_key = _q_password(
                f"API key for {preset['label']} (Enter to keep current):"
                if existing_key else f"API key for {preset['label']}:"
            )
            result["api_key"] = new_key if new_key else existing_key
        else:
            result["api_key"] = Prompt.ask(
                f"[bold]API key for {preset['label']}[/bold]",
                default=existing_key or "",
            )
        console.print()
    else:
        result["api_key"] = ""

    # Model — fetch live models from API (also validates connection)
    catalog = MODEL_CATALOG.get(provider_name, [])
    default_model = (
        d.get("model") if same_provider and d.get("model")
        else preset["default_model"]
    )

    # Try to fetch real models from the provider API
    live_models: list[str] | None = None
    try:
        from arc.llm.factory import fetch_models
        console.print("[dim]Fetching available models...[/dim]")
        live_models = fetch_models(
            provider_name,
            result.get("base_url", preset["base_url"]),
            result.get("api_key", ""),
        )
        console.print(
            f"  [green]✓[/green] Connected — {len(live_models)} model{'s' if len(live_models) != 1 else ''} available"
        )
        console.print()
    except Exception as fetch_err:
        # Distinguish auth errors from connectivity errors
        import httpx as _httpx
        if isinstance(fetch_err, _httpx.HTTPStatusError) and fetch_err.response.status_code in (401, 403):
            console.print(
                f"  [red]✗ Authentication failed ({fetch_err.response.status_code}).[/red]\n"
                "  [dim]Check your API key. Falling back to suggested models.[/dim]"
            )
        elif isinstance(fetch_err, (_httpx.ConnectError, _httpx.TimeoutException)):
            console.print(
                f"  [yellow]⚠ Could not reach {provider_name}: {fetch_err}[/yellow]\n"
                "  [dim]Your API key could be wrong, or the server may be unreachable.[/dim]\n"
                "  [dim]Falling back to suggested models.[/dim]"
            )
        else:
            console.print(
                f"  [yellow]⚠ Could not fetch models: {fetch_err}[/yellow]\n"
                "  [dim]Your API key could be wrong. Falling back to suggested models.[/dim]"
            )
        console.print()

    # Build the model choice list
    if live_models is not None and len(live_models) > 0:
        # Use real models from the API
        model_choices = [
            {"value": m, "name": m}
            for m in live_models
        ]
    elif catalog:
        # Fallback to hardcoded catalog
        model_choices = list(catalog)
    else:
        model_choices = []

    if interactive and model_choices:
        # If the current model isn't in the list, add it at the top so
        # the user can press Enter to keep it
        choice_values = {c["value"] for c in model_choices}
        if default_model and default_model not in choice_values:
            model_choices.insert(0, {
                "value": default_model,
                "name": f"{default_model} (current)",
            })

        model_choices.append(
            {"value": "__custom__", "name": "✏️  Enter a custom model name"},
        )
        model = _q_select(
            "Which model?",
            model_choices,
            default=default_model,
        )
        if model == "__custom__":
            model = _q_text("Model name:", default=default_model)
        console.print(f"  [green]✓[/green] {model}")
        console.print()
        result["model"] = model
    elif interactive:
        result["model"] = _q_text("Model name:", default=default_model)
        console.print()
    else:
        result["model"] = Prompt.ask("[bold]Which model?[/bold]", default=default_model)
        console.print()

    return result


def _pick_worker_model(
    console: Console,
    interactive: bool = True,
    defaults: dict | None = None,
) -> dict | None:
    """Optionally configure a separate model for background workers."""
    has_existing = bool(defaults)
    console.print(
        "[dim]Background workers can use a cheaper/faster model to save costs.[/dim]"
    )

    if interactive:
        use_separate = _q_confirm(
            "Configure a separate worker model?",
            default=has_existing,
        )
    else:
        use_separate = Confirm.ask(
            "Configure a separate worker model?",
            default=has_existing,
        )
    console.print()

    if not use_separate:
        return None

    return _pick_provider(console, interactive=interactive, defaults=defaults)


def _pick_tavily(
    console: Console,
    interactive: bool = True,
    existing_key: str = "",
    existing_ngrok: str = "",
) -> dict | None:
    """Optionally configure Tavily API and ngrok for Liquid Web."""
    from rich.prompt import Confirm, Prompt

    console.print(
        "[dim]Liquid Web lets Arc search products across the web and render\n"
        "a beautiful comparison UI. It uses the Tavily API (free tier: 1000 searches/month).\n"
        "Get a key at [bold]https://tavily.com[/bold][/dim]"
    )

    has_existing = bool(existing_key)
    if interactive:
        enable = _q_confirm(
            "Enable Liquid Web (Tavily search)?",
            default=has_existing,
        )
    else:
        enable = Confirm.ask(
            "Enable Liquid Web (Tavily search)?",
            default=has_existing,
        )
    console.print()

    if not enable:
        return None

    if interactive:
        api_key = _q_password("Tavily API key:") if not existing_key else _q_text(
            "Tavily API key:", default=existing_key
        )
    else:
        api_key = Prompt.ask("[bold]Tavily API key[/bold]", default=existing_key or "")

    if not api_key.strip():
        console.print("  [yellow]⚠ No key provided — Liquid Web disabled.[/yellow]")
        console.print()
        return None

    console.print("  [green]✓[/green] Tavily API configured — Liquid Web enabled!")
    console.print()

    # Ngrok — optional, needed for public URLs (Telegram, sharing)
    console.print(
        "[dim]Ngrok creates a public URL so the comparison page is accessible\n"
        "from Telegram, mobile, or anywhere. Free at [bold]https://ngrok.com[/bold]\n"
        "Without it, results are only available on localhost.[/dim]"
    )
    if interactive:
        ngrok_token = _q_text(
            "Ngrok auth token (blank to skip):",
            default=existing_ngrok,
        )
    else:
        ngrok_token = Prompt.ask(
            "[bold]Ngrok auth token (blank to skip)[/bold]",
            default=existing_ngrok or "",
        )
    console.print()

    if ngrok_token.strip():
        console.print("  [green]✓[/green] Ngrok configured — public URLs enabled!")
    else:
        console.print("  [dim]No ngrok token — results will be on localhost only.[/dim]")
    console.print()

    return {"api_key": api_key.strip(), "ngrok_token": ngrok_token.strip()}


def _pick_telegram(
    console: Console,
    interactive: bool = True,
    existing_token: str = "",
    existing_chat_id: str = "",
    existing_allowed: str = "",
) -> dict | None:
    """Optionally configure Telegram bot for bidirectional chat."""
    from rich.prompt import Confirm, Prompt

    console.print(
        "[dim]Telegram lets you chat with Arc from your phone via a Telegram bot.\n"
        "Create a bot with [bold]@BotFather[/bold] on Telegram to get a token.\n"
        "Send /start to your bot, then use @userinfobot to get your chat_id.[/dim]"
    )

    has_existing = bool(existing_token)
    if interactive:
        enable = _q_confirm(
            "Configure Telegram bot?",
            default=has_existing,
        )
    else:
        enable = Confirm.ask(
            "Configure Telegram bot?",
            default=has_existing,
        )
    console.print()

    if not enable:
        return None

    if interactive:
        token = _q_text("Telegram bot token:", default=existing_token)
        chat_id = _q_text(
            "Your Telegram chat_id (for notifications):",
            default=existing_chat_id,
        )
        allowed = _q_text(
            "Allowed user chat_ids (comma-separated, blank=everyone):",
            default=existing_allowed,
        )
    else:
        token = Prompt.ask("[bold]Telegram bot token[/bold]", default=existing_token or "")
        chat_id = Prompt.ask(
            "[bold]Your Telegram chat_id[/bold]", default=existing_chat_id or ""
        )
        allowed = Prompt.ask(
            "[bold]Allowed user chat_ids (comma-separated)[/bold]",
            default=existing_allowed or "",
        )

    if not token.strip():
        console.print("  [yellow]⚠ No token provided — Telegram disabled.[/yellow]")
        console.print()
        return None

    console.print("  [green]✓[/green] Telegram bot configured!")
    console.print()
    return {
        "token": token.strip(),
        "chat_id": chat_id.strip(),
        "allowed_users": [u.strip() for u in allowed.split(",") if u.strip()],
    }


# ── Section menu (reconfiguration) ───────────────────────────────

SETUP_SECTIONS = [
    {"value": "identity", "name": "🤖  Identity — name & personality"},
    {"value": "models",   "name": "🧠  Models — LLM provider, model & worker"},
    {"value": "tavily",   "name": "🌐  Liquid Web — Tavily search API"},
    {"value": "telegram", "name": "📱  Telegram — bot platform"},
    {"value": "all",      "name": "⚙️   Everything — full reconfiguration"},
    {"value": "exit",     "name": "Done"},
]


def _show_current_config(console: Console, e: dict) -> None:
    """Display a compact summary of current configuration."""
    lines = []

    # Identity
    user = e.get("user_name", "User")
    agent = e.get("agent_name", "Arc")
    pers = e.get("personality", "helpful")
    lines.append(f"  🤖 [bold]Identity:[/bold]   {user} → {agent} ({pers})")

    # Models
    provider = e.get("provider", "?")
    model = e.get("model", "?")
    model_str = f"{provider}/{model}"
    if e.get("worker_provider"):
        model_str += f"  [dim]worker: {e['worker_provider']}/{e.get('worker_model', '?')}[/dim]"
    lines.append(f"  🧠 [bold]Models:[/bold]     {model_str}")

    # Tavily
    if e.get("tavily_api_key"):
        ngrok_status = "  + ngrok" if e.get("ngrok_auth_token") else "  [dim](localhost only)[/dim]"
        lines.append(f"  🌐 [bold]Liquid Web:[/bold]  [green]✓ enabled[/green]{ngrok_status}")
    else:
        lines.append("  🌐 [bold]Liquid Web:[/bold]  [dim]✗ not configured[/dim]")

    # Telegram
    if e.get("telegram_token"):
        lines.append("  📱 [bold]Telegram:[/bold]   [green]✓ enabled[/green]")
    else:
        lines.append("  📱 [bold]Telegram:[/bold]   [dim]✗ not configured[/dim]")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Current Configuration[/bold]",
            border_style="dim",
        )
    )
    console.print()


def _pick_section(console: Console, interactive: bool = True) -> str:
    """Let the user choose which section to reconfigure."""
    if interactive:
        return _q_select(
            "What would you like to configure?",
            SETUP_SECTIONS,
            default="all",
        )
    else:
        from rich.table import Table

        console.print("[bold]What would you like to reconfigure?[/bold]")
        console.print()
        table = Table(show_header=False, box=None, padding=(0, 2))
        for i, s in enumerate(SETUP_SECTIONS, 1):
            table.add_row(f"[bold cyan]{i}[/bold cyan]", s["name"])
        console.print(table)
        console.print()
        choice = Prompt.ask(
            "[bold]Pick a number[/bold]",
            choices=[str(i) for i in range(1, len(SETUP_SECTIONS) + 1)],
            default=str(len(SETUP_SECTIONS)),  # default = "Everything"
        )
        return SETUP_SECTIONS[int(choice) - 1]["value"]


# ── Main setup wizard ────────────────────────────────────────────


def run_first_time_setup(
    config_path: Path,
    identity_path: Path,
    console: Console | None = None,
    existing: dict | None = None,
) -> dict:
    """
    Run the first-time setup wizard.

    On first run, walks through all sections.  On reconfiguration,
    shows a section menu so the user can update only what they need.

    Args:
        existing: previously saved config values to use as defaults.

    Returns dict with configuration values.
    """
    console = console or Console()
    interactive = _is_interactive()
    e = existing or {}  # shorthand
    is_reconfig = bool(e)

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
            "[dim]" + (
                "Select a section to reconfigure."
                if is_reconfig else
                "Welcome, human. Let's get acquainted."
            ) + "[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # ── Section selection (reconfigure only) ──────────────────
    ALL_SECTIONS = frozenset({"identity", "models", "tavily", "telegram"})

    if is_reconfig:
        # Loop: pick section → configure → save → back to menu
        while True:
            _show_current_config(console, e)
            section = _pick_section(console, interactive)
            console.print()
            if section == "exit":
                console.print("  [dim]No changes made.[/dim]")
                return e
            sections = ALL_SECTIONS if section == "all" else frozenset({section})

            result = _run_sections(
                sections, e, console, interactive,
                config_path, identity_path,
            )

            # Update e with what was just configured so the menu reflects it
            e.update(result)

            # If they chose "Everything", no need to loop
            if section == "all":
                return result
    else:
        sections = ALL_SECTIONS  # first run → everything
        return _run_sections(
            sections, e, console, interactive,
            config_path, identity_path,
        )


def _run_sections(
    sections: frozenset[str],
    e: dict,
    console: Console,
    interactive: bool,
    config_path: Path,
    identity_path: Path,
) -> dict:
    """Run the selected setup sections, save config, and return result dict."""

    # ── Identity ──────────────────────────────────────────────
    if "identity" in sections:
        default_user = e.get("user_name", "User")
        default_agent = e.get("agent_name", "Arc")
        default_personality = e.get("personality", "helpful")

        if interactive:
            user_name = _q_text("What should I call you?", default=default_user)
        else:
            user_name = Prompt.ask("[bold]What should I call you?[/bold]", default=default_user)
        console.print()

        if interactive:
            agent_name = _q_text(f"Nice to meet you, {user_name}! Now give me a name:", default=default_agent)
        else:
            agent_name = Prompt.ask(f"[bold]Nice to meet you, {user_name}! Now give me a name[/bold]", default=default_agent)
        console.print()

        # Personality
        personalities = list_personalities()

        custom_system_prompt_default = ""
        try:
            existing_identity = SoulManager(identity_path).load()
            if existing_identity.get("personality_id") == "custom":
                custom_system_prompt_default = existing_identity.get(
                    "custom_system_prompt", ""
                )
        except Exception:
            custom_system_prompt_default = ""

        if interactive:
            personality_choices = [
                {"value": p.id, "name": f"{p.emoji}  {p.name} — {p.description}"}
                for p in personalities
            ]
            personality_id = _q_select(
                f"Alright, I'm {agent_name}. What kind of AI am I?",
                personality_choices,
                default=default_personality,
            )
            personality = get_personality(personality_id)
        else:
            from rich.table import Table
            console.print(f"[bold]Alright, I'm {agent_name}. What kind of AI am I?[/bold]")
            console.print()
            table = Table(show_header=False, box=None, padding=(0, 2))
            for i, p in enumerate(personalities, 1):
                table.add_row(f"[bold cyan]{i}[/bold cyan]", f"{p.emoji} [bold]{p.name}[/bold]")
                table.add_row("", f"[dim]{p.description}[/dim]")
                table.add_row("", "")
            console.print(table)
            fallback_choice = "1"
            for i, p in enumerate(personalities, 1):
                if p.id == default_personality:
                    fallback_choice = str(i)
                    break
            choice = Prompt.ask("[bold]Pick a number[/bold]",
                                choices=[str(i) for i in range(1, len(personalities) + 1)],
                                default=fallback_choice)
            personality = personalities[int(choice) - 1]

        console.print(f"  [green]✓[/green] {personality.emoji} {personality.name} it is!")
        console.print()

        custom_system_prompt = ""
        if personality.id == "custom":
            custom_system_prompt = _q_multiline_text(
                console,
                "Enter the full custom system prompt",
                default=custom_system_prompt_default,
            )
            if not custom_system_prompt.strip():
                custom_system_prompt = get_personality("helpful").system_prompt
            console.print("  [green]✓[/green] Custom system prompt saved.")
            console.print()
    else:
        # Carry forward existing identity
        user_name = e.get("user_name", "User")
        agent_name = e.get("agent_name", "Arc")
        personality = get_personality(e.get("personality", "helpful"))
        custom_system_prompt = ""

    # ── LLM Provider + Worker ─────────────────────────────────
    if "models" in sections:
        provider_defaults = {
            "provider": e.get("provider", "ollama"),
            "model": e.get("model", ""),
            "base_url": e.get("base_url", ""),
            "api_key": e.get("api_key", ""),
        }
        provider_cfg = _pick_provider(console, interactive=interactive, defaults=provider_defaults)

        # Worker model (optional)
        worker_defaults: dict | None = None
        if e.get("worker_provider"):
            worker_defaults = {
                "provider": e["worker_provider"],
                "model": e.get("worker_model", ""),
                "base_url": e.get("worker_base_url", ""),
                "api_key": e.get("worker_api_key", ""),
            }
        worker_cfg = _pick_worker_model(
            console, interactive=interactive, defaults=worker_defaults,
        )

        console.print(
            f"  [bold]Provider:[/bold] {provider_cfg['provider']}  "
            f"[bold]Model:[/bold] {provider_cfg['model']}"
        )
        if worker_cfg:
            console.print(
                f"  [bold]Worker:[/bold]   {worker_cfg['provider']}/{worker_cfg['model']}"
            )
        console.print()
    else:
        # Carry forward existing model config
        provider_cfg = {
            "provider": e.get("provider", "ollama"),
            "model": e.get("model", ""),
            "base_url": e.get("base_url", ""),
            "api_key": e.get("api_key", ""),
        }
        worker_cfg = None
        if e.get("worker_provider"):
            worker_cfg = {
                "provider": e["worker_provider"],
                "model": e.get("worker_model", ""),
                "base_url": e.get("worker_base_url", ""),
                "api_key": e.get("worker_api_key", ""),
            }

    # ── Tavily (Liquid Web) ──────────────────────────────────
    if "tavily" in sections:
        tavily_cfg = _pick_tavily(
            console, interactive=interactive,
            existing_key=e.get("tavily_api_key", ""),
            existing_ngrok=e.get("ngrok_auth_token", ""),
        )
    else:
        tavily_cfg = {
            "api_key": e["tavily_api_key"],
            "ngrok_token": e.get("ngrok_auth_token", ""),
        } if e.get("tavily_api_key") else None

    # ── Telegram Bot ─────────────────────────────────────────
    if "telegram" in sections:
        telegram_cfg = _pick_telegram(
            console, interactive=interactive,
            existing_token=e.get("telegram_token", ""),
            existing_chat_id=e.get("telegram_chat_id", ""),
            existing_allowed=",".join(e.get("telegram_allowed_users", [])),
        )
    else:
        if e.get("telegram_token"):
            telegram_cfg = {
                "token": e["telegram_token"],
                "chat_id": e.get("telegram_chat_id", ""),
                "allowed_users": e.get("telegram_allowed_users", []),
            }
        else:
            telegram_cfg = None

    # ── Create identity file (only if identity was changed) ──
    if "identity" in sections:
        soul = SoulManager(identity_path)
        soul.create(
            agent_name,
            user_name,
            personality.id,
            custom_system_prompt=custom_system_prompt,
        )

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

    if tavily_cfg:
        config_lines += [
            "[tavily]",
            f'api_key = "{tavily_cfg["api_key"]}"',
            "",
        ]
        if tavily_cfg.get("ngrok_token"):
            config_lines += [
                "[ngrok]",
                f'auth_token = "{tavily_cfg["ngrok_token"]}"',
                "",
            ]

    if telegram_cfg:
        config_lines += [
            "[telegram]",
            f'token = "{telegram_cfg["token"]}"',
        ]
        if telegram_cfg.get("chat_id"):
            config_lines.append(f'chat_id = "{telegram_cfg["chat_id"]}"')
        if telegram_cfg.get("allowed_users"):
            users_str = ", ".join(f'"{u}"' for u in telegram_cfg["allowed_users"])
            config_lines.append(f"allowed_users = [{users_str}]")
        config_lines.append("")

    config_content = "\n".join(config_lines)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content, encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────
    model_info = f"{provider_cfg['provider']}/{provider_cfg['model']}"
    if worker_cfg:
        model_info += f"\n[green]✓[/green] Worker: {worker_cfg['provider']}/{worker_cfg['model']}"
    if tavily_cfg:
        model_info += "\n[green]✓[/green] Liquid Web: enabled (Tavily)"
    if telegram_cfg:
        model_info += "\n[green]✓[/green] Telegram bot: enabled"

    console.print(
        Panel(
            f"[green]✓[/green] Identity saved to [cyan]{identity_path}[/cyan]\n"
            f"[green]✓[/green] Config saved to [cyan]{config_path}[/cyan]\n"
            f"[green]✓[/green] Model: {model_info}\n"
            "\n"
            "[bold]You're all set! Here's what you can do:[/bold]\n"
            "\n"
            "  [cyan]arc chat[/cyan]          — Talk to me\n"
            "  [cyan]arc telegram[/cyan]      — Run as Telegram bot\n"
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
    if tavily_cfg:
        result["tavily_api_key"] = tavily_cfg["api_key"]
        if tavily_cfg.get("ngrok_token"):
            result["ngrok_auth_token"] = tavily_cfg["ngrok_token"]
    if telegram_cfg:
        result["telegram_token"] = telegram_cfg["token"]
        result["telegram_chat_id"] = telegram_cfg.get("chat_id", "")
        result["telegram_allowed_users"] = telegram_cfg.get("allowed_users", [])
    return result
