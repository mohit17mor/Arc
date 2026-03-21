"""Tests for CLI commands."""

import asyncio
import pytest
from pathlib import Path
from typer.testing import CliRunner
from arc.cli.main import app


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    """arc version shows version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_init_creates_files(runner, tmp_path, monkeypatch):
    """arc init creates config and identity files."""
    # Redirect home directory
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Mock input — setup flow (non-interactive fallback):
    # 1. User name, 2. Agent name, 3. Personality (3),
    # 4. Provider (1=Ollama), 5. Base URL, 6. Model,
    # 7. Worker model? (n), 8. Tavily? (n), 9. Telegram? (n)
    result = runner.invoke(
        app,
        ["init"],
        input="Alex\nFriday\n3\n1\nhttp://localhost:11434\nllama3.1\nn\nn\nn\n",
    )

    assert result.exit_code == 0
    assert (tmp_path / ".arc" / "config.toml").exists()
    assert (tmp_path / ".arc" / "identity.md").exists()


def test_init_reconfigure_keeps_defaults(runner, tmp_path, monkeypatch):
    """arc init on existing config pre-populates defaults; Enter keeps them."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # First run — create initial config
    result1 = runner.invoke(
        app,
        ["init"],
        input="Alex\nFriday\n3\n1\nhttp://localhost:11434\nllama3.1\nn\nn\nn\n",
    )
    assert result1.exit_code == 0

    # Second run — reconfigure
    # Non-interactive path: section_menu (default=Everything),
    # user_name, agent_name, personality, provider,
    # base_url, model, worker_confirm, tavily_confirm, telegram_confirm
    result2 = runner.invoke(
        app,
        ["init"],
        input="\n\n\n\n\n\n\n\n\n\n",
    )
    assert result2.exit_code == 0
    assert "Current Configuration" in result2.stdout

    # Verify config preserved
    cfg_text = (tmp_path / ".arc" / "config.toml").read_text()
    assert 'default_model = "llama3.1"' in cfg_text
    assert 'user_name = "Alex"' in cfg_text


def test_init_reconfigure_single_section(runner, tmp_path, monkeypatch):
    """arc init with section selection only prompts for that section."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # First run — create initial config
    result1 = runner.invoke(
        app,
        ["init"],
        input="Alex\nFriday\n3\n1\nhttp://localhost:11434\nllama3.1\nn\nn\nn\n",
    )
    assert result1.exit_code == 0

    # Second run — pick section 3 (Liquid Web / Tavily) only,
    # then exit (6) from the menu loop
    # Non-interactive: section_menu=3, tavily prompt (n), section_menu=6 (exit)
    result2 = runner.invoke(
        app,
        ["init"],
        input="3\nn\n6\n",
    )
    assert result2.exit_code == 0

    # Verify identity and model config are preserved (not re-prompted)
    cfg_text = (tmp_path / ".arc" / "config.toml").read_text()
    assert 'default_model = "llama3.1"' in cfg_text
    assert 'user_name = "Alex"' in cfg_text
    assert 'agent_name = "Friday"' in cfg_text


def test_chat_without_init(runner, tmp_path, monkeypatch):
    """arc chat fails gracefully without init."""
    # Redirect to empty home
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 1
    assert "not configured" in result.stdout.lower() or "arc init" in result.stdout.lower()


def test_init_custom_personality_writes_custom_prompt(runner, tmp_path, monkeypatch):
    """arc init supports custom full system prompt for the main identity."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = runner.invoke(
        app,
        ["init"],
        input=(
            "Alex\n"
            "Friday\n"
            "6\n"
            "You are a strict coding assistant.\n"
            "Be direct and concise.\n"
            "END\n"
            "1\n"
            "http://localhost:11434\n"
            "llama3.1\n"
            "n\n"
            "n\n"
            "n\n"
        ),
    )

    assert result.exit_code == 0
    cfg_text = (tmp_path / ".arc" / "config.toml").read_text(encoding="utf-8")
    identity_text = (tmp_path / ".arc" / "identity.md").read_text(encoding="utf-8")

    assert 'personality = "custom"' in cfg_text
    assert "You are a strict coding assistant." in identity_text
    assert "Be direct and concise." in identity_text


@pytest.mark.asyncio
async def test_chat_wires_plan_updates_to_cli(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from arc.cli import main as cli_main
    from arc.core.events import EventType

    config_path = tmp_path / 'config.toml'
    config_path.write_text('ok')
    monkeypatch.setattr(cli_main, 'get_config_path', lambda: config_path)

    subscribed: list[str] = []

    class FakeKernel:
        def on(self, event_type, handler):
            subscribed.append(event_type)

    class FakeCLI:
        def __init__(self, *args, **kwargs):
            pass

        def set_approval_flow(self, flow):
            pass

        def set_escalation_bus(self, bus):
            pass

        def set_skill_manager(self, skill_manager):
            pass

        def set_skill_router(self, router):
            pass

        def set_mcp_manager(self, mcp_manager):
            pass

        def set_turn_controller(self, controller):
            pass

        def set_memory_manager(self, manager):
            pass

        def set_workflow_skill(self, skill):
            pass

        def set_pending_queue(self, queue):
            pass

        def set_scheduler_store(self, store):
            pass

        async def run(self, handler):
            return None

    class FakeRouter:
        def register(self, channel):
            pass

    class FakeChannel:
        def __init__(self, queue):
            self.queue = queue

        def set_active(self, active):
            pass

    class FakeWorkflowSkill:
        pass

    class FakeMCPManager:
        has_servers = False

    async def fake_bootstrap(**kwargs):
        return SimpleNamespace(
            identity={'agent_name': 'Arc', 'user_name': 'You'},
            agent=SimpleNamespace(security=SimpleNamespace(approval_flow=object())),
            escalation_bus=object(),
            skill_manager=SimpleNamespace(get_skill=lambda name: None),
            skill_router=object(),
            mcp_manager=FakeMCPManager(),
            turn_controller=object(),
            memory_manager=None,
            notification_router=FakeRouter(),
            config=SimpleNamespace(scheduler=SimpleNamespace(enabled=False)),
            sched_store=None,
            kernel=FakeKernel(),
            cost_tracker=SimpleNamespace(start_turn=lambda: None, summary=lambda: {}),
            start=lambda: asyncio.sleep(0),
            shutdown=lambda: asyncio.sleep(0),
        )

    monkeypatch.setattr('arc.cli.bootstrap.bootstrap', fake_bootstrap)
    monkeypatch.setattr('arc.platforms.cli.app.CLIPlatform', FakeCLI)
    monkeypatch.setattr('arc.notifications.channels.cli.CLIChannel', FakeChannel)
    monkeypatch.setattr('arc.workflow.skill.WorkflowSkill', FakeWorkflowSkill)

    await cli_main._run_chat(None, False)

    assert EventType.AGENT_PLAN_UPDATE in subscribed

