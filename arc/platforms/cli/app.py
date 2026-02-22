"""
CLI Platform — interactive terminal chat with rich feedback.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pathlib import Path

from arc.platforms.base import Platform, MessageHandler
from arc.core.events import Event, EventType


class CLIPlatform(Platform):
    """
    Rich terminal interface for Arc.

    Features:
    - Streaming responses
    - Thinking spinner
    - Tool call display
    - Command history
    - Special commands (/help, /cost, /clear, /exit)
    """

    def __init__(
        self,
        console: Console | None = None,
        agent_name: str = "Arc",
        user_name: str = "You",
    ) -> None:
        self._console = console or Console()
        self._agent_name = agent_name
        self._user_name = user_name
        self._running = False
        self._cost_tracker: dict = {}
        
        # For displaying tool calls
        self._current_status: str = ""
        self._tool_calls: list[dict] = []

        # Command history
        history_path = Path.home() / ".arc" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self._session = PromptSession(history=FileHistory(str(history_path)))

    @property
    def name(self) -> str:
        return "cli"

    def set_cost_tracker(self, tracker: dict) -> None:
        """Set reference to cost tracker for /cost command."""
        self._cost_tracker = tracker

    def on_event(self, event: Event) -> None:
        """Handle events from the agent for display."""
        if event.type == EventType.AGENT_THINKING:
            iteration = event.data.get("iteration", 1)
            if iteration == 1:
                self._current_status = "thinking"
            else:
                self._current_status = "analyzing"
                
        elif event.type == EventType.SKILL_TOOL_CALL:
            tool_name = event.data.get("tool", "unknown")
            arguments = event.data.get("arguments", {})
            self._tool_calls.append({
                "name": tool_name,
                "arguments": arguments,
                "status": "running",
            })
            
        elif event.type == EventType.SKILL_TOOL_RESULT:
            if self._tool_calls:
                success = event.data.get("success", False)
                self._tool_calls[-1]["status"] = "success" if success else "error"
                self._tool_calls[-1]["preview"] = event.data.get("output_preview", "")

    def _reset_status(self) -> None:
        """Reset status for new message."""
        self._current_status = ""
        self._tool_calls = []

    async def run(self, handler: MessageHandler) -> None:
        """Run the interactive CLI loop."""
        self._running = True

        # Welcome message
        self._console.print()
        self._console.print(
            Panel(
                f"[bold]{self._agent_name}[/bold] is ready.\n"
                f"[dim]Type [bold]/help[/bold] for commands. "
                f"[bold]/exit[/bold] to quit.[/dim]",
                border_style="cyan",
            )
        )
        self._console.print()

        while self._running:
            try:
                # Get user input
                user_input = await self._get_input()

                if user_input is None:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    should_continue = await self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Process with agent
                await self._process_message(user_input, handler)

            except KeyboardInterrupt:
                self._console.print("\n[dim]Use /exit to quit[/dim]")
            except EOFError:
                break

        self._console.print("\n[dim]Goodbye![/dim]")

    async def stop(self) -> None:
        """Stop the CLI."""
        self._running = False

    async def _get_input(self) -> str | None:
        """Get input from user."""
        try:
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(
                None,
                lambda: self._session.prompt(
                    f"\n{self._user_name} > ",
                ),
            )
            return user_input.strip() if user_input else None
        except (KeyboardInterrupt, EOFError):
            return None

    async def _handle_command(self, command: str) -> bool:
        """Handle a / command. Returns False if should exit."""
        cmd = command.lower().strip()

        if cmd in ("/exit", "/quit", "/q"):
            return False

        elif cmd in ("/help", "/h", "/?"):
            self._console.print(
                Panel(
                    "[bold]Commands[/bold]\n\n"
                    "  [cyan]/help[/cyan]   — Show this help\n"
                    "  [cyan]/cost[/cyan]   — Show token usage and cost\n"
                    "  [cyan]/clear[/cyan]  — Clear conversation history\n"
                    "  [cyan]/exit[/cyan]   — Exit the chat",
                    border_style="blue",
                )
            )

        elif cmd == "/cost":
            if self._cost_tracker:
                summary = self._cost_tracker
                self._console.print(
                    Panel(
                        f"[bold]Session Cost[/bold]\n\n"
                        f"  Requests:      {summary.get('requests', 0)}\n"
                        f"  Input tokens:  {summary.get('input_tokens', 0):,}\n"
                        f"  Output tokens: {summary.get('output_tokens', 0):,}\n"
                        f"  Total tokens:  {summary.get('total_tokens', 0):,}\n"
                        f"  Cost:          ${summary.get('cost_usd', 0):.4f}",
                        border_style="yellow",
                    )
                )
            else:
                self._console.print("[dim]Cost tracking not available[/dim]")

        elif cmd == "/clear":
            self._console.print("[dim]Conversation cleared[/dim]")

        else:
            self._console.print(f"[dim]Unknown command: {command}[/dim]")

        return True

    async def _process_message(
        self,
        user_input: str,
        handler: MessageHandler,
    ) -> None:
        """Process a user message and stream the response."""
        self._console.print()
        self._reset_status()

        response_text = ""
        first_chunk = True
        spinner_shown = False

        try:
            # Show thinking spinner initially
            with Live(
                self._build_status_display(thinking=True),
                console=self._console,
                refresh_per_second=10,
                transient=True,
            ) as live:
                spinner_shown = True
                
                async for chunk in handler(user_input):
                    # First chunk of actual response - stop spinner
                    if first_chunk and chunk.strip():
                        first_chunk = False
                        # Update display one final time with all tool calls
                        live.update(self._build_status_display(thinking=False))
                        
                    response_text += chunk

                    # Update live display with current status
                    if self._current_status or self._tool_calls:
                        live.update(self._build_status_display(
                            thinking=True,
                            partial_response=response_text if not first_chunk else ""
                        ))

            # Print final response with agent name
            if response_text.strip():
                self._print_response(response_text)

        except Exception as e:
            self._console.print(f"\n[red]Error: {e}[/red]")

    def _build_status_display(
        self,
        thinking: bool = False,
        partial_response: str = "",
    ) -> Group:
        """Build the status display with spinner and tool calls."""
        elements = []

        # Thinking spinner
        if thinking and not partial_response:
            status_text = {
                "thinking": "Thinking",
                "analyzing": "Analyzing results",
                "": "Thinking",
            }.get(self._current_status, "Thinking")
            
            spinner = Spinner("dots", text=f"[cyan]{status_text}...[/cyan]")
            elements.append(spinner)

        # Tool calls
        for tool in self._tool_calls:
            tool_display = self._format_tool_call(tool)
            elements.append(tool_display)

        if not elements:
            # Default spinner if nothing else
            elements.append(Spinner("dots", text="[cyan]Thinking...[/cyan]"))

        return Group(*elements)

    def _format_tool_call(self, tool: dict) -> Text:
        """Format a tool call for display."""
        name = tool["name"]
        args = tool["arguments"]
        status = tool["status"]

        # Status icon
        if status == "running":
            icon = "🔧"
            color = "yellow"
        elif status == "success":
            icon = "✓"
            color = "green"
        else:
            icon = "✗"
            color = "red"

        # Format arguments (simplified)
        args_str = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
            for k, v in list(args.items())[:3]  # Show max 3 args
        )
        if len(args) > 3:
            args_str += ", ..."

        text = Text()
        text.append(f"\n{icon} ", style=color)
        text.append(f"{name}", style=f"bold {color}")
        text.append(f"({args_str})", style="dim")
        
        # Add preview for completed tools
        if status == "success" and tool.get("preview"):
            preview = tool["preview"][:100]
            if len(tool.get("preview", "")) > 100:
                preview += "..."
            text.append(f"\n   └─ ", style="dim")
            text.append(preview, style="dim italic")

        return text

    def _print_response(self, response: str) -> None:
        """Print the final response with formatting."""
        self._console.print()
        self._console.print(f"[bold cyan]{self._agent_name}[/bold cyan]")
        self._console.print(response.strip())