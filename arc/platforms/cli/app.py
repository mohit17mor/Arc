"""
CLI Platform — interactive terminal chat.

Simplified approach: print status as it happens, no fancy animations.
More reliable across different terminals.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

from rich.console import Console
from rich.panel import Panel
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
        
        # Track what's been printed so we don't duplicate
        self._printed_thinking = False
        self._tool_call_count = 0

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
        """Handle events from the agent for display — print immediately."""
        if event.type == EventType.AGENT_THINKING:
            iteration = event.data.get("iteration", 1)
            if iteration == 1 and not self._printed_thinking:
                self._console.print("[dim]Thinking...[/dim]")
                self._printed_thinking = True
            elif iteration > 1:
                self._console.print("[dim]Analyzing...[/dim]")
                
        elif event.type == EventType.SKILL_TOOL_CALL:
            tool_name = event.data.get("tool", "unknown")
            arguments = event.data.get("arguments", {})
            
            # Format arguments
            args_parts = []
            for k, v in list(arguments.items())[:2]:
                if isinstance(v, str):
                    v_short = v[:25] + "..." if len(v) > 25 else v
                    args_parts.append(f'{k}="{v_short}"')
                else:
                    args_parts.append(f"{k}={v}")
            args_str = ", ".join(args_parts)
            
            self._console.print(f"[yellow]⟳[/yellow] [bold]{tool_name}[/bold]({args_str})")
            self._tool_call_count += 1
            
        elif event.type == EventType.SKILL_TOOL_RESULT:
            success = event.data.get("success", False)
            preview = event.data.get("output_preview", "")
            
            if success:
                icon = "[green]✓[/green]"
            else:
                icon = "[red]✗[/red]"
            
            if preview:
                # Show first line of preview, cleaned up
                preview_line = preview.replace("\n", " ").strip()[:60]
                if len(preview) > 60:
                    preview_line += "..."
                self._console.print(f"  {icon} [dim]{preview_line}[/dim]")
            else:
                self._console.print(f"  {icon} [dim]Done[/dim]")

    def _reset_state(self) -> None:
        """Reset state for new message."""
        self._printed_thinking = False
        self._tool_call_count = 0

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
        self._reset_state()

        response_started = False
        response_text = ""

        try:
            async for chunk in handler(user_input):
                # First text chunk — print agent name header
                if chunk.strip() and not response_started:
                    response_started = True
                    self._console.print()  # Blank line after tool calls
                    self._console.print(f"[bold cyan]{self._agent_name}[/bold cyan]")
                
                if response_started:
                    # Print chunk immediately (streaming)
                    self._console.print(chunk, end="", highlight=False)
                    response_text += chunk

            # Ensure newline at end
            if response_started:
                self._console.print()
            elif self._tool_call_count > 0:
                # Tools were called but no text response
                self._console.print()
                self._console.print(f"[bold cyan]{self._agent_name}[/bold cyan]")
                self._console.print("[dim]Done.[/dim]")

        except Exception as e:
            self._console.print(f"\n[red]Error: {e}[/red]")