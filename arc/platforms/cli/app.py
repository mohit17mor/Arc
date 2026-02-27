"""
CLI Platform — interactive terminal chat.

Handles:
- Streaming responses
- Tool call display  
- Security approval prompts
- Command history
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Any

from rich.console import Console
from rich.panel import Panel
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
    - Security approval prompts
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
        
        # Security approval handling
        self._approval_flow: Any = None  # Set via set_approval_flow()
        self._pending_approval: asyncio.Event = asyncio.Event()
        self._skill_manager: Any = None  # Set via set_skill_manager()
        self._memory_manager: Any = None  # Set via set_memory_manager()
        self._scheduler_store: Any = None  # Set via set_scheduler_store()
        self._pending_queue: asyncio.Queue | None = None  # Set via set_pending_queue()
        self._turn_in_progress: bool = False  # True while agent is generating

        # Track state for display
        self._printed_thinking = False
        self._tool_call_count = 0
        self._waiting_for_approval = False
        
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
    
    def set_approval_flow(self, flow: Any) -> None:
        """Set reference to approval flow for security prompts."""
        self._approval_flow = flow

    def set_skill_manager(self, skill_manager: Any) -> None:
        """Set reference to skill manager for /skills command."""
        self._skill_manager = skill_manager

    def set_memory_manager(self, memory_manager: Any) -> None:
        """Set reference to memory manager for /memory command."""
        self._memory_manager = memory_manager

    def set_scheduler_store(self, scheduler_store: Any) -> None:
        """Set reference to scheduler store for /jobs command."""
        self._scheduler_store = scheduler_store

    def set_pending_queue(self, queue: asyncio.Queue) -> None:
        """Set the queue that receives completed background-job results."""
        self._pending_queue = queue
    
    def on_event(self, event: Event) -> None:
        """Handle events from the agent for display."""
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
            
            # Format arguments preview
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
            
            icon = "[green]✓[/green]" if success else "[red]✗[/red]"
            
            if preview:
                preview_line = preview.replace("\n", " ").strip()[:60]
                if len(preview) > 60:
                    preview_line += "..."
                self._console.print(f"  {icon} [dim]{preview_line}[/dim]")
            else:
                self._console.print(f"  {icon} [dim]Done[/dim]")
        
        elif event.type == EventType.SECURITY_APPROVAL:
            # Handle approval request asynchronously
            asyncio.create_task(self._handle_approval_request(event))
    
    async def _handle_approval_request(self, event: Event) -> None:
        """Show approval prompt and resolve the request."""
        request_id = event.data.get("request_id", "")
        tool_name = event.data.get("tool_name", "unknown")
        tool_description = event.data.get("tool_description", "")
        arguments = event.data.get("arguments", {})
        capabilities = event.data.get("capabilities", [])
        
        self._waiting_for_approval = True
        
        # Format arguments for display
        args_display = []
        for k, v in arguments.items():
            if isinstance(v, str) and len(v) > 50:
                v = v[:50] + "..."
            args_display.append(f"  [cyan]{k}[/cyan]: {v}")
        args_str = "\n".join(args_display) if args_display else "  (no arguments)"
        
        # Show approval prompt
        self._console.print()
        self._console.print(
            Panel(
                f"[bold yellow]⚠ Permission Required[/bold yellow]\n\n"
                f"[bold]{tool_name}[/bold]: {tool_description}\n\n"
                f"[bold]Arguments:[/bold]\n{args_str}\n\n"
                f"[bold]Capabilities:[/bold] {', '.join(capabilities)}\n\n"
                f"[dim]Options:[/dim]\n"
                f"  [green]y[/green] = Allow once\n"
                f"  [green]a[/green] = Allow always (remember)\n"
                f"  [red]n[/red] = Deny\n"
                f"  [red]d[/red] = Deny always (remember)",
                border_style="yellow",
            )
        )
        
        # Get user response.
        # NOTE: We use plain input() in a thread executor rather than
        # Rich's Prompt.ask() because on Windows, Rich tries to create
        # its own console reader which deadlocks against prompt_toolkit's
        # terminal ownership.
        self._console.print("[bold]Allow? ([green]y[/green]=once  [green]a[/green]=always  [red]n[/red]=deny  [red]d[/red]=deny always)[/bold] ", end="")
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, input)
            response = raw.strip().lower() or "n"
            if response not in ("y", "a", "n", "d"):
                response = "n"
        except (KeyboardInterrupt, EOFError):
            response = "n"
        
        # Map response to decision
        response_map = {
            "y": "allow_once",
            "a": "allow_always",
            "n": "deny",
            "d": "deny_always",
        }
        decision = response_map.get(response, "deny")
        
        # Show confirmation
        if decision.startswith("allow"):
            self._console.print(f"[green]✓[/green] Allowed")
        else:
            self._console.print(f"[red]✗[/red] Denied")
        
        self._console.print()
        self._waiting_for_approval = False
        
        # Resolve the approval request
        if self._approval_flow:
            self._approval_flow.resolve_approval(request_id, decision)
    
    def _reset_state(self) -> None:
        """Reset display state for new message."""
        self._printed_thinking = False
        self._tool_call_count = 0

    def _inject_pending_results(self, user_input: str) -> str:
        """
        Drain the pending-job queue and, if any results are waiting,
        prepend them to the user's message so the main agent sees them
        as context and can weave them into its response naturally.

        A brief terminal notice is shown so the user knows something
        is being injected before they see the agent's reply.
        """
        if self._pending_queue is None or self._pending_queue.empty():
            return user_input

        from arc.notifications.base import Notification
        results: list[Notification] = []
        while not self._pending_queue.empty():
            try:
                results.append(self._pending_queue.get_nowait())
            except Exception:
                break

        if not results:
            return user_input

        self._console.print(
            f"[dim]⏰ {len(results)} background task(s) completed — "
            f"injecting into context...[/dim]"
        )

        parts = [
            "The following background task(s) completed since your last message. "
            "Briefly mention the key findings before responding to the user.\n"
        ]
        for r in results:
            import datetime
            ts = datetime.datetime.fromtimestamp(r.fired_at).strftime("%H:%M")
            parts.append(
                f"[Background task: \"{r.job_name}\" completed at {ts}]\n"
                f"{r.content}\n"
            )
        parts.append(f"\n---\nUser message: {user_input}")
        return "\n".join(parts)
    
    async def _watcher_loop(self) -> None:
        """
        Background task: deliver queued job results immediately when the
        user is idle (no turn in progress).  If a turn IS in progress,
        leave items in the queue so _inject_pending_results picks them up
        and the main agent can weave them into its reply.
        """
        import datetime
        from arc.notifications.base import Notification

        while self._running:
            await asyncio.sleep(1.0)
            if (
                self._turn_in_progress
                or self._pending_queue is None
                or self._pending_queue.empty()
            ):
                continue

            # Drain and display while still idle
            while not self._pending_queue.empty() and not self._turn_in_progress:
                try:
                    notif: Notification = self._pending_queue.get_nowait()
                except Exception:
                    break
                ts = datetime.datetime.fromtimestamp(notif.fired_at).strftime("%H:%M")
                self._console.print()
                self._console.print(
                    Panel(
                        f"[bold cyan]⏰ {notif.job_name}[/bold cyan]  "
                        f"[dim]{ts}[/dim]\n\n"
                        f"{notif.content}",
                        border_style="cyan",
                        subtitle="[dim]background task[/dim]",
                    )
                )
                self._console.print()

    async def run(self, handler: MessageHandler) -> None:
        """Run the interactive CLI loop."""
        self._running = True

        # Start the idle-notification watcher
        watcher_task = asyncio.create_task(self._watcher_loop(), name="notification-watcher")

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
        
        try:
            while self._running:
                try:
                    # Skip input if waiting for approval
                    if self._waiting_for_approval:
                        await asyncio.sleep(0.1)
                        continue
                    
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

                    # Inject anything that arrived while this turn was starting
                    # (watcher pauses during a turn; this catches the race window)
                    user_input = self._inject_pending_results(user_input)

                    # Process with agent — hold watcher off during generation
                    self._turn_in_progress = True
                    try:
                        await self._process_message(user_input, handler)
                    finally:
                        self._turn_in_progress = False
                
                except KeyboardInterrupt:
                    self._console.print("\n[dim]Use /exit to quit[/dim]")
                except EOFError:
                    break
        finally:
            watcher_task.cancel()
            try:
                await watcher_task
            except asyncio.CancelledError:
                pass

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
                lambda: self._session.prompt(f"\n{self._user_name} > "),
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
                    "  [cyan]/help[/cyan]     \u2014 Show this help\n"
                    "  [cyan]/skills[/cyan]   \u2014 List available skills and tools\n"
                    "  [cyan]/memory[/cyan]   \u2014 Show long-term memory (core facts)\n"
                    "  [cyan]/memory episodic[/cyan] \u2014 Show recent episodic memories\n"
                    "  [cyan]/memory forget <id>[/cyan] \u2014 Delete a core memory by id\n"
                    "  [cyan]/jobs[/cyan]     \u2014 List scheduled jobs\n"
                    "  [cyan]/jobs cancel <name>[/cyan] \u2014 Cancel a scheduled job\n"
                    "  [cyan]/cost[/cyan]     \u2014 Show token usage and cost\n"
                    "  [cyan]/perms[/cyan]    \u2014 Show remembered permissions\n"
                    "  [cyan]/clear[/cyan]    \u2014 Clear conversation history\n"
                    "  [cyan]/exit[/cyan]     \u2014 Exit the chat",
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
        
        elif cmd in ("/skills", "/skill"):
            if not self._skill_manager:
                self._console.print("[dim]No skill manager available[/dim]")
            else:
                lines: list[str] = ["[bold]Available Skills[/bold]\n"]
                for skill_name in sorted(self._skill_manager.skill_names):
                    skill = self._skill_manager.get_skill(skill_name)
                    if skill is None:
                        continue
                    manifest = skill.manifest()
                    lines.append(f"  [bold cyan]{manifest.name}[/bold cyan] v{manifest.version}")
                    lines.append(f"  [dim]{manifest.description}[/dim]")
                    for tool_spec in manifest.tools:
                        lines.append(f"    [yellow]⟳[/yellow] [bold]{tool_spec.name}[/bold]")
                        # Wrap description at ~60 chars for readability
                        desc = tool_spec.description.split(". ")[0]  # first sentence only
                        if len(desc) > 70:
                            desc = desc[:67] + "..."
                        lines.append(f"      [dim]{desc}[/dim]")
                    lines.append("")
                self._console.print(
                    Panel("\n".join(lines).rstrip(), border_style="blue")
                )

        elif cmd == "/perms":
            self._console.print("[dim]Permission memory: use /clear to reset[/dim]")
        
        elif cmd == "/clear":
            self._console.print("[dim]Conversation cleared[/dim]")

        elif cmd.startswith("/memory"):
            await self._handle_memory_command(command.strip())

        elif cmd.startswith("/jobs"):
            await self._handle_jobs_command(command.strip())

        else:
            self._console.print(f"[dim]Unknown command: {command}[/dim]")
        
        return True
    
    async def _handle_jobs_command(self, command: str) -> None:
        """Handle /jobs subcommands."""
        store = self._scheduler_store
        if store is None:
            self._console.print("[dim]Scheduler is not available[/dim]")
            return

        parts = command.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "cancel":
            if len(parts) < 3:
                self._console.print("[dim]Usage: /jobs cancel <name_or_id>[/dim]")
                return
            name_or_id = parts[2].strip()
            try:
                job = await store.get_by_name(name_or_id)
                if job is None:
                    all_jobs = await store.get_all()
                    job = next((j for j in all_jobs if j.id == name_or_id), None)
                if job is None:
                    self._console.print(f"[dim]No job found: {name_or_id}[/dim]")
                    return
                await store.delete(job.id)
                self._console.print(f"[green]✓[/green] Cancelled job: [bold]{job.name}[/bold]")
            except Exception as e:
                self._console.print(f"[red]Scheduler error: {e}[/red]")

        else:
            # Default: list all jobs
            try:
                import datetime
                from arc.scheduler.triggers import make_trigger
                jobs = await store.get_all()
                if not jobs:
                    self._console.print("[dim]No scheduled jobs[/dim]")
                    return
                lines = ["[bold]Scheduled Jobs[/bold]\n"]
                for job in jobs:
                    status_colour = "green" if job.active else "dim"
                    status = "active" if job.active else "inactive"
                    trigger = make_trigger(job.trigger)
                    next_dt = (
                        datetime.datetime.fromtimestamp(job.next_run).strftime("%Y-%m-%d %H:%M")
                        if job.next_run > 0 else "—"
                    )
                    lines.append(
                        f"  [bold cyan]{job.name}[/bold cyan] "
                        f"[{status_colour}]({status})[/{status_colour}] "
                        f"[dim]id={job.id}[/dim]\n"
                        f"  trigger: {trigger.description}   next: {next_dt}\n"
                        f"  [dim]{job.prompt[:80]}{'...' if len(job.prompt) > 80 else ''}[/dim]"
                    )
                self._console.print(Panel("\n\n".join(lines).rstrip(), border_style="cyan"))
            except Exception as e:
                self._console.print(f"[red]Scheduler error: {e}[/red]")

    async def _handle_memory_command(self, command: str) -> None:
        """Handle /memory subcommands."""
        mm = self._memory_manager
        if mm is None:
            self._console.print("[dim]Long-term memory is not available[/dim]")
            return

        parts = command.split(maxsplit=2)
        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "episodic":
            try:
                items = await mm.list_episodic(limit=10)
                if not items:
                    self._console.print("[dim]No episodic memories yet[/dim]")
                    return
                lines = ["[bold]Recent Episodic Memories[/bold]\n"]
                for item in items:
                    import datetime
                    ts = datetime.datetime.fromtimestamp(item.created_at).strftime("%Y-%m-%d %H:%M")
                    content = item.content[:120]
                    lines.append(
                        f"  [dim]id={item.id}  importance={item.importance:.2f}  {ts}[/dim]\n"
                        f"  {content}\n"
                    )
                self._console.print(Panel("\n".join(lines).rstrip(), border_style="magenta"))
            except Exception as e:
                self._console.print(f"[red]Memory error: {e}[/red]")

        elif sub == "forget":
            if len(parts) < 3:
                self._console.print("[dim]Usage: /memory forget <id>[/dim]")
                return
            fact_id = parts[2].strip()
            try:
                await mm.delete_core(fact_id)
                self._console.print(f"[green]✓[/green] Deleted core memory: [bold]{fact_id}[/bold]")
            except Exception as e:
                self._console.print(f"[red]Memory error: {e}[/red]")

        else:
            # Default: show core facts
            try:
                core_facts = await mm.get_all_core()
                if not core_facts:
                    self._console.print("[dim]No core memories yet[/dim]")
                    return
                lines = ["[bold]Core Memories[/bold]\n"]
                for fact in core_facts:
                    conf = getattr(fact, "confidence", 1.0)
                    lines.append(
                        f"  [cyan]{fact.id}[/cyan] [dim](conf={conf:.2f})[/dim]\n"
                        f"  {fact.content}\n"
                    )
                self._console.print(Panel("\n".join(lines).rstrip(), border_style="magenta"))
            except Exception as e:
                self._console.print(f"[red]Memory error: {e}[/red]")

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
                    self._console.print()
                    self._console.print(f"[bold cyan]{self._agent_name}[/bold cyan]")
                
                if response_started:
                    self._console.print(chunk, end="", highlight=False)
                    response_text += chunk
            
            # Ensure newline at end
            if response_started:
                self._console.print()
            elif self._tool_call_count > 0:
                self._console.print()
                self._console.print(f"[bold cyan]{self._agent_name}[/bold cyan]")
                self._console.print("[dim]Done.[/dim]")
        
        except Exception as e:
            self._console.print(f"\n[red]Error: {e}[/red]")