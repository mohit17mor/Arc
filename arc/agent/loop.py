"""
Agent Loop — the core think → act → observe cycle.

This is where the magic happens. The agent:
1. Composes context from memory
2. Sends to LLM (think)
3. Executes any tool calls (act) — with security approval
4. Processes results (observe)
5. Repeats until done or max iterations
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from arc.memory.manager import MemoryManager

from arc.core.errors import LLMError
from arc.core.events import Event, EventType
from arc.core.kernel import Kernel
from arc.core.types import (
    AgentState,
    AgentStatus,
    Message,
    StopReason,
    ToolCall,
    ToolResult,
)
from arc.llm.base import LLMProvider
from arc.memory.compaction import CompactionState
from arc.core.run_control import RunCancelledError, RunControlManager, RunHandle
from arc.memory.context import ContextComposer
from arc.memory.session import SessionMemory
from arc.security.engine import SecurityEngine
from arc.skills.builtin.planning import PlanningSkill
from arc.skills.manager import SkillManager
from arc.skills.router import SkillRouter

logger = logging.getLogger(__name__)

# LLM retry configuration
_MAX_LLM_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry
_REPEATED_FAILURE_THRESHOLD = 2
_META_TURN_PATTERNS = (
    re.compile(r"\bwhy did you\b"),
    re.compile(r"\bwhy didn't you\b"),
    re.compile(r"\bwhy did you not\b"),
    re.compile(r"\bwhat happened\b"),
    re.compile(r"\bwhat went wrong\b"),
    re.compile(r"\bexplain what happened\b"),
    re.compile(r"\bwhy were you\b"),
)


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""
    
    max_iterations: int = 25
    temperature: float = 0.7
    recent_window: int = 20
    excluded_skills: frozenset[str] = field(default_factory=frozenset)
    """Skill names whose tools are hidden from the LLM for this agent instance.
    Useful for sub-agents that should not be able to schedule jobs, manage
    memory, or perform other meta-level operations.
    """


class AgentLoop:
    """
    The agent's execution loop.
    
    Each turn:
    1. COMPOSE — Build messages from memory
    2. THINK — Call LLM with composed context
    3. ACT — Execute any tool calls (with security checks)
    4. OBSERVE — Process results, decide if done
    5. REMEMBER — Store the turn in memory

    Usage:
        loop = AgentLoop(
            kernel=kernel,
            llm=ollama_provider,
            skill_manager=skill_manager,
            security=security_engine,
            system_prompt="You are a helpful assistant.",
        )

        async for chunk in loop.run("What files are in this directory?"):
            print(chunk, end="", flush=True)
    """
    
    def __init__(
        self,
        kernel: Kernel,
        llm: LLMProvider,
        skill_manager: SkillManager,
        security: SecurityEngine,
        system_prompt: str,
        config: AgentConfig | None = None,
        memory_manager: MemoryManager | None = None,
        agent_id: str = "main",
        router: SkillRouter | None = None,
        run_control: RunControlManager | None = None,
    ) -> None:
        self._kernel = kernel
        self._llm = llm
        self._skills = skill_manager
        self._security = security
        self._config = config or AgentConfig()
        self._memory_manager = memory_manager
        self._agent_id = agent_id
        self._router = router
        self._run_control = run_control
        
        # Planning — each agent gets its own PlanningSkill instance
        self._planning = PlanningSkill()
        self._planning_initialized = False
        
        # Memory
        self._memory = SessionMemory()
        self._memory.set_system_prompt(system_prompt)
        
        # Context composer
        model_info = llm.get_model_info()
        self._composer = ContextComposer(
            token_counter=llm.count_tokens,
            max_tokens=model_info.context_window,
            reserve_output=model_info.max_output_tokens,
        )
        self._context_window = model_info.context_window
        
        # State
        self._state = AgentState(agent_id="agent")
        self._iteration = 0
        self._explain_only_reason: str | None = None
        self._failed_tool_signatures: dict[str, int] = {}
        self._active_run_handle: RunHandle | None = None
        self._current_run_id: str | None = None
        self._last_run_id: str | None = None
        
        # Compaction — background for main agent, sync for others
        self._compaction = CompactionState()
        self._is_main_agent = (agent_id == "main")

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt used for future LLM calls."""
        self._memory.set_system_prompt(prompt)

    async def run(self, user_input: str) -> AsyncIterator[str]:
        """
        Process user input and yield response chunks.
        
        This is a streaming generator — chunks are yielded as they arrive.
        """
        run_handle: RunHandle | None = None
        if self._run_control is not None:
            run_handle = self._run_control.start_run(
                kind="agent",
                source=self._agent_id,
                metadata={"input_preview": user_input[:200], "agent_id": self._agent_id},
            )
            self._active_run_handle = run_handle
            self._current_run_id = run_handle.run_id
            self._last_run_id = run_handle.run_id

        # Add user message to memory
        self._memory.add_user_message(user_input)
        self._state.status = AgentStatus.COMPOSING
        self._iteration = 0
        self._failed_tool_signatures = {}
        self._explain_only_reason = None

        if self._is_meta_turn(user_input):
            self._explain_only_reason = (
                "This turn is asking for explanation or reflection. "
                "Do not use tools. Explain directly what happened and why."
            )

        # Initialize planning skill (once, lazy)
        if not self._planning_initialized:
            await self._planning.initialize(self._kernel, {"agent_id": self._agent_id})
            self._planning_initialized = True

        # Reset the router so each user turn starts with a clean slate
        if self._router:
            self._router.reset()
        
        await self._emit(EventType.AGENT_START, {"input": user_input})
        
        try:
            while self._iteration < self._config.max_iterations:
                await self._run_checkpoint()
                self._iteration += 1
                
                await self._emit(
                    EventType.AGENT_THINKING,
                    {"iteration": self._iteration},
                )
                
                # 1. COMPOSE context
                
                # Apply pending compaction (main agent, background mode)
                if self._is_main_agent:
                    self._compaction.apply_if_ready(self._memory)
                
                context = await self._composer.compose(
                    session=self._memory,
                    recent_window=self._config.recent_window,
                    query=user_input,
                    memory_manager=self._memory_manager,
                )
                
                # Sync compaction for background agents (task/worker/scheduler)
                if not self._is_main_agent:
                    compacted = await self._compaction.maybe_compact_sync(
                        session=self._memory,
                        token_count=context.token_count,
                        token_budget=self._composer.token_budget,
                        llm=self._llm,
                    )
                    if compacted:
                        # Re-compose after compaction
                        context = await self._composer.compose(
                            session=self._memory,
                            recent_window=self._config.recent_window,
                            query=user_input,
                            memory_manager=self._memory_manager,
                        )
                
                # Inject current plan into the system message so the LLM
                # sees it on EVERY iteration (mechanism 2: plan always visible).
                if self._planning.has_plan and context.messages:
                    plan_text = self._planning.format_plan_for_context()
                    first = context.messages[0]
                    if first.role == "system" and first.content:
                        context.messages[0] = Message.system(
                            first.content + "\n\n" + plan_text
                        )
                
                # 2. THINK — call LLM
                self._state.status = AgentStatus.THINKING
                
                # Get available tools
                if self._explain_only_reason:
                    tool_specs = []
                elif self._router:
                    # Two-tier: always-on + activated + use_skill meta-tool
                    tool_specs = self._router.get_active_tool_specs()
                else:
                    # Flat mode (legacy / no router): all tools minus excluded
                    excluded = self._config.excluded_skills
                    all_specs = self._skills.get_all_tool_specs()
                    tool_specs = [
                        ts for ts in all_specs
                        if self._skills.get_tool_skill(ts.name) not in excluded
                    ] if excluded else all_specs
                
                # Always include the planning tool unless this turn is locked
                # into explanation mode.
                if not self._explain_only_reason:
                    planning_specs = list(self._planning.manifest().tools)
                    # Avoid duplicates if somehow already present
                    existing_names = {ts.name for ts in tool_specs}
                    for ps in planning_specs:
                        if ps.name not in existing_names:
                            tool_specs.append(ps)
                
                collected_text = ""
                collected_tool_calls: list[ToolCall] = []
                stop_reason: StopReason | None = None
                input_tokens = 0
                output_tokens = 0
                cached_input_tokens = 0
                
                # ── LLM call with retry ──────────────────────────────────
                llm_error: Exception | None = None
                for _attempt in range(_MAX_LLM_RETRIES):
                    try:
                        async for chunk in self._llm.generate(
                            messages=context.messages,
                            tools=tool_specs if tool_specs else None,
                            temperature=self._config.temperature,
                        ):
                            await self._run_checkpoint()
                            # Stream text to caller
                            if chunk.text:
                                collected_text += chunk.text
                                yield chunk.text
                            
                            # Collect tool calls
                            if chunk.tool_calls:
                                collected_tool_calls.extend(chunk.tool_calls)
                            
                            if chunk.stop_reason:
                                stop_reason = chunk.stop_reason
                                input_tokens = chunk.input_tokens
                                output_tokens = chunk.output_tokens
                                cached_input_tokens = chunk.cached_input_tokens

                        llm_error = None
                        break  # success — exit retry loop

                    except LLMError as e:
                        llm_error = e
                        retryable = getattr(e, "retryable", False)
                        if not retryable or _attempt >= _MAX_LLM_RETRIES - 1:
                            break  # non-retryable or last attempt
                        delay = _RETRY_BASE_DELAY * (2 ** _attempt)
                        logger.warning(
                            f"LLM error (attempt {_attempt + 1}/{_MAX_LLM_RETRIES},"
                            f" retrying in {delay}s): {e}"
                        )
                        await asyncio.sleep(delay)
                        # Reset for retry — text already streamed can't be un-sent,
                        # but tool calls from a partial response must be cleared
                        collected_tool_calls.clear()
                        stop_reason = None

                    except Exception as e:
                        llm_error = e
                        break  # unknown error — don't retry

                # ── Handle LLM failure gracefully ────────────────────────
                if llm_error is not None:
                    error_msg = (
                        f"\n\nI encountered an error communicating with the LLM: "
                        f"{llm_error}\n\nPlease try again."
                    )
                    yield error_msg
                    # Store the partial exchange so conversation state stays clean
                    self._memory.add_assistant_message(
                        (collected_text + error_msg) if collected_text else error_msg
                    )
                    self._state.status = AgentStatus.COMPLETE
                    await self._emit(
                        EventType.AGENT_ERROR,
                        {"error": str(llm_error), "recovered": True},
                    )
                    return
                
                # Emit LLM response event
                await self._emit(
                    EventType.LLM_RESPONSE,
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "stop_reason": stop_reason.value if stop_reason else None,
                        "has_tool_calls": len(collected_tool_calls) > 0,
                    },
                )

                if (
                    self._explain_only_reason
                    and collected_tool_calls
                    and self._iteration < self._config.max_iterations
                ):
                    self._memory.add_assistant_message(collected_text or None)
                    self._memory.add_user_message(self._explain_only_reason)
                    continue

                if (
                    not self._explain_only_reason
                    and not self._planning.has_active_plan
                    and self._should_require_plan(collected_tool_calls)
                    and not any(tc.name == "update_plan" for tc in collected_tool_calls)
                    and self._iteration < self._config.max_iterations
                ):
                    self._memory.add_assistant_message(collected_text or None)
                    self._memory.add_user_message(
                        "Before using tools for this task, create a short plan "
                        "with update_plan first. Use 3-5 steps, then continue."
                    )
                    continue
                
                # 3. Check if done (no tool calls)
                if stop_reason == StopReason.COMPLETE or not collected_tool_calls:
                    # ── Verification: did the LLM promise an action but not call a tool? ──
                    # Some LLMs say "I'll search for that" or "Let me look that up"
                    # in text but finish without emitting a tool_call.  Detect this
                    # pattern and nudge the LLM to actually follow through.
                    if (
                        collected_text
                        and not collected_tool_calls
                        and tool_specs
                        and self._iteration < self._config.max_iterations
                        and self._text_promises_action(collected_text)
                    ):
                        logger.debug(
                            "LLM promised action in text but didn't call a tool — nudging"
                        )
                        self._memory.add_assistant_message(collected_text)
                        self._memory.add_user_message(
                            "You said you would take an action, but you didn't call "
                            "any tool. Please actually call the appropriate tool now "
                            "instead of just describing what you would do."
                        )
                        # Continue the loop — next iteration will re-compose and call LLM
                        continue

                    # ── Plan enforcement (mechanism 4): nudge on incomplete plan ──
                    if (
                        not self._explain_only_reason
                        and self._planning.has_active_plan
                        and self._planning.has_incomplete_steps
                        and self._iteration < self._config.max_iterations
                    ):
                        logger.debug(
                            "LLM declared done but plan has incomplete steps — nudging"
                        )
                        self._memory.add_assistant_message(collected_text)
                        self._memory.add_user_message(
                            "You still have unfinished steps in your plan. "
                            "Please complete the remaining steps or update "
                            "your plan to reflect what was actually done."
                        )
                        continue

                    self._cleanup_completed_plan_state()
                    self._memory.add_assistant_message(collected_text)
                    self._state.status = AgentStatus.COMPLETE

                    # Fire-and-forget background memory tasks
                    self._fire_memory_tasks(user_input, collected_text)
                    
                    # Trigger background compaction if approaching limit
                    # (main agent only — bg agents use sync in compose)
                    if self._is_main_agent:
                        self._compaction.check_and_start_background(
                            session=self._memory,
                            token_count=input_tokens,
                            token_budget=self._context_window,
                            llm=self._llm,
                        )

                    await self._emit(
                        EventType.AGENT_COMPLETE,
                        {"iterations": self._iteration},
                    )
                    if run_handle is not None:
                        await run_handle.finish_completed()
                    return
                
                # 4. ACT — execute tool calls
                self._state.status = AgentStatus.ACTING
                
                # Store assistant message with tool calls
                self._memory.add_assistant_message(
                    content=collected_text if collected_text else None,
                    tool_calls=collected_tool_calls,
                )

                tool_results: list[tuple[ToolCall, ToolResult]] = []
                for tool_call in collected_tool_calls:
                    await self._run_checkpoint()
                    # Intercept update_plan — handled by per-agent PlanningSkill
                    if tool_call.name == "update_plan":
                        result = await self._planning.execute_tool(
                            "update_plan", tool_call.arguments
                        )
                        result.tool_call_id = tool_call.id
                        self._memory.add_tool_result(result, tool_call.name)
                    # Intercept use_skill — handled by router, not by skills
                    elif self._router and self._router.is_use_skill_call(tool_call.name):
                        msg = self._router.activate(
                            tool_call.arguments.get("skill_name", "")
                        )
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            success=True,
                            output=msg,
                        )
                        self._memory.add_tool_result(result, tool_call.name)
                    else:
                        try:
                            result = await self._execute_tool_with_approval(tool_call)
                        except Exception as e:
                            logger.warning(
                                f"Tool execution crashed for {tool_call.name}: {e}"
                            )
                            result = ToolResult(
                                tool_call_id=tool_call.id,
                                success=False,
                                output="",
                                error=f"Tool execution crashed: {e}",
                            )
                        self._memory.add_tool_result(result, tool_call.name)
                    tool_results.append((tool_call, result))

                breaker_reason = self._update_failure_loop_state(tool_results)
                if (
                    breaker_reason
                    and self._iteration < self._config.max_iterations
                ):
                    self._explain_only_reason = breaker_reason
                    self._memory.add_user_message(breaker_reason)
                
                # 5. OBSERVE — loop continues with tool results in context
            
            # Max iterations reached — synthesise with everything gathered so far
            # rather than silently dropping the context.
            yield "\n\n"
            synthesis_text = ""
            async for chunk in self._synthesise_on_limit():
                synthesis_text += chunk
                yield chunk

            # Store synthesis turn in memory (background)
            self._cleanup_completed_plan_state()
            self._fire_memory_tasks(user_input, synthesis_text)

            self._state.status = AgentStatus.COMPLETE
            await self._emit(
                EventType.AGENT_COMPLETE,
                {"iterations": self._iteration, "reason": "max_iterations"},
            )
            if run_handle is not None:
                await run_handle.finish_completed()
        
        except RunCancelledError as e:
            self._state.status = AgentStatus.COMPLETE
            await self._planning.mark_interrupted(reason=e.action.value)
            await self._emit(
                EventType.AGENT_COMPLETE,
                {"iterations": self._iteration, "reason": e.action.value},
            )
            return

        except Exception as e:
            self._state.status = AgentStatus.ERROR
            if run_handle is not None:
                run_handle.finish_failed()
            await self._emit(EventType.AGENT_ERROR, {"error": str(e)})
            # Store error in memory so conversation state stays consistent
            error_text = f"\n\nSorry, I encountered an unexpected error: {e}"
            yield error_text
            self._memory.add_assistant_message(error_text)
            self._state.status = AgentStatus.COMPLETE
    
        finally:
            self._active_run_handle = None
            self._current_run_id = None

    def _fire_memory_tasks(self, user_input: str, assistant_text: str) -> None:
        """Schedule background memory storage tasks (fire-and-forget)."""
        if self._memory_manager is None:
            return
        session_id = id(self._memory)  # stable ID within this session
        asyncio.create_task(
            self._memory_manager.store_turn(
                user_content=user_input,
                assistant_content=assistant_text,
                session_id=str(session_id),
            )
        )
        if self._memory_manager.should_distill:
            recent = self._memory.get_messages()[-self._config.recent_window :]
            asyncio.create_task(
                self._memory_manager.distill_to_core(
                    messages=recent,
                    llm=self._llm,
                )
            )

    def _cleanup_completed_plan_state(self) -> None:
        """Drop completed plan chatter after a run has fully finished."""
        if not self._planning.is_completed:
            return
        self._memory.prune_tool_history("update_plan")
        self._planning.clear_completed()

    async def _synthesise_on_limit(self) -> AsyncIterator[str]:
        """
        Called when max_iterations is exhausted.

        Injects a final user nudge and calls the LLM **without tools** so it
        is forced to produce a text answer from whatever context is already in
        memory — tool results, page content, search snippets, etc.
        """
        context = await self._composer.compose(
            session=self._memory,
            recent_window=self._config.recent_window,
        )
        # Append a nudge as a user turn so the model sees it as a new instruction.
        nudge = Message.user(
            "You have used the maximum number of tool calls. "
            "Do NOT call any more tools. "
            "Based solely on the information you have gathered in this conversation, "
            "provide your best complete answer to the original question right now."
        )
        async for chunk in self._llm.generate(
            messages=context.messages + [nudge],
            tools=None,   # no tools — forces a text completion
            temperature=self._config.temperature,
        ):
            if chunk.text:
                yield chunk.text

    async def _execute_tool_with_approval(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call with security checks and approval flow."""
        
        # Get tool spec — O(1) lookup
        tool_spec = self._skills.get_tool_spec(tool_call.name)
        
        if tool_spec is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                output="",
                error=f"Unknown tool: {tool_call.name}",
            )
        
        # Full security check with interactive approval
        decision = await self._security.check_and_approve(
            tool_spec,
            tool_call.arguments,
        )
        
        if not decision.allowed:
            await self._emit(
                EventType.SECURITY_DENIED,
                {
                    "tool": tool_call.name,
                    "reason": decision.reason,
                },
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                output="",
                error=f"Permission denied: {decision.reason}",
            )
        
        # Permission granted — emit tool call event and execute
        await self._emit(
            EventType.SKILL_TOOL_CALL,
            {"tool": tool_call.name, "arguments": tool_call.arguments},
        )
        
        result = await self._skills.execute_tool(tool_call.name, tool_call.arguments)
        result.tool_call_id = tool_call.id
        
        await self._emit(
            EventType.SKILL_TOOL_RESULT,
            {
                "tool": tool_call.name,
                "success": result.success,
                "output_preview": result.output[:200] if result.output else "",
            },
        )
        
        return result
    
    async def _run_checkpoint(self) -> None:
        if self._active_run_handle is None:
            return
        await self._active_run_handle.checkpoint()

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event tagged with this agent's id."""
        event = Event(type=event_type, source=self._agent_id, data=data)
        await self._kernel.emit(event)
    
    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    @property
    def last_run_id(self) -> str | None:
        return self._last_run_id

    @property
    def memory(self) -> SessionMemory:
        """Access to session memory."""
        return self._memory
    
    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state
    
    @property
    def security(self) -> SecurityEngine:
        """Access to security engine (for CLI to set approval flow)."""
        return self._security
    
    def reset(self) -> None:
        """Reset the agent for a new conversation."""
        self._memory.clear()
        self._state = AgentState(agent_id="agent")
        self._iteration = 0
        self._explain_only_reason = None
        self._failed_tool_signatures = {}

    # ━━━ Action verification ━━━

    # Phrases that indicate the LLM intends to take an action but may
    # have finished without actually emitting a tool call.
    _ACTION_PHRASES: tuple[str, ...] = (
        "let me ",
        "i'll ",
        "i will ",
        "let me search",
        "let me look",
        "let me check",
        "let me find",
        "let me read",
        "let me run",
        "let me open",
        "let me browse",
        "let me navigate",
        "i'll search",
        "i'll look",
        "i'll check",
        "i'll find",
        "i'll read",
        "i'll run",
        "i'll open",
        "i'll browse",
        "i'll navigate",
        "i'll use the",
        "i'll call",
        "i will search",
        "i will look",
        "i will check",
        "i will use",
        "searching for",
        "looking up",
        "checking ",
        "running the",
        "calling the",
        "using the tool",
        "activate the",
        "use_skill",
    )

    @classmethod
    def _text_promises_action(cls, text: str) -> bool:
        """
        Detect if the LLM's text response promises a tool action.

        Returns True if the text contains phrases like "Let me search",
        "I'll look that up", etc. — indicating the LLM intended to call
        a tool but completed without actually doing so.

        Only checks the first ~500 chars — action promises are always
        at the beginning of the response, not buried in a long answer.
        """
        prefix = text[:500].lower()
        return any(phrase in prefix for phrase in cls._ACTION_PHRASES)

    @staticmethod
    def _is_meta_turn(text: str) -> bool:
        prefix = text[:300].lower()
        return any(pattern.search(prefix) for pattern in _META_TURN_PATTERNS)

    @staticmethod
    def _tool_signature(tool_call: ToolCall) -> str:
        try:
            payload = json.dumps(tool_call.arguments, sort_keys=True, default=str)
        except TypeError:
            payload = str(tool_call.arguments)
        return f"{tool_call.name}:{payload}"

    @classmethod
    def _should_require_plan(cls, tool_calls: list[ToolCall]) -> bool:
        return any(tc.name != "update_plan" for tc in tool_calls)

    def _update_failure_loop_state(
        self,
        tool_results: list[tuple[ToolCall, ToolResult]],
    ) -> str | None:
        for tool_call, result in tool_results:
            signature = self._tool_signature(tool_call)
            if result.success:
                self._failed_tool_signatures.pop(signature, None)
                continue

            failures = self._failed_tool_signatures.get(signature, 0) + 1
            self._failed_tool_signatures[signature] = failures
            if failures >= _REPEATED_FAILURE_THRESHOLD:
                return (
                    "Several recent tool attempts failed. Stop using tools for "
                    "this turn and explain what you tried, what failed, and "
                    "what should be done next."
                )

        return None
