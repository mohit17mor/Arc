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
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from arc.memory.manager import MemoryManager

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
from arc.memory.context import ContextComposer
from arc.memory.session import SessionMemory
from arc.security.engine import SecurityEngine
from arc.skills.manager import SkillManager

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._kernel = kernel
        self._llm = llm
        self._skills = skill_manager
        self._security = security
        self._config = config or AgentConfig()
        self._memory_manager = memory_manager
        
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
        
        # State
        self._state = AgentState(agent_id="agent")
        self._iteration = 0
    
    async def run(self, user_input: str) -> AsyncIterator[str]:
        """
        Process user input and yield response chunks.
        
        This is a streaming generator — chunks are yielded as they arrive.
        """
        # Add user message to memory
        self._memory.add_user_message(user_input)
        self._state.status = AgentStatus.COMPOSING
        self._iteration = 0
        
        await self._emit(EventType.AGENT_START, {"input": user_input})
        
        try:
            while self._iteration < self._config.max_iterations:
                self._iteration += 1
                
                await self._emit(
                    EventType.AGENT_THINKING,
                    {"iteration": self._iteration},
                )
                
                # 1. COMPOSE context
                context = await self._composer.compose(
                    session=self._memory,
                    recent_window=self._config.recent_window,
                    query=user_input,
                    memory_manager=self._memory_manager,
                )
                
                # 2. THINK — call LLM
                self._state.status = AgentStatus.THINKING
                
                # Get available tools, excluding any skills blocked for this agent
                excluded = self._config.excluded_skills
                all_specs = self._skills.get_all_tool_specs()
                tool_specs = [
                    ts for ts in all_specs
                    if self._skills.get_tool_skill(ts.name) not in excluded
                ] if excluded else all_specs
                
                collected_text = ""
                collected_tool_calls: list[ToolCall] = []
                stop_reason: StopReason | None = None
                input_tokens = 0
                output_tokens = 0
                
                async for chunk in self._llm.generate(
                    messages=context.messages,
                    tools=tool_specs if tool_specs else None,
                    temperature=self._config.temperature,
                ):
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
                
                # Emit LLM response event
                await self._emit(
                    EventType.LLM_RESPONSE,
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "stop_reason": stop_reason.value if stop_reason else None,
                        "has_tool_calls": len(collected_tool_calls) > 0,
                    },
                )
                
                # 3. Check if done (no tool calls)
                if stop_reason == StopReason.COMPLETE or not collected_tool_calls:
                    self._memory.add_assistant_message(collected_text)
                    self._state.status = AgentStatus.COMPLETE

                    # Fire-and-forget background memory tasks
                    self._fire_memory_tasks(user_input, collected_text)

                    await self._emit(
                        EventType.AGENT_COMPLETE,
                        {"iterations": self._iteration},
                    )
                    return
                
                # 4. ACT — execute tool calls
                self._state.status = AgentStatus.ACTING
                
                # Store assistant message with tool calls
                self._memory.add_assistant_message(
                    content=collected_text if collected_text else None,
                    tool_calls=collected_tool_calls,
                )
                
                for tool_call in collected_tool_calls:
                    result = await self._execute_tool_with_approval(tool_call)
                    self._memory.add_tool_result(result, tool_call.name)
                
                # 5. OBSERVE — loop continues with tool results in context
            
            # Max iterations reached — synthesise with everything gathered so far
            # rather than silently dropping the context.
            yield "\n\n"
            synthesis_text = ""
            async for chunk in self._synthesise_on_limit():
                synthesis_text += chunk
                yield chunk

            # Store synthesis turn in memory (background)
            self._fire_memory_tasks(user_input, synthesis_text)

            self._state.status = AgentStatus.COMPLETE
            await self._emit(
                EventType.AGENT_COMPLETE,
                {"iterations": self._iteration, "reason": "max_iterations"},
            )
        
        except Exception as e:
            self._state.status = AgentStatus.ERROR
            await self._emit(EventType.AGENT_ERROR, {"error": str(e)})
            raise
    
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
        
        # Get tool spec
        tool_specs = self._skills.get_all_tool_specs()
        tool_spec = next((t for t in tool_specs if t.name == tool_call.name), None)
        
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
    
    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event."""
        event = Event(type=event_type, source="agent", data=data)
        await self._kernel.emit(event)
    
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