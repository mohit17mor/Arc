"""
Security Approval Flow — interactive permission prompts.

When a tool requires approval, this module:
1. Emits a security:approval event
2. Waits for a response from the platform (CLI/API/etc.)
3. Returns the decision

The platform is responsible for showing the prompt and collecting user input.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from arc.core.events import Event, EventType
from arc.core.types import SecurityDecision, ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRequest:
    """A pending approval request."""
    
    request_id: str
    tool_name: str
    tool_description: str
    arguments: dict[str, Any]
    capabilities: list[str]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class ApprovalFlow:
    """
    Manages the approval request/response flow.
    
    Flow:
    1. Agent calls request_approval()
    2. ApprovalFlow emits security:approval event
    3. Platform (CLI) shows prompt, user makes choice
    4. Platform calls resolve_approval() with the response
    5. request_approval() returns with the decision
    
    Usage:
        flow = ApprovalFlow(kernel)
        
        # In agent loop:
        decision = await flow.request_approval(tool_spec, arguments)
        
        # In CLI (event handler):
        flow.resolve_approval(request_id, "allow_once")
    """
    
    def __init__(self, kernel: Any, timeout: float = 300.0) -> None:
        self._kernel = kernel
        self._timeout = timeout  # 5 minutes default
        self._pending: dict[str, ApprovalRequest] = {}
        self._request_counter = 0
    
    async def request_approval(
        self,
        tool_spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> SecurityDecision:
        """
        Request user approval for a tool call.
        
        Emits security:approval event and waits for response.
        Returns SecurityDecision based on user's choice.
        """
        # Generate unique request ID
        self._request_counter += 1
        request_id = f"approval_{self._request_counter}"
        
        # Create pending request
        try:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()
        except RuntimeError:
            # Fallback for edge cases
            future = asyncio.get_event_loop().create_future()
        
        request = ApprovalRequest(
            request_id=request_id,
            tool_name=tool_spec.name,
            tool_description=tool_spec.description,
            arguments=arguments,
            capabilities=[c.value for c in tool_spec.required_capabilities],
            future=future,
        )
        
        self._pending[request_id] = request
        
        # Emit approval event
        event = Event(
            type=EventType.SECURITY_APPROVAL,
            source="security",
            data={
                "request_id": request_id,
                "tool_name": tool_spec.name,
                "tool_description": tool_spec.description,
                "arguments": arguments,
                "capabilities": request.capabilities,
            },
        )
        
        await self._kernel.emit(event)
        logger.debug(f"Approval requested: {request_id} for {tool_spec.name}")
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=self._timeout)
            logger.debug(f"Approval response: {request_id} = {response}")
        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout: {request_id}")
            self._pending.pop(request_id, None)
            return SecurityDecision(
                allowed=False,
                reason="approval_timeout",
                requires_approval=False,
            )
        finally:
            self._pending.pop(request_id, None)
        
        # Convert response to SecurityDecision
        return self._response_to_decision(response)
    
    def resolve_approval(self, request_id: str, response: str) -> bool:
        """
        Resolve a pending approval request.
        
        Called by the platform when user makes a choice.
        
        Args:
            request_id: The request ID from the approval event
            response: One of "allow_once", "allow_always", "deny", "deny_always"
        
        Returns:
            True if request was found and resolved, False otherwise
        """
        request = self._pending.get(request_id)
        if request is None:
            logger.warning(f"Unknown approval request: {request_id}")
            return False
        
        if not request.future.done():
            request.future.set_result(response)
            return True
        
        return False
    
    def _response_to_decision(self, response: str) -> SecurityDecision:
        """Convert user response string to SecurityDecision."""
        response = response.lower().strip()
        
        if response == "allow_once":
            return SecurityDecision(
                allowed=True,
                reason="user:approved_once",
                requires_approval=False,
                user_response="allow_once",
                remembered=False,
            )
        elif response == "allow_always":
            return SecurityDecision(
                allowed=True,
                reason="user:approved_always",
                requires_approval=False,
                user_response="allow_always",
                remembered=True,
            )
        elif response == "deny_always":
            return SecurityDecision(
                allowed=False,
                reason="user:denied_always",
                requires_approval=False,
                user_response="deny_always",
                remembered=True,
            )
        else:  # "deny" or anything else
            return SecurityDecision(
                allowed=False,
                reason="user:denied",
                requires_approval=False,
                user_response="deny",
                remembered=False,
            )
    
    @property
    def pending_count(self) -> int:
        """Number of pending approval requests."""
        return len(self._pending)
    
    def cancel_all(self) -> None:
        """Cancel all pending requests. Used during shutdown."""
        for request in self._pending.values():
            if not request.future.done():
                request.future.cancel()
        self._pending.clear()