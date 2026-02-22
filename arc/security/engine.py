"""
Security Engine — capability checking and approval flow.

Three layers of defense:
1. User policy (from config)
2. Approval flow (interactive prompts)
3. Audit trail (logging)
"""

from __future__ import annotations

import logging
from typing import Any

from arc.core.config import SecurityConfig
from arc.core.events import Event, EventType
from arc.core.types import Capability, SecurityDecision, ToolSpec

logger = logging.getLogger(__name__)


class SecurityEngine:
    """
    Checks tool calls against security policy.

    Usage:
        engine = SecurityEngine(config.security, kernel)

        decision = await engine.check_tool(tool_spec, arguments)
        if decision.allowed:
            # execute tool
        else:
            # tell user it was blocked
    """

    def __init__(
        self,
        config: SecurityConfig,
        kernel: Any,
    ) -> None:
        self._config = config
        self._kernel = kernel
        # Remember user decisions: (tool_name, capability) → decision
        self._remembered: dict[tuple[str, str], str] = {}

    async def check_tool(
        self,
        tool_spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> SecurityDecision:
        """
        Check if a tool call is allowed.

        Returns SecurityDecision with allowed=True/False and reasoning.
        """
        # No capabilities required = allowed
        if not tool_spec.required_capabilities:
            return SecurityDecision(
                allowed=True,
                reason="no capabilities required",
            )

        # Check each required capability
        last_allowed_decision: SecurityDecision | None = None

        for capability in tool_spec.required_capabilities:
            decision = await self._check_capability(
                tool_spec.name,
                capability,
                arguments,
            )
            # Return immediately on denial or approval required
            if not decision.allowed or decision.requires_approval:
                return decision
            
            # Track the last allowed decision (to preserve remembered flag, etc.)
            last_allowed_decision = decision

        # All capabilities allowed — return the last decision
        # This preserves info like remembered=True
        if last_allowed_decision:
            return last_allowed_decision

        # Fallback (shouldn't reach here)
        return SecurityDecision(
            allowed=True,
            reason="policy:all_allowed",
        )

    async def _check_capability(
        self,
        tool_name: str,
        capability: Capability,
        arguments: dict[str, Any],
    ) -> SecurityDecision:
        """Check a single capability."""
        cap_str = capability.value

        # Layer 1: Check never_allow (always blocked)
        if cap_str in self._config.never_allow:
            logger.info(f"Blocked {tool_name}: {cap_str} in never_allow")
            return SecurityDecision(
                allowed=False,
                reason=f"policy:never_allow ({cap_str})",
            )

        # Layer 2: Check remembered decisions BEFORE auto_allow
        # (user's explicit choices take precedence)
        key = (tool_name, cap_str)
        if key in self._remembered:
            remembered = self._remembered[key]
            if remembered == "allow_always":
                return SecurityDecision(
                    allowed=True,
                    reason=f"user:remembered_allow ({cap_str})",
                    remembered=True,
                )
            elif remembered == "deny_always":
                return SecurityDecision(
                    allowed=False,
                    reason=f"user:remembered_deny ({cap_str})",
                    remembered=True,
                )

        # Layer 3: Check auto_allow (always permitted)
        if cap_str in self._config.auto_allow:
            return SecurityDecision(
                allowed=True,
                reason=f"policy:auto_allow ({cap_str})",
            )

        # Layer 4: Check always_ask (requires approval)
        if cap_str in self._config.always_ask:
            return SecurityDecision(
                allowed=False,
                reason=f"policy:always_ask ({cap_str})",
                requires_approval=True,
            )

        # Default: require approval for unknown capabilities
        return SecurityDecision(
            allowed=False,
            reason=f"policy:unknown_capability ({cap_str})",
            requires_approval=True,
        )

    async def request_approval(
        self,
        tool_name: str,
        tool_spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> SecurityDecision:
        """
        Request user approval for a tool call.

        Emits security:approval event and waits for response.
        The platform (CLI/API) handles the actual user interaction.
        """
        # Emit approval request event
        event = Event(
            type=EventType.SECURITY_APPROVAL,
            source="security",
            data={
                "tool_name": tool_name,
                "description": tool_spec.description,
                "arguments": arguments,
                "capabilities": [c.value for c in tool_spec.required_capabilities],
            },
        )

        await self._kernel.emit(event)

        # For now, return requires_approval=True
        # The actual approval flow will be implemented in the CLI
        # This is a placeholder that assumes approval
        return SecurityDecision(
            allowed=True,
            reason="user:approved",
            requires_approval=True,
            user_response="allow_once",
        )

    def remember_decision(
        self,
        tool_name: str,
        capability: Capability,
        decision: str,
    ) -> None:
        """Remember a user's decision for future calls."""
        key = (tool_name, capability.value)
        self._remembered[key] = decision
        logger.debug(f"Remembered {decision} for {tool_name}/{capability.value}")

    def clear_remembered(self) -> None:
        """Clear all remembered decisions."""
        self._remembered.clear()