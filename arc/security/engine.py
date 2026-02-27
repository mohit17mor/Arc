"""
Security Engine — capability checking and approval flow.

Three layers of defense:
1. Policy checks (never_allow, auto_allow, always_ask)
2. Remembered decisions (from previous user choices)
3. Interactive approval (ask user when needed)
"""

from __future__ import annotations

import logging
from typing import Any

from arc.core.config import SecurityConfig
from arc.core.events import Event, EventType
from arc.core.types import Capability, SecurityDecision, ToolSpec
from arc.security.approval import ApprovalFlow

logger = logging.getLogger(__name__)


class SecurityEngine:
    """
    Checks tool calls against security policy.
    
    Usage:
        engine = SecurityEngine(config.security, kernel)
        
        # Simple check (no interactive approval)
        decision = await engine.check_tool(tool_spec, arguments)
        
        # Full check with interactive approval
        decision = await engine.check_and_approve(tool_spec, arguments)
    """
    
    def __init__(
        self,
        config: SecurityConfig,
        kernel: Any,
    ) -> None:
        self._config = config
        self._kernel = kernel
        self._approval_flow = ApprovalFlow(kernel)
        # Remember user decisions: (tool_name, capability_str) → decision
        self._remembered: dict[tuple[str, str], str] = {}

    @classmethod
    def make_permissive(cls, kernel: Any) -> "SecurityEngine":
        """
        Return a SecurityEngine that auto-approves every capability.

        Use this for background sub-agents (scheduler jobs, etc.) that run
        without a user present.  Sharing the main SecurityEngine with those
        agents causes a stdin deadlock — the approval flow tries to read from
        the terminal while the CLI is already blocking on user input.
        """
        all_caps = [c.value for c in Capability]
        permissive_config = SecurityConfig(
            auto_allow=all_caps,
            always_ask=[],
            never_allow=[],
        )
        return cls(permissive_config, kernel)
    
    @property
    def approval_flow(self) -> ApprovalFlow:
        """Access to the approval flow for resolving requests."""
        return self._approval_flow
    
    async def check_tool(
        self,
        tool_spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> SecurityDecision:
        """
        Check if a tool call is allowed by policy.
        
        This does NOT trigger interactive approval — use check_and_approve() for that.
        Returns SecurityDecision with requires_approval=True if user input is needed.
        """
        # No capabilities required = always allowed
        if not tool_spec.required_capabilities:
            return SecurityDecision(
                allowed=True,
                reason="no_capabilities_required",
            )
        
        # Track the last successful decision to preserve flags like `remembered`
        last_allowed_decision: SecurityDecision | None = None
        
        # Check each required capability
        for capability in tool_spec.required_capabilities:
            decision = self._check_capability(tool_spec.name, capability)
            
            # Return immediately on denial or if approval required
            if not decision.allowed or decision.requires_approval:
                return decision
            
            # Track the decision (to preserve remembered flag, etc.)
            last_allowed_decision = decision
        
        # All capabilities passed — return the last decision to preserve metadata
        if last_allowed_decision is not None:
            return last_allowed_decision
        
        # Fallback (shouldn't reach here if capabilities exist)
        return SecurityDecision(
            allowed=True,
            reason="policy:all_allowed",
        )
    
    async def check_and_approve(
        self,
        tool_spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> SecurityDecision:
        """
        Check policy AND request interactive approval if needed.
        
        This is the main entry point for the agent loop.
        It will block waiting for user input if approval is required.
        """
        # First, do policy check
        decision = await self.check_tool(tool_spec, arguments)
        
        # If allowed or denied by policy (not requiring approval), return immediately
        if decision.allowed or not decision.requires_approval:
            return decision
        
        # Need user approval — trigger interactive flow
        approval_decision = await self._approval_flow.request_approval(
            tool_spec,
            arguments,
        )
        
        # Remember the decision if user chose "always"
        if approval_decision.remembered and approval_decision.user_response:
            for capability in tool_spec.required_capabilities:
                self.remember_decision(
                    tool_spec.name,
                    capability,
                    approval_decision.user_response,
                )
        
        return approval_decision
    
    def _check_capability(
        self,
        tool_name: str,
        capability: Capability,
    ) -> SecurityDecision:
        """Check a single capability against policy and remembered decisions."""
        cap_str = capability.value
        
        # Layer 1: never_allow (always blocked, no override)
        if cap_str in self._config.never_allow:
            logger.info(f"Blocked {tool_name}: {cap_str} in never_allow")
            return SecurityDecision(
                allowed=False,
                reason=f"policy:never_allow ({cap_str})",
                requires_approval=False,
            )
        
        # Layer 2: Check remembered decisions (user's explicit choices)
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
                    requires_approval=False,
                )
        
        # Layer 3: auto_allow (always permitted)
        if cap_str in self._config.auto_allow:
            return SecurityDecision(
                allowed=True,
                reason=f"policy:auto_allow ({cap_str})",
            )
        
        # Layer 4: always_ask (requires user approval)
        if cap_str in self._config.always_ask:
            return SecurityDecision(
                allowed=False,
                reason=f"policy:always_ask ({cap_str})",
                requires_approval=True,
            )
        
        # Default: unknown capabilities require approval
        return SecurityDecision(
            allowed=False,
            reason=f"policy:unknown_capability ({cap_str})",
            requires_approval=True,
        )
    
    def remember_decision(
        self,
        tool_name: str,
        capability: Capability,
        decision: str,
    ) -> None:
        """
        Remember a user's decision for future calls.
        
        Args:
            tool_name: Name of the tool
            capability: The capability (enum or will use .value)
            decision: One of "allow_always", "deny_always"
        """
        # Handle both Capability enum and string
        if isinstance(capability, Capability):
            cap_str = capability.value
        else:
            cap_str = str(capability)
        
        key = (tool_name, cap_str)
        self._remembered[key] = decision
        logger.debug(f"Remembered {decision} for {tool_name}/{cap_str}")
    
    def clear_remembered(self) -> None:
        """Clear all remembered decisions."""
        self._remembered.clear()
        logger.debug("Cleared all remembered security decisions")
    
    def get_remembered(self) -> dict[tuple[str, str], str]:
        """Get all remembered decisions (for debugging/display)."""
        return dict(self._remembered)
    
    def shutdown(self) -> None:
        """Clean up resources. Call during kernel shutdown."""
        self._approval_flow.cancel_all()