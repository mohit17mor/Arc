"""Tests for the Security Approval Flow."""

import asyncio
import pytest

from arc.core.kernel import Kernel
from arc.core.config import ArcConfig, SecurityConfig
from arc.core.types import Capability, ToolSpec
from arc.security.approval import ApprovalFlow, ApprovalRequest
from arc.security.engine import SecurityEngine


@pytest.fixture
def kernel():
    return Kernel(config=ArcConfig())


@pytest.fixture
def approval_flow(kernel):
    return ApprovalFlow(kernel, timeout=1.0)  # Short timeout for tests


@pytest.fixture
def write_tool():
    return ToolSpec(
        name="write_file",
        description="Write content to a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        },
        required_capabilities=frozenset([Capability.FILE_WRITE]),
    )


@pytest.fixture
def shell_tool():
    return ToolSpec(
        name="execute",
        description="Execute a shell command",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
        },
        required_capabilities=frozenset([Capability.SHELL_EXEC]),
    )


class TestApprovalFlow:
    """Tests for ApprovalFlow class."""
    
    @pytest.mark.asyncio
    async def test_request_creates_pending(self, approval_flow, write_tool):
        """Requesting approval creates a pending request."""
        # Start approval request (don't await, will timeout)
        task = asyncio.create_task(
            approval_flow.request_approval(write_tool, {"path": "test.txt"})
        )
        
        # Give it a moment to register
        await asyncio.sleep(0.05)
        
        assert approval_flow.pending_count == 1
        
        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_resolve_allow_once(self, approval_flow, write_tool):
        """Resolving with allow_once returns correct decision."""
        async def resolve_after_delay():
            await asyncio.sleep(0.05)
            # Find the pending request and resolve it
            for req_id in list(approval_flow._pending.keys()):
                approval_flow.resolve_approval(req_id, "allow_once")
        
        asyncio.create_task(resolve_after_delay())
        
        decision = await approval_flow.request_approval(
            write_tool,
            {"path": "test.txt", "content": "hello"},
        )
        
        assert decision.allowed is True
        assert decision.user_response == "allow_once"
        assert decision.remembered is False
    
    @pytest.mark.asyncio
    async def test_resolve_allow_always(self, approval_flow, write_tool):
        """Resolving with allow_always sets remembered flag."""
        async def resolve_after_delay():
            await asyncio.sleep(0.05)
            for req_id in list(approval_flow._pending.keys()):
                approval_flow.resolve_approval(req_id, "allow_always")
        
        asyncio.create_task(resolve_after_delay())
        
        decision = await approval_flow.request_approval(
            write_tool,
            {"path": "test.txt", "content": "hello"},
        )
        
        assert decision.allowed is True
        assert decision.user_response == "allow_always"
        assert decision.remembered is True
    
    @pytest.mark.asyncio
    async def test_resolve_deny(self, approval_flow, write_tool):
        """Resolving with deny returns denied decision."""
        async def resolve_after_delay():
            await asyncio.sleep(0.05)
            for req_id in list(approval_flow._pending.keys()):
                approval_flow.resolve_approval(req_id, "deny")
        
        asyncio.create_task(resolve_after_delay())
        
        decision = await approval_flow.request_approval(
            write_tool,
            {"path": "test.txt", "content": "hello"},
        )
        
        assert decision.allowed is False
        assert decision.user_response == "deny"
        assert decision.remembered is False
    
    @pytest.mark.asyncio
    async def test_resolve_deny_always(self, approval_flow, write_tool):
        """Resolving with deny_always sets remembered flag."""
        async def resolve_after_delay():
            await asyncio.sleep(0.05)
            for req_id in list(approval_flow._pending.keys()):
                approval_flow.resolve_approval(req_id, "deny_always")
        
        asyncio.create_task(resolve_after_delay())
        
        decision = await approval_flow.request_approval(
            write_tool,
            {"path": "test.txt", "content": "hello"},
        )
        
        assert decision.allowed is False
        assert decision.user_response == "deny_always"
        assert decision.remembered is True
    
    @pytest.mark.asyncio
    async def test_timeout_returns_denied(self, approval_flow, write_tool):
        """Timeout without response returns denied decision."""
        decision = await approval_flow.request_approval(
            write_tool,
            {"path": "test.txt", "content": "hello"},
        )
        
        assert decision.allowed is False
        assert "timeout" in decision.reason
    
    @pytest.mark.asyncio
    async def test_resolve_unknown_request(self, approval_flow):
        """Resolving unknown request returns False."""
        result = approval_flow.resolve_approval("nonexistent_id", "allow_once")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cancel_all_clears_pending(self, approval_flow, write_tool):
        """cancel_all clears all pending requests."""
        # Start a request
        task = asyncio.create_task(
            approval_flow.request_approval(write_tool, {"path": "test.txt"})
        )
        await asyncio.sleep(0.05)
        
        assert approval_flow.pending_count == 1
        
        approval_flow.cancel_all()
        
        assert approval_flow.pending_count == 0
        
        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestSecurityEngineWithApproval:
    """Tests for SecurityEngine.check_and_approve()."""
    
    @pytest.fixture
    def security_config(self):
        return SecurityConfig(
            auto_allow=["file:read"],
            always_ask=["file:write", "shell:exec"],
            never_allow=["file:delete"],
        )
    
    @pytest.fixture
    def engine(self, security_config, kernel):
        engine = SecurityEngine(security_config, kernel)
        # Set short timeout for tests
        engine._approval_flow._timeout = 0.5
        return engine
    
    @pytest.mark.asyncio
    async def test_auto_allow_no_approval_needed(self, engine):
        """auto_allow capabilities don't trigger approval flow."""
        tool = ToolSpec(
            name="read_file",
            description="Read a file",
            parameters={},
            required_capabilities=frozenset([Capability.FILE_READ]),
        )
        
        decision = await engine.check_and_approve(tool, {"path": "test.txt"})
        
        assert decision.allowed is True
        assert engine.approval_flow.pending_count == 0
    
    @pytest.mark.asyncio
    async def test_never_allow_no_approval_offered(self, engine):
        """never_allow capabilities are denied without approval option."""
        tool = ToolSpec(
            name="delete_file",
            description="Delete a file",
            parameters={},
            required_capabilities=frozenset([Capability.FILE_DELETE]),
        )
        
        decision = await engine.check_and_approve(tool, {"path": "test.txt"})
        
        assert decision.allowed is False
        assert decision.requires_approval is False
        assert engine.approval_flow.pending_count == 0
    
    @pytest.mark.asyncio
    async def test_always_ask_triggers_approval(self, engine, write_tool):
        """always_ask capabilities trigger approval flow."""
        # Resolve approval immediately
        async def resolve():
            await asyncio.sleep(0.05)
            for req_id in list(engine.approval_flow._pending.keys()):
                engine.approval_flow.resolve_approval(req_id, "allow_once")
        
        asyncio.create_task(resolve())
        
        decision = await engine.check_and_approve(
            write_tool,
            {"path": "test.txt", "content": "hello"},
        )
        
        assert decision.allowed is True
    
    @pytest.mark.asyncio
    async def test_remembered_allow_skips_approval(self, engine, shell_tool):
        """Remembered allow_always decision skips approval."""
        # Remember a previous decision
        engine.remember_decision("execute", Capability.SHELL_EXEC, "allow_always")
        
        decision = await engine.check_and_approve(
            shell_tool,
            {"command": "ls -la"},
        )
        
        assert decision.allowed is True
        assert decision.remembered is True
        # No approval was requested
        assert engine.approval_flow.pending_count == 0
    
    @pytest.mark.asyncio
    async def test_approval_remembers_allow_always(self, engine, write_tool):
        """User choosing allow_always gets remembered."""
        async def resolve():
            await asyncio.sleep(0.05)
            for req_id in list(engine.approval_flow._pending.keys()):
                engine.approval_flow.resolve_approval(req_id, "allow_always")
        
        asyncio.create_task(resolve())
        
        # First call triggers approval
        await engine.check_and_approve(write_tool, {"path": "a.txt"})
        
        # Second call should be auto-allowed from memory
        decision2 = await engine.check_and_approve(write_tool, {"path": "b.txt"})
        
        assert decision2.allowed is True
        assert decision2.remembered is True
    
    @pytest.mark.asyncio
    async def test_approval_remembers_deny_always(self, engine, write_tool):
        """User choosing deny_always gets remembered."""
        async def resolve():
            await asyncio.sleep(0.05)
            for req_id in list(engine.approval_flow._pending.keys()):
                engine.approval_flow.resolve_approval(req_id, "deny_always")
        
        asyncio.create_task(resolve())
        
        # First call triggers approval, user denies always
        await engine.check_and_approve(write_tool, {"path": "a.txt"})
        
        # Second call should be auto-denied from memory
        decision2 = await engine.check_and_approve(write_tool, {"path": "b.txt"})
        
        assert decision2.allowed is False
        assert decision2.remembered is True


class TestApprovalEventData:
    """Tests for approval event data structure."""
    
    @pytest.mark.asyncio
    async def test_approval_event_contains_details(self, kernel):
        """Approval event contains tool details for display."""
        flow = ApprovalFlow(kernel, timeout=0.5)
        
        tool = ToolSpec(
            name="dangerous_tool",
            description="Does something dangerous",
            parameters={"type": "object", "properties": {"target": {"type": "string"}}},
            required_capabilities=frozenset([Capability.SHELL_EXEC]),
        )
        
        received_events = []
        
        async def capture_event(event):
            received_events.append(event)
            # Immediately deny to unblock
            flow.resolve_approval(event.data["request_id"], "deny")
        
        kernel.on("security:approval", capture_event)
        
        await flow.request_approval(tool, {"target": "/etc/passwd"})
        
        assert len(received_events) == 1
        event = received_events[0]
        
        assert event.data["tool_name"] == "dangerous_tool"
        assert event.data["tool_description"] == "Does something dangerous"
        assert event.data["arguments"] == {"target": "/etc/passwd"}
        assert "shell:exec" in event.data["capabilities"]