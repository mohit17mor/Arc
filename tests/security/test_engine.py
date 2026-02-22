"""Tests for the Security Engine."""

import pytest
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig, SecurityConfig
from arc.core.types import Capability, ToolSpec
from arc.security.engine import SecurityEngine


@pytest.fixture
def kernel():
    return Kernel(config=ArcConfig())


@pytest.fixture
def security_config():
    return SecurityConfig(
        auto_allow=["file:read"],
        always_ask=["file:write", "shell:exec"],
        never_allow=["file:delete"],
    )


@pytest.fixture
def engine(security_config, kernel):
    return SecurityEngine(security_config, kernel)


@pytest.fixture
def read_tool():
    return ToolSpec(
        name="read_file",
        description="Read a file",
        parameters={},
        required_capabilities=frozenset([Capability.FILE_READ]),
    )


@pytest.fixture
def write_tool():
    return ToolSpec(
        name="write_file",
        description="Write a file",
        parameters={},
        required_capabilities=frozenset([Capability.FILE_WRITE]),
    )


@pytest.fixture
def delete_tool():
    return ToolSpec(
        name="delete_file",
        description="Delete a file",
        parameters={},
        required_capabilities=frozenset([Capability.FILE_DELETE]),
    )


@pytest.fixture
def shell_tool():
    return ToolSpec(
        name="execute",
        description="Run command",
        parameters={},
        required_capabilities=frozenset([Capability.SHELL_EXEC]),
    )


@pytest.mark.asyncio
async def test_auto_allow(engine, read_tool):
    """auto_allow capabilities are allowed without asking."""
    decision = await engine.check_tool(read_tool, {"path": "test.txt"})
    assert decision.allowed is True
    assert "auto_allow" in decision.reason or "all_allowed" in decision.reason


@pytest.mark.asyncio
async def test_never_allow(engine, delete_tool):
    """never_allow capabilities are always blocked."""
    decision = await engine.check_tool(delete_tool, {"path": "test.txt"})
    assert decision.allowed is False
    assert "never_allow" in decision.reason


@pytest.mark.asyncio
async def test_always_ask(engine, write_tool):
    """always_ask capabilities require approval."""
    decision = await engine.check_tool(write_tool, {"path": "test.txt"})
    assert decision.allowed is False
    assert decision.requires_approval is True
    assert "always_ask" in decision.reason


@pytest.mark.asyncio
async def test_remember_allow(engine, shell_tool):
    """Remembered allow decisions are respected."""
    # First check — requires approval
    decision1 = await engine.check_tool(shell_tool, {"command": "ls"})
    assert decision1.requires_approval is True

    # Remember the decision
    engine.remember_decision("execute", Capability.SHELL_EXEC, "allow_always")

    # Second check — should be allowed
    decision2 = await engine.check_tool(shell_tool, {"command": "ls"})
    assert decision2.allowed is True
    assert decision2.remembered is True


@pytest.mark.asyncio
async def test_remember_deny(engine, shell_tool):
    """Remembered deny decisions are respected."""
    engine.remember_decision("execute", Capability.SHELL_EXEC, "deny_always")

    decision = await engine.check_tool(shell_tool, {"command": "rm -rf /"})
    assert decision.allowed is False
    assert decision.remembered is True


@pytest.mark.asyncio
async def test_clear_remembered(engine, shell_tool):
    """clear_remembered removes all remembered decisions."""
    engine.remember_decision("execute", Capability.SHELL_EXEC, "allow_always")
    engine.clear_remembered()

    decision = await engine.check_tool(shell_tool, {"command": "ls"})
    assert decision.requires_approval is True  # Back to asking


@pytest.mark.asyncio
async def test_multiple_capabilities():
    """Tool requiring multiple capabilities checks all of them."""
    config = SecurityConfig(
        auto_allow=["file:read"],
        always_ask=["network:http"],
        never_allow=[],
    )
    kernel = Kernel(config=ArcConfig())
    engine = SecurityEngine(config, kernel)

    # Tool that needs both file:read and network:http
    tool = ToolSpec(
        name="upload_file",
        description="Upload a file",
        parameters={},
        required_capabilities=frozenset([
            Capability.FILE_READ,
            Capability.NETWORK_HTTP,
        ]),
    )

    # Should require approval because network:http is in always_ask
    decision = await engine.check_tool(tool, {})
    assert decision.requires_approval is True