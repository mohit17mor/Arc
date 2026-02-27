"""Tests for VirtualPlatform."""

from __future__ import annotations

import asyncio
import pytest

from arc.platforms.virtual.app import VirtualPlatform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def echo_handler(message: str):
    """Simple handler that echoes back the message."""
    yield f"echo: {message}"


async def multi_chunk_handler(message: str):
    """Handler that yields multiple chunks."""
    for word in message.split():
        yield word + " "


async def error_handler(message: str):
    """Handler that raises mid-stream."""
    yield "before error"
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_virtual_platform_name():
    vp = VirtualPlatform(name="test-vp")
    assert vp.name == "test-vp"


@pytest.mark.asyncio
async def test_virtual_platform_default_name():
    vp = VirtualPlatform()
    assert vp.name == "virtual"


@pytest.mark.asyncio
async def test_send_message_returns_response():
    vp = VirtualPlatform(name="echo")

    task = asyncio.create_task(vp.run(echo_handler))
    response = await vp.send_message("hello")
    await vp.stop()
    await task

    assert response == "echo: hello"


@pytest.mark.asyncio
async def test_send_message_multi_chunk():
    vp = VirtualPlatform(name="chunks")

    task = asyncio.create_task(vp.run(multi_chunk_handler))
    response = await vp.send_message("one two three")
    await vp.stop()
    await task

    assert response == "one two three "


@pytest.mark.asyncio
async def test_send_multiple_messages_sequentially():
    vp = VirtualPlatform(name="seq")

    task = asyncio.create_task(vp.run(echo_handler))

    r1 = await vp.send_message("first")
    r2 = await vp.send_message("second")
    r3 = await vp.send_message("third")

    await vp.stop()
    await task

    assert r1 == "echo: first"
    assert r2 == "echo: second"
    assert r3 == "echo: third"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_error_captured_in_response():
    """Errors in the handler are captured as [Error: ...] — platform keeps running."""
    vp = VirtualPlatform(name="err")

    task = asyncio.create_task(vp.run(error_handler))
    response = await vp.send_message("trigger error")
    await vp.stop()
    await task

    # Platform should survive and return something
    assert "[Error:" in response or "before error" in response


# ---------------------------------------------------------------------------
# Stop behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_terminates_run():
    vp = VirtualPlatform(name="stopper")
    task = asyncio.create_task(vp.run(echo_handler))

    await vp.stop()
    await asyncio.wait_for(task, timeout=2.0)

    assert not vp.is_running


@pytest.mark.asyncio
async def test_is_running_reflects_state():
    vp = VirtualPlatform(name="state")
    assert not vp.is_running

    task = asyncio.create_task(vp.run(echo_handler))
    await asyncio.sleep(0)  # yield so run() starts
    assert vp.is_running

    await vp.stop()
    await task
    assert not vp.is_running


# ---------------------------------------------------------------------------
# get_last_output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_last_output():
    vp = VirtualPlatform(name="last")

    task = asyncio.create_task(vp.run(echo_handler))
    await vp.send_message("foo")
    await vp.stop()
    await task

    assert vp.get_last_output() == "echo: foo"
