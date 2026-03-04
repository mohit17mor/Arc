"""Tests for TelegramPlatform."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from arc.platforms.telegram.app import TelegramPlatform, _split_text


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
    yield "before error "
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Unit tests — no Telegram dependency required
# ---------------------------------------------------------------------------

class TestTelegramPlatformBasic:
    """Test basic TelegramPlatform properties (no network)."""

    def test_name(self):
        tp = TelegramPlatform(token="test-token")
        assert tp.name == "telegram"

    def test_default_allows_all(self):
        tp = TelegramPlatform(token="test-token")
        assert tp._is_allowed(12345) is True

    def test_allowlist_blocks_unknown(self):
        tp = TelegramPlatform(token="test-token", allowed_chat_ids={"111"})
        assert tp._is_allowed(111) is True
        assert tp._is_allowed(222) is False

    def test_allowlist_string_comparison(self):
        """Chat IDs are compared as strings."""
        tp = TelegramPlatform(token="test-token", allowed_chat_ids={"999"})
        assert tp._is_allowed(999) is True
        assert tp._is_allowed(1000) is False

    def test_per_chat_lock(self):
        tp = TelegramPlatform(token="test-token")
        lock1 = tp._get_lock(100)
        lock2 = tp._get_lock(100)
        lock3 = tp._get_lock(200)
        assert lock1 is lock2  # same chat_id → same lock
        assert lock1 is not lock3  # different chat_id → different lock


class TestSplitText:
    """Test the _split_text helper."""

    def test_short_text_no_split(self):
        assert _split_text("hello", 4096) == ["hello"]

    def test_exact_limit(self):
        text = "x" * 4096
        assert _split_text(text, 4096) == [text]

    def test_split_at_paragraph(self):
        text = "A" * 3000 + "\n\n" + "B" * 3000
        chunks = _split_text(text, 4096)
        assert len(chunks) == 2
        assert chunks[0].endswith("\n\n")
        assert chunks[1] == "B" * 3000

    def test_split_at_newline(self):
        text = "A" * 3000 + "\n" + "B" * 3000
        chunks = _split_text(text, 4096)
        assert len(chunks) == 2

    def test_split_at_space(self):
        text = "word " * 1000  # 5000 chars
        chunks = _split_text(text, 4096)
        assert len(chunks) >= 2
        assert "".join(chunks) == text

    def test_very_long_no_breaks(self):
        text = "X" * 10000
        chunks = _split_text(text, 4096)
        assert len(chunks) == 3
        assert "".join(chunks) == text


# ---------------------------------------------------------------------------
# Message handling tests (mocked Telegram objects)
# ---------------------------------------------------------------------------

def _make_update(chat_id: int = 123, text: str = "hello", first_name: str = "Test"):
    """Create a mock Telegram Update object."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user.first_name = first_name
    update.message.text = text
    update.message.reply_text = AsyncMock(return_value=MagicMock())
    update.effective_chat.send_action = AsyncMock()

    # The reply message (placeholder for streaming)
    reply = MagicMock()
    reply.edit_text = AsyncMock()
    reply.get_bot.return_value.send_message = AsyncMock()
    update.message.reply_text.return_value = reply

    return update


def _make_context():
    """Create a mock telegram.ext.CallbackContext."""
    ctx = MagicMock()
    ctx.chat_data = {}
    return ctx


class TestCommandHandlers:
    """Test slash command handlers."""

    @pytest.mark.asyncio
    async def test_start_allowed(self):
        tp = TelegramPlatform(token="t", agent_name="TestBot")
        update = _make_update(chat_id=123)
        ctx = _make_context()

        await tp._cmd_start(update, ctx)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "TestBot" in text

    @pytest.mark.asyncio
    async def test_start_blocked(self):
        tp = TelegramPlatform(token="t", allowed_chat_ids={"999"})
        update = _make_update(chat_id=123)
        ctx = _make_context()

        await tp._cmd_start(update, ctx)
        text = update.message.reply_text.call_args[0][0]
        assert "not authorized" in text.lower()

    @pytest.mark.asyncio
    async def test_clear_sets_flag(self):
        tp = TelegramPlatform(token="t")
        update = _make_update(chat_id=123)
        ctx = _make_context()

        await tp._cmd_clear(update, ctx)
        assert ctx.chat_data.get("clear_requested") is True

    @pytest.mark.asyncio
    async def test_status_shows_chat_id(self):
        tp = TelegramPlatform(token="t")
        update = _make_update(chat_id=42)
        ctx = _make_context()

        await tp._cmd_status(update, ctx)
        text = update.message.reply_text.call_args[0][0]
        assert "42" in text

    @pytest.mark.asyncio
    async def test_help_shows_commands(self):
        tp = TelegramPlatform(token="t")
        update = _make_update(chat_id=123)
        ctx = _make_context()

        await tp._cmd_help(update, ctx)
        text = update.message.reply_text.call_args[0][0]
        assert "/start" in text
        assert "/clear" in text


class TestMessageHandling:
    """Test _on_message and _process_message."""

    @pytest.mark.asyncio
    async def test_unauthorized_message_rejected(self):
        tp = TelegramPlatform(token="t", allowed_chat_ids={"999"})
        update = _make_update(chat_id=123, text="hi")
        ctx = _make_context()

        await tp._on_message(update, ctx)
        text = update.message.reply_text.call_args[0][0]
        assert "unauthorized" in text.lower()

    @pytest.mark.asyncio
    async def test_message_no_handler(self):
        tp = TelegramPlatform(token="t")
        tp._handler = None
        update = _make_update(chat_id=123, text="hi")
        ctx = _make_context()

        await tp._on_message(update, ctx)
        text = update.message.reply_text.call_args[0][0]
        assert "not ready" in text.lower()

    @pytest.mark.asyncio
    async def test_message_empty_text(self):
        tp = TelegramPlatform(token="t")
        tp._handler = echo_handler
        update = _make_update(chat_id=123, text="")
        update.message.text = ""
        ctx = _make_context()

        # Should not crash, just return
        await tp._on_message(update, ctx)

    @pytest.mark.asyncio
    async def test_process_message_echo(self):
        tp = TelegramPlatform(token="t")
        tp._handler = echo_handler
        update = _make_update(chat_id=123, text="world")
        ctx = _make_context()

        await tp._process_message(update, ctx, "world")

        # Check that placeholder was sent then edited with final response
        reply = update.message.reply_text.return_value
        # Final edit should contain "echo: world"
        last_edit = reply.edit_text.call_args_list[-1][0][0]
        assert "echo: world" in last_edit

    @pytest.mark.asyncio
    async def test_process_message_error_handling(self):
        tp = TelegramPlatform(token="t")
        tp._handler = error_handler
        update = _make_update(chat_id=123, text="oops")
        ctx = _make_context()

        await tp._process_message(update, ctx, "oops")

        reply = update.message.reply_text.return_value
        # Should send error message
        last_edit = reply.edit_text.call_args_list[-1][0][0]
        assert "error" in last_edit.lower()

    @pytest.mark.asyncio
    async def test_concurrent_message_rejected(self):
        """Second message while first is processing should get 'please wait'."""
        tp = TelegramPlatform(token="t")

        # Create a slow handler
        async def slow_handler(msg: str):
            await asyncio.sleep(5)
            yield "done"

        tp._handler = slow_handler

        update1 = _make_update(chat_id=100, text="first")
        update2 = _make_update(chat_id=100, text="second")
        ctx = _make_context()

        # Acquire the lock manually to simulate processing
        lock = tp._get_lock(100)
        await lock.acquire()

        # Second message with same chat_id should be rejected
        await tp._on_message(update2, ctx)
        text = update2.message.reply_text.call_args[0][0]
        assert "still working" in text.lower() or "please wait" in text.lower()

        lock.release()


class TestSafeEdit:
    """Test _safe_edit message editing."""

    @pytest.mark.asyncio
    async def test_truncates_long_text(self):
        tp = TelegramPlatform(token="t")
        msg = MagicMock()
        msg.edit_text = AsyncMock()

        long_text = "X" * 5000
        await tp._safe_edit(msg, long_text)

        called_text = msg.edit_text.call_args[0][0]
        assert len(called_text) <= 4096
        assert "truncated" in called_text

    @pytest.mark.asyncio
    async def test_ignores_not_modified_error(self):
        tp = TelegramPlatform(token="t")
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=Exception("Message is not modified"))

        # Should not raise
        await tp._safe_edit(msg, "same text")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestTelegramConfig:
    """Test TelegramConfig changes."""

    def test_allowed_users_default_empty(self):
        from arc.core.config import TelegramConfig
        cfg = TelegramConfig()
        assert cfg.allowed_users == []

    def test_platform_configured_with_token_only(self):
        from arc.core.config import TelegramConfig
        cfg = TelegramConfig(token="abc")
        assert cfg.platform_configured is True
        assert cfg.configured is False  # no chat_id

    def test_configured_with_both(self):
        from arc.core.config import TelegramConfig
        cfg = TelegramConfig(token="abc", chat_id="123")
        assert cfg.configured is True
        assert cfg.platform_configured is True

    def test_allowed_users_populated(self):
        from arc.core.config import TelegramConfig
        cfg = TelegramConfig(
            token="abc", chat_id="123", allowed_users=["111", "222"]
        )
        assert cfg.allowed_users == ["111", "222"]
