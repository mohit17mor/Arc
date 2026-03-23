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

    @pytest.mark.asyncio
    async def test_stop_marks_platform_not_running(self):
        tp = TelegramPlatform(token="test-token")
        tp._running = True

        await tp.stop()

        assert tp._running is False


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
        # Mock _application.bot for send_message
        tp._application = MagicMock()
        tp._application.bot.send_message = AsyncMock()
        update = _make_update(chat_id=123, text="world")
        ctx = _make_context()

        await tp._process_message(update, ctx, "world")

        # Should send response as a NEW message (not edit placeholder)
        tp._application.bot.send_message.assert_called()
        sent_text = tp._application.bot.send_message.call_args[1]["text"]
        assert "echo: world" in sent_text

    @pytest.mark.asyncio
    async def test_process_message_error_handling(self):
        tp = TelegramPlatform(token="t")
        tp._handler = error_handler
        tp._application = MagicMock()
        tp._application.bot.send_message = AsyncMock()
        update = _make_update(chat_id=123, text="oops")
        ctx = _make_context()

        await tp._process_message(update, ctx, "oops")

        # Should send error message
        tp._application.bot.send_message.assert_called()
        sent_text = tp._application.bot.send_message.call_args[1]["text"]
        assert "error" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_process_message_no_placeholder_edit(self):
        """Response should be a new message, not an edit of a placeholder."""
        tp = TelegramPlatform(token="t")
        tp._handler = echo_handler
        tp._application = MagicMock()
        tp._application.bot.send_message = AsyncMock()
        update = _make_update(chat_id=123, text="test")
        ctx = _make_context()

        await tp._process_message(update, ctx, "test")

        # No reply_text (no "Thinking..." placeholder)
        update.message.reply_text.assert_not_called()
        # Response sent via bot.send_message
        tp._application.bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_typing_indicator_sent(self):
        """Typing indicator should be sent for longer-running handlers."""
        tp = TelegramPlatform(token="t")

        async def slow_echo(msg: str):
            await asyncio.sleep(0.1)  # give typing task time to fire
            yield f"echo: {msg}"

        tp._handler = slow_echo
        tp._application = MagicMock()
        tp._application.bot.send_message = AsyncMock()
        update = _make_update(chat_id=123, text="test")
        ctx = _make_context()

        await tp._process_message(update, ctx, "test")

        # Typing action should have been sent at least once
        update.effective_chat.send_action.assert_called()

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

    @pytest.mark.asyncio
    async def test_process_message_appends_context_window_footer(self):
        tp = TelegramPlatform(token="t")
        tp._handler = echo_handler
        tp._application = MagicMock()
        tp._application.bot.send_message = AsyncMock()
        tp.set_cost_tracker(
            type(
                "Tracker",
                (),
                {
                    "turn_peak_input": 1200,
                    "turn_output_tokens": 80,
                    "context_window": 4000,
                },
            )()
        )
        update = _make_update(chat_id=123, text="world")
        ctx = _make_context()

        await tp._process_message(update, ctx, "world")

        sent_text = tp._application.bot.send_message.call_args[1]["text"]
        assert "1,200 / 4,000 ctx" in sent_text
        assert "80 out" in sent_text

    @pytest.mark.asyncio
    async def test_process_message_appends_in_out_footer_without_context_window(self):
        tp = TelegramPlatform(token="t")
        tp._handler = echo_handler
        tp._application = MagicMock()
        tp._application.bot.send_message = AsyncMock()
        tp.set_cost_tracker(
            type(
                "Tracker",
                (),
                {
                    "turn_peak_input": 0,
                    "turn_input_tokens": 50,
                    "turn_output_tokens": 25,
                    "context_window": 0,
                },
            )()
        )
        update = _make_update(chat_id=123, text="world")
        ctx = _make_context()

        await tp._process_message(update, ctx, "world")

        sent_text = tp._application.bot.send_message.call_args[1]["text"]
        assert "75 tokens" in sent_text
        assert "50 in" in sent_text
        assert "25 out" in sent_text

    @pytest.mark.asyncio
    async def test_send_response_retries_parse_errors_and_continues(self):
        tp = TelegramPlatform(token="t")
        tp._application = MagicMock()
        bot = tp._application.bot
        bot.send_message = AsyncMock(side_effect=[RuntimeError("400 parse error"), None])

        await tp._send_response(123, "hello")

        assert bot.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_send_response_logs_when_retry_also_fails(self):
        tp = TelegramPlatform(token="t")
        tp._application = MagicMock()
        bot = tp._application.bot
        bot.send_message = AsyncMock(side_effect=[RuntimeError("400 parse error"), RuntimeError("still bad")])

        with patch("arc.platforms.telegram.app.logger.warning") as warning:
            await tp._send_response(123, "hello")

        warning.assert_called_once()
        assert "Failed to send message chunk" in warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_response_splits_large_messages(self):
        tp = TelegramPlatform(token="t")
        tp._application = MagicMock()
        tp._application.bot.send_message = AsyncMock()
        long_text = ("A" * 3000) + "\n\n" + ("B" * 3000)

        await tp._send_response(123, long_text)

        assert tp._application.bot.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_keep_typing_exits_cleanly_on_cancellation(self):
        tp = TelegramPlatform(token="t")
        chat = MagicMock()
        chat.send_action = AsyncMock()

        task = asyncio.create_task(tp._keep_typing(chat))
        await asyncio.sleep(0)
        task.cancel()
        await task

        chat.send_action.assert_called()


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
