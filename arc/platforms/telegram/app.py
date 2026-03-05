"""
TelegramPlatform — full bidirectional chat via Telegram Bot API.

Unlike TelegramChannel (outbound notifications only), this is a full
Platform implementation:  it receives user messages from Telegram,
passes them to the agent, and streams responses back.

Architecture:
    - Uses python-telegram-bot (async) for long polling
    - Each incoming message is processed sequentially per chat_id
    - Responses are chunked (Telegram has a 4096-char message limit)
    - Supports /start, /clear, /status slash commands
    - Allowlist-based security: only configured chat_ids can interact

This follows the exact same Platform(ABC) contract as CLIPlatform and
VirtualPlatform, making it trivial to add Discord, WhatsApp, etc. later
by implementing the same interface.

Usage:
    platform = TelegramPlatform(
        token="BOT_TOKEN",
        allowed_chat_ids={"123456"},
    )
    await platform.run(agent.run)

Config (~/.arc/config.toml):
    [telegram]
    token = "BOT_TOKEN"
    chat_id = "123456"          # for notifications (existing)
    allowed_users = ["123456"]  # who can chat with the bot
"""

from __future__ import annotations

import asyncio
import html
import logging
from typing import Any

from arc.platforms.base import Platform, MessageHandler

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MESSAGE_LENGTH = 4096


class TelegramPlatform(Platform):
    """
    Bidirectional Telegram bot platform.

    Receives user messages via long polling, runs them through
    the agent handler, and streams responses back by editing
    a placeholder message in real-time.
    """

    def __init__(
        self,
        token: str,
        allowed_chat_ids: set[str] | None = None,
        agent_name: str = "Arc",
    ) -> None:
        self._token = token
        self._allowed_chat_ids = allowed_chat_ids  # None = allow all
        self._agent_name = agent_name
        self._running = False
        self._handler: MessageHandler | None = None
        self._application: Any = None  # telegram.ext.Application
        self._processing_lock: dict[int, asyncio.Lock] = {}  # per-chat locks

    @property
    def name(self) -> str:
        return "telegram"

    async def run(self, handler: MessageHandler) -> None:
        """
        Start the Telegram bot and process messages.

        Blocks until stop() is called or the bot is interrupted.
        """
        try:
            from telegram import Update, BotCommand
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler as TGMessageHandler,
                filters,
            )
        except ImportError:
            raise RuntimeError(
                "python-telegram-bot is required for Telegram platform. "
                "Install it with: pip install 'python-telegram-bot>=21'"
            )

        self._handler = handler
        self._running = True

        # Build application
        self._application = (
            Application.builder()
            .token(self._token)
            .build()
        )

        # Register handlers
        self._application.add_handler(CommandHandler("start", self._cmd_start))
        self._application.add_handler(CommandHandler("clear", self._cmd_clear))
        self._application.add_handler(CommandHandler("status", self._cmd_status))
        self._application.add_handler(CommandHandler("help", self._cmd_help))
        self._application.add_handler(
            TGMessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        # Set bot commands (shows in Telegram's menu)
        async def _post_init(app: Any) -> None:
            await app.bot.set_my_commands([
                BotCommand("start", "Start a conversation"),
                BotCommand("clear", "Reset conversation context"),
                BotCommand("status", "Show bot status"),
                BotCommand("help", "Show available commands"),
            ])

        self._application.post_init = _post_init

        logger.info("Telegram platform starting — polling for messages")

        # Run polling (blocks until stopped)
        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message"],
        )

        # Keep running until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        logger.info("Telegram platform stopping")

    def _is_allowed(self, chat_id: int) -> bool:
        """Check if a chat_id is in the allowlist."""
        if self._allowed_chat_ids is None:
            return True  # no allowlist = allow everyone
        return str(chat_id) in self._allowed_chat_ids

    def _get_lock(self, chat_id: int) -> asyncio.Lock:
        """Get or create a per-chat processing lock."""
        if chat_id not in self._processing_lock:
            self._processing_lock[chat_id] = asyncio.Lock()
        return self._processing_lock[chat_id]

    # ── Command handlers ─────────────────────────────────────

    async def _cmd_start(self, update: Any, context: Any) -> None:
        """Handle /start command."""
        if not self._is_allowed(update.effective_chat.id):
            await update.message.reply_text(
                "⛔ You are not authorized to use this bot.\n"
                "Ask the owner to add your chat_id to the allowed_users list."
            )
            return

        await update.message.reply_text(
            f"👋 Hi! I'm {self._agent_name}, your AI assistant.\n\n"
            "Just send me a message and I'll help you out.\n\n"
            "Commands:\n"
            "/clear — Reset conversation\n"
            "/status — Show bot status\n"
            "/help — Show this message"
        )

    async def _cmd_clear(self, update: Any, context: Any) -> None:
        """Handle /clear command — signals conversation reset."""
        if not self._is_allowed(update.effective_chat.id):
            return
        # The actual memory clearing happens in the CLI runner;
        # here we just acknowledge. The platform doesn't own memory.
        await update.message.reply_text(
            "🔄 Conversation context will be refreshed on next message."
        )
        # Store a flag so the next message knows to reset
        context.chat_data["clear_requested"] = True

    async def _cmd_status(self, update: Any, context: Any) -> None:
        """Handle /status command."""
        if not self._is_allowed(update.effective_chat.id):
            return
        await update.message.reply_text(
            f"🟢 {self._agent_name} is online and ready.\n"
            f"Chat ID: `{update.effective_chat.id}`",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update: Any, context: Any) -> None:
        """Handle /help command."""
        if not self._is_allowed(update.effective_chat.id):
            return
        await update.message.reply_text(
            f"🤖 *{self._agent_name}*\n\n"
            "Send me any message and I'll respond using AI.\n\n"
            "*Commands:*\n"
            "/start — Welcome message\n"
            "/clear — Reset conversation context\n"
            "/status — Check if bot is online\n"
            "/help — Show this message\n\n"
            "_Tip: I can browse the web, search for products, "
            "manage files, run commands, and more!_",
            parse_mode="Markdown",
        )

    # ── Message handler ──────────────────────────────────────

    async def _on_message(self, update: Any, context: Any) -> None:
        """Handle incoming text messages."""
        chat_id = update.effective_chat.id

        if not self._is_allowed(chat_id):
            await update.message.reply_text(
                "⛔ Unauthorized. Your chat_id is not in the allowed list."
            )
            return

        if not self._handler:
            await update.message.reply_text("⚠️ Bot is not ready yet.")
            return

        user_text = update.message.text
        if not user_text:
            return

        user_name = update.effective_user.first_name or "User"
        logger.info(
            "Telegram message from %s (chat_id=%d): %s",
            user_name, chat_id, user_text[:100],
        )

        # Process one message at a time per chat
        lock = self._get_lock(chat_id)
        if lock.locked():
            await update.message.reply_text(
                "⏳ I'm still working on your previous message. Please wait..."
            )
            return

        async with lock:
            await self._process_message(update, context, user_text)

    async def _process_message(
        self, update: Any, context: Any, user_text: str,
    ) -> None:
        """Process a single message: collect full response, then send."""
        chat_id = update.effective_chat.id

        # Send "typing" indicator (re-sent every ~4s while processing)
        typing_task = asyncio.create_task(
            self._keep_typing(update.effective_chat)
        )

        # Collect full response from agent (no mid-stream edits)
        collected = ""
        try:
            async for chunk in self._handler(user_text):
                collected += chunk
        except Exception as e:
            logger.error("Telegram agent error: %s", e, exc_info=True)
            collected = f"❌ An error occurred: {e}"
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # Send the complete response as a new message (preserves ordering)
        if collected.strip():
            await self._send_response(chat_id, collected)
        else:
            await self._send_response(
                chat_id, "🤔 I didn't have anything to say."
            )

    async def _keep_typing(self, chat: Any) -> None:
        """Re-send typing indicator every 4 seconds until cancelled."""
        try:
            while True:
                await chat.send_action("typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    async def _send_response(self, chat_id: int, text: str) -> None:
        """Send the full response, splitting into multiple messages if needed."""
        bot = self._application.bot
        chunks = _split_text(text, _MAX_MESSAGE_LENGTH)

        for chunk in chunks:
            try:
                await bot.send_message(chat_id=chat_id, text=chunk)
            except Exception as e:
                # Markdown-like chars can break — retry as plain
                if "parse" in str(e).lower() or "400" in str(e):
                    try:
                        await bot.send_message(chat_id=chat_id, text=chunk)
                    except Exception as e2:
                        logger.warning("Failed to send message chunk: %s", e2)
                else:
                    logger.warning("Failed to send message chunk: %s", e)


def _split_text(text: str, max_length: int) -> list[str]:
    """
    Split text into chunks respecting the max length.

    Tries to split at paragraph boundaries, then sentence boundaries,
    then at the max length.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try to find a good split point
        split_at = max_length

        # Try paragraph break
        para_break = remaining.rfind("\n\n", 0, max_length)
        if para_break > max_length // 2:
            split_at = para_break + 2
        else:
            # Try line break
            line_break = remaining.rfind("\n", 0, max_length)
            if line_break > max_length // 2:
                split_at = line_break + 1
            else:
                # Try space
                space = remaining.rfind(" ", 0, max_length)
                if space > max_length // 2:
                    split_at = space + 1

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]

    return chunks
