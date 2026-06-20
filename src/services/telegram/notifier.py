"""Standalone, stateless Telegram sender for push notifications.

Unlike :class:`~src.services.telegram.bot.TelegramBot` (which runs a long-lived
polling application inside the FastAPI process), this module creates a fresh
``telegram.Bot``, sends one or more messages, and returns. It holds no state and
acquires no leader lock, so it is safe to call from a separate process such as an
Airflow worker after the ingestion pipeline finishes.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

TELEGRAM_MAX = 4096


def split_message(text: str, limit: int = TELEGRAM_MAX) -> List[str]:
    """Split text into chunks under Telegram's per-message limit, on line breaks.

    A single line longer than ``limit`` is hard-split so no chunk ever exceeds it.
    """
    limit = min(limit, TELEGRAM_MAX)
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    current = ""
    for line in text.split("\n"):
        # Hard-split a line that alone exceeds the limit.
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            if current:
                parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


async def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    parse_mode: str = "Markdown",
    disable_web_page_preview: bool = True,
    max_message_length: int = TELEGRAM_MAX,
) -> int:
    """Send a (possibly long) message to a chat via a one-off ``telegram.Bot``.

    The message is split to respect Telegram's length limit. If a part fails to
    send with ``parse_mode`` (e.g. unbalanced Markdown), it is retried as plain
    text so a formatting glitch never drops the notification.

    :returns: Number of message parts successfully sent.
    """
    if not bot_token or not chat_id:
        raise ValueError("send_telegram_message requires both bot_token and chat_id")

    # Imported lazily so importing this module never hard-requires the dependency.
    from telegram import Bot

    bot = Bot(token=bot_token)
    parts = split_message(text, limit=max_message_length)
    sent = 0

    async with bot:
        for part in parts:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=part,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_web_page_preview,
                )
            except Exception as e:
                logger.warning(f"Markdown send failed, retrying as plain text: {e}")
                await bot.send_message(
                    chat_id=chat_id,
                    text=part,
                    disable_web_page_preview=disable_web_page_preview,
                )
            sent += 1

    logger.info(f"Sent Telegram notification to chat {chat_id} in {sent} part(s)")
    return sent
