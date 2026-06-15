import logging
from typing import Optional

from src.config import get_settings
from src.services.telegram.bot import TelegramBot

logger = logging.getLogger(__name__)


def make_telegram_service(
    opensearch_client,
    embeddings_client,
    llm_client,
    agentic_service=None,
    cache_client=None,
) -> Optional[TelegramBot]:
    """Create the Telegram bot if enabled in settings.

    Args:
        opensearch_client: OpenSearch client (search).
        embeddings_client: Embeddings service (query vectors).
        llm_client: LLM client (answer generation / fallback RAG).
        agentic_service: Optional agentic RAG service (preferred for questions).
        cache_client: Optional Redis cache client.

    Returns:
        A configured ``TelegramBot``, or ``None`` if disabled/misconfigured.
    """
    settings = get_settings()
    tg = settings.telegram

    if not tg.enabled:
        logger.info("Telegram bot is disabled (set TELEGRAM__ENABLED=true to enable)")
        return None

    if not tg.bot_token:
        logger.warning("Telegram bot enabled but TELEGRAM__BOT_TOKEN is not configured")
        return None

    bot = TelegramBot(
        bot_token=tg.bot_token,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        llm_client=llm_client,
        agentic_service=agentic_service,
        cache_client=cache_client,
        allowed_user_ids=tg.allowed_ids,
        use_agentic=tg.use_agentic,
        default_top_k=tg.default_top_k,
        default_use_hybrid=tg.default_use_hybrid,
        max_message_length=tg.max_message_length,
    )

    logger.info("Telegram bot created successfully")
    return bot
