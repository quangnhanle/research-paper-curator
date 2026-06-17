import asyncio
import logging
import os
from contextlib import asynccontextmanager

import redis
import uvicorn
from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.routers import hybrid_search, papers, ping
from src.routers.agentic_ask import router as agentic_router
from src.routers.ask import ask_router, stream_router
from src.services.agents.factory import make_agentic_rag_service
from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.llm.factory import make_llm_client
from src.services.opensearch.factory import make_opensearch_client
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.telegram.factory import make_telegram_service

_TELEGRAM_LOCK_KEY = "telegram:bot:leader"
_TELEGRAM_LOCK_TTL = 30  # seconds; heartbeat renews every TTL/3


async def _run_leader_heartbeat(r: redis.Redis, stop: asyncio.Event) -> None:
    """Renew the Telegram leader lock so it doesn't expire while the bot runs."""
    interval = _TELEGRAM_LOCK_TTL // 3
    while not stop.is_set():
        try:
            r.expire(_TELEGRAM_LOCK_KEY, _TELEGRAM_LOCK_TTL)
        except Exception:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(stop.wait()), timeout=interval)
        except asyncio.TimeoutError:
            pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize search service
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    # Verify OpenSearch connectivity and create index if needed
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Setup hybrid index (supports all search types)
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid index created")
        else:
            logger.info("Hybrid index already exists")

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize other services (kept for future endpoints and notebook demos)
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.llm_client = make_llm_client()
    app.state.langfuse_tracer = make_langfuse_tracer()
    app.state.cache_client = make_cache_client(settings)

    # Agentic RAG (LangGraph) service - reuses the clients above
    app.state.agentic_rag_service = make_agentic_rag_service(
        opensearch_client=opensearch_client,
        embeddings_client=app.state.embeddings_service,
        llm_client=app.state.llm_client,
    )
    logger.info("Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, LLM, Agentic RAG")

    # Telegram bot (optional - only if TELEGRAM__ENABLED=true)
    # Use a Redis leader lock so exactly one of the N uvicorn workers runs the bot.
    app.state.telegram_bot = None
    app.state.telegram_leader_redis = None
    app.state.telegram_leader_stop = None
    app.state.telegram_leader_task = None
    try:
        lock_redis = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            password=settings.redis.password if settings.redis.password else None,
            db=settings.redis.db,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        acquired = lock_redis.set(_TELEGRAM_LOCK_KEY, os.getpid(), nx=True, ex=_TELEGRAM_LOCK_TTL)
        if acquired:
            telegram_bot = make_telegram_service(
                opensearch_client=opensearch_client,
                embeddings_client=app.state.embeddings_service,
                llm_client=app.state.llm_client,
                agentic_service=app.state.agentic_rag_service,
                cache_client=app.state.cache_client,
            )
            if telegram_bot:
                await telegram_bot.start()
                app.state.telegram_bot = telegram_bot
                stop_event = asyncio.Event()
                app.state.telegram_leader_redis = lock_redis
                app.state.telegram_leader_stop = stop_event
                app.state.telegram_leader_task = asyncio.create_task(
                    _run_leader_heartbeat(lock_redis, stop_event)
                )
        else:
            logger.info("Telegram bot already running in another worker (skipping)")
            lock_redis.close()
    except Exception as e:
        logger.error(f"Failed to start Telegram bot: {e}", exc_info=True)

    logger.info("API ready")
    yield

    # Cleanup
    if getattr(app.state, "telegram_bot", None):
        if getattr(app.state, "telegram_leader_stop", None):
            app.state.telegram_leader_stop.set()
        if getattr(app.state, "telegram_leader_task", None):
            app.state.telegram_leader_task.cancel()
        if getattr(app.state, "telegram_leader_redis", None):
            try:
                app.state.telegram_leader_redis.delete(_TELEGRAM_LOCK_KEY)
            except Exception:
                pass
        try:
            await app.state.telegram_bot.stop()
        except Exception as e:
            logger.warning(f"Error stopping Telegram bot: {e}")
    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(ping.router, prefix="/api/v1")
app.include_router(papers.router, prefix="/api/v1")
app.include_router(hybrid_search.router, prefix="/api/v1")  # Hybrid search supporting all modes
app.include_router(ask_router, prefix="/api/v1")  # RAG question answering with LLM
app.include_router(stream_router, prefix="/api/v1")  # Streaming RAG responses
app.include_router(agentic_router, prefix="/api/v1")  # Agentic RAG (LangGraph) with guardrail + grading


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
