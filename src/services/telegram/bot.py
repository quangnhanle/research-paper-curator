import logging
from typing import Optional

from src.schemas.api.agentic import AgenticAskRequest
from src.schemas.api.ask import AskRequest, AskResponse
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logger = logging.getLogger(__name__)

TELEGRAM_MAX = 4096


class TelegramBot:
    """Conversational Telegram interface to the RAG / agentic-RAG pipeline."""

    def __init__(
        self,
        bot_token: str,
        opensearch_client,
        embeddings_client,
        llm_client,
        agentic_service=None,
        cache_client=None,
        allowed_user_ids: Optional[set[int]] = None,
        use_agentic: bool = True,
        default_top_k: int = 3,
        default_use_hybrid: bool = True,
        max_message_length: int = 4000,
    ):
        self.bot_token = bot_token
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.llm = llm_client
        self.agentic = agentic_service
        self.cache = cache_client
        self.allowed_user_ids = allowed_user_ids or set()
        self.use_agentic = use_agentic
        self.default_top_k = default_top_k
        self.default_use_hybrid = default_use_hybrid
        self.max_message_length = min(max_message_length, TELEGRAM_MAX)
        self.application: Optional[Application] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Start the bot in polling mode."""
        logger.info("Starting Telegram bot...")
        self.application = Application.builder().token(self.bot_token).build()

        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("ask", self._ask_command))
        self.application.add_handler(CommandHandler("search", self._search_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_question))

        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started successfully (polling mode)")

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        if self.application:
            if self.application.updater:
                await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped")

    # ------------------------------------------------------------------ #
    # Access control
    # ------------------------------------------------------------------ #
    def _is_authorized(self, update: Update) -> bool:
        if not self.allowed_user_ids:
            return True  # empty whitelist = allow everyone
        user = update.effective_user
        return bool(user and user.id in self.allowed_user_ids)

    async def _deny(self, update: Update) -> None:
        await update.message.reply_text("Sorry, you're not authorized to use this bot.")

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return await self._deny(update)
        await update.message.reply_text(
            "Welcome to the arXiv Paper Curator!\n\n"
            "Ask me questions about computer science / ML research papers and I'll answer "
            "with sources from arXiv.\n\n"
            "Commands:\n"
            "/ask <question> - Ask a research question\n"
            "/search <keywords> - Find relevant papers\n"
            "/status - Check system health\n"
            "/help - Show usage tips\n\n"
            "Or just send me a question directly!"
        )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return await self._deny(update)
        await update.message.reply_text(
            "Send any question about CS / ML research papers.\n\n"
            "Examples:\n"
            "- What are transformer architectures?\n"
            "- How does BERT differ from GPT?\n"
            "- Explain self-attention mechanisms\n\n"
            "I use an agentic workflow: I validate the question is in scope, retrieve "
            "relevant papers, check they're useful, and refine the search if needed.\n\n"
            "Use /search to list matching papers without a generated answer."
        )

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return await self._deny(update)

        lines = ["System Status\n"]
        # OpenSearch
        try:
            ok = self.opensearch.health_check()
            lines.append(f"{'✅' if ok else '❌'} OpenSearch")
        except Exception:
            lines.append("❌ OpenSearch")
        # LLM
        try:
            await self.llm.health_check()
            lines.append("✅ LLM provider")
        except Exception:
            lines.append("❌ LLM provider")
        # Cache
        if self.cache:
            lines.append("✅ Cache")
        else:
            lines.append("⚪ Cache (disabled)")

        await update.message.reply_text("\n".join(lines))

    async def _ask_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return await self._deny(update)
        if not context.args:
            await update.message.reply_text("Usage: /ask <question>\nExample: /ask What is self-attention?")
            return
        await self._answer_question(update, " ".join(context.args))

    async def _search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return await self._deny(update)
        if not context.args:
            await update.message.reply_text("Usage: /search <keywords>\nExample: /search neural networks")
            return

        query = " ".join(context.args)
        await update.message.chat.send_action(ChatAction.TYPING)
        try:
            query_embedding = None
            try:
                query_embedding = await self.embeddings.embed_query(query)
            except Exception as e:
                logger.warning(f"Embedding failed for search, using BM25: {e}")

            results = self.opensearch.search_unified(
                query=query,
                query_embedding=query_embedding,
                size=10,
                use_hybrid=query_embedding is not None,
            )

            # Deduplicate by arxiv_id (chunks share a paper)
            seen, papers = set(), []
            for hit in results.get("hits", []):
                arxiv_id = hit.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in seen:
                    seen.add(arxiv_id)
                    papers.append(hit)
                if len(papers) >= 5:
                    break

            if not papers:
                await update.message.reply_text("No papers found. Try different keywords.")
                return

            msg = f"Found {len(papers)} papers:\n\n"
            for idx, hit in enumerate(papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                msg += f"{idx}. {title}\nhttps://arxiv.org/abs/{arxiv_id}\n\n"

            await update.message.reply_text(msg, disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            await update.message.reply_text(f"Search failed: {str(e)}")

    async def _handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return await self._deny(update)
        await self._answer_question(update, update.message.text)

    # ------------------------------------------------------------------ #
    # Core Q&A
    # ------------------------------------------------------------------ #
    async def _answer_question(self, update: Update, query: str) -> None:
        await update.message.chat.send_action(ChatAction.TYPING)

        cache_request = AskRequest(query=query, top_k=self.default_top_k, use_hybrid=self.default_use_hybrid)

        # 1) Exact-match cache
        if self.cache:
            try:
                cached = await self.cache.find_cached_response(cache_request)
                if cached:
                    await self._send_answer(update, cached.answer, cached.sources, cached=True)
                    return
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        # 2) Generate (agentic workflow preferred, plain RAG fallback)
        try:
            if self.use_agentic and self.agentic:
                agentic_resp = await self.agentic.ask(
                    AgenticAskRequest(query=query, top_k=self.default_top_k, use_hybrid=self.default_use_hybrid)
                )
                answer, sources = agentic_resp.answer, agentic_resp.sources
                chunks_used, search_mode = agentic_resp.chunks_used, agentic_resp.search_mode
            else:
                answer, sources, chunks_used, search_mode = await self._plain_rag(query)

            if not answer:
                await update.message.reply_text("No relevant papers found. Try rephrasing your question.")
                return

            # 3) Store in cache
            if self.cache:
                try:
                    await self.cache.store_response(
                        cache_request,
                        AskResponse(
                            query=query,
                            answer=answer,
                            sources=sources,
                            chunks_used=chunks_used,
                            search_mode=search_mode,
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Cache store failed: {e}")

            await self._send_answer(update, answer, sources)
        except Exception as e:
            logger.error(f"Question handling failed: {e}", exc_info=True)
            await update.message.reply_text(f"Error processing your question: {str(e)}")

    async def _plain_rag(self, query: str):
        """Fallback single-pass RAG (used if the agentic service is unavailable)."""
        query_embedding = None
        search_mode = "bm25"
        if self.default_use_hybrid:
            try:
                query_embedding = await self.embeddings.embed_query(query)
                search_mode = "hybrid"
            except Exception as e:
                logger.warning(f"Embedding failed, using BM25: {e}")

        results = self.opensearch.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=self.default_top_k,
            use_hybrid=self.default_use_hybrid and query_embedding is not None,
        )

        chunks, sources = set(), set()
        chunk_list = []
        for hit in results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")
            chunk_list.append({"arxiv_id": arxiv_id, "chunk_text": hit.get("chunk_text", hit.get("abstract", ""))})
            if arxiv_id:
                clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources.add(f"https://arxiv.org/pdf/{clean}.pdf")

        if not chunk_list:
            return "", [], 0, search_mode

        rag = await self.llm.generate_rag_answer(query=query, chunks=chunk_list)
        answer = rag.get("answer", "")
        srcs = rag.get("sources") or list(sources)
        return answer, srcs, len(chunk_list), search_mode

    # ------------------------------------------------------------------ #
    # Formatting / sending
    # ------------------------------------------------------------------ #
    async def _send_answer(self, update: Update, answer: str, sources: list[str], cached: bool = False) -> None:
        message = f"*Answer:*\n{answer}\n"
        if sources:
            message += "\n*Sources:*\n"
            for idx, url in enumerate(sources[:5], 1):
                arxiv_id = url.rstrip("/").split("/")[-1].replace(".pdf", "")
                message += f"{idx}. https://arxiv.org/abs/{arxiv_id}\n"
        if cached:
            message += "\n⚡ Cached"

        for part in self._split_message(message):
            try:
                await update.message.reply_text(part, parse_mode="Markdown", disable_web_page_preview=True)
            except Exception:
                # Markdown can break on special chars - fall back to plain text.
                await update.message.reply_text(part, disable_web_page_preview=True)

    def _split_message(self, text: str) -> list[str]:
        """Split a long message under Telegram's per-message limit, on line breaks."""
        limit = self.max_message_length
        if len(text) <= limit:
            return [text]

        parts, current = [], ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > limit:
                if current:
                    parts.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            parts.append(current)
        return parts
