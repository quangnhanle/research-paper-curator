"""Airflow task: push a digest of newly crawled papers to Telegram.

Runs after the ingestion pipeline. It is fully decoupled from the polling bot
in the FastAPI process: it builds a Markdown digest from XCom stats + the most
recently stored papers and sends it through the stateless notifier.
"""

import asyncio
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Telegram legacy-Markdown special characters that break parsing inside text.
_MD_SPECIAL = ("_", "*", "`", "[")


def _escape_md(text: str) -> str:
    """Escape characters that would break Telegram legacy Markdown."""
    for ch in _MD_SPECIAL:
        text = text.replace(ch, f"\\{ch}")
    return text


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())  # collapse whitespace/newlines
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def build_digest(report: Dict, papers: List[Dict], abstract_chars: int = 200) -> str:
    """Build the Markdown digest message.

    :param report: The ``daily_report`` dict (counts + status).
    :param papers: List of dicts with ``title``, ``arxiv_id``, ``abstract``.
    :param abstract_chars: Abstract chars per paper (0 = omit abstracts).
    """
    fetch = report.get("fetch_statistics", {})
    index = report.get("indexing_statistics", {})
    status = report.get("pipeline_status", "unknown")
    target_date = fetch.get("target_date", "unknown")

    status_icon = {"success": "✅", "partial": "⚠️"}.get(status, "❌")

    lines = [
        f"{status_icon} *arXiv ingestion — {_escape_md(str(target_date))}*",
        "",
        f"📥 Fetched: {fetch.get('papers_fetched', 0)}  •  "
        f"💾 Stored: {fetch.get('papers_stored', 0)}  •  "
        f"🔎 Indexed: {index.get('papers_processed', 0)}",
    ]

    if papers:
        lines.append("")
        lines.append(f"*Latest {len(papers)} papers:*")
        for idx, p in enumerate(papers, 1):
            title = _escape_md(p.get("title", "Untitled").strip())
            arxiv_id = p.get("arxiv_id", "")
            lines.append("")
            lines.append(f"{idx}. {title}")
            if arxiv_id:
                lines.append(f"https://arxiv.org/abs/{arxiv_id}")
            if abstract_chars > 0:
                abstract = _truncate(p.get("abstract", ""), abstract_chars)
                if abstract:
                    lines.append(f"_{_escape_md(abstract)}_")
    else:
        lines.append("")
        lines.append("_No new papers were stored in this run._")

    return "\n".join(lines)


def _fetch_recent_papers(limit: int) -> List[Dict]:
    """Pull the most recently ingested papers from the database as plain dicts."""
    from .common import get_cached_services

    _arxiv, _pdf, database, _meta, _os = get_cached_services()
    with database.get_session() as session:
        from src.repositories.paper import PaperRepository

        repo = PaperRepository(session)
        papers = repo.get_recent_papers(limit=limit)
        return [
            {
                "title": p.title,
                "arxiv_id": p.arxiv_id,
                "abstract": p.abstract or "",
            }
            for p in papers
        ]


def notify_telegram(**context):
    """Send the ingestion digest to Telegram if push notifications are enabled."""
    from src.config import get_settings
    from src.services.telegram.notifier import send_telegram_message

    tg = get_settings().telegram

    if not tg.notify_on_ingestion:
        logger.info("Telegram ingestion notifications disabled (TELEGRAM__NOTIFY_ON_INGESTION=false)")
        return {"status": "disabled"}
    if not tg.bot_token or not tg.notify_chat_id:
        logger.warning("Telegram notify enabled but TELEGRAM__BOT_TOKEN / TELEGRAM__NOTIFY_CHAT_ID not configured")
        return {"status": "misconfigured"}

    ti = context.get("ti")
    report = (ti.xcom_pull(task_ids="generate_daily_report", key="daily_report") if ti else None) or {}

    try:
        papers = _fetch_recent_papers(tg.notify_max_papers)
    except Exception as e:
        logger.error(f"Failed to load recent papers for digest: {e}", exc_info=True)
        papers = []

    message = build_digest(report, papers, abstract_chars=tg.notify_abstract_chars)

    try:
        parts = asyncio.run(
            send_telegram_message(
                bot_token=tg.bot_token,
                chat_id=tg.notify_chat_id,
                text=message,
                max_message_length=tg.max_message_length,
            )
        )
        logger.info(f"Telegram digest sent in {parts} part(s)")
        return {"status": "sent", "parts": parts, "papers": len(papers)}
    except Exception as e:
        logger.error(f"Failed to send Telegram digest: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
