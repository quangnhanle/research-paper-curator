"""Graph nodes for the agentic RAG workflow.

Nodes are bound to the runtime services (OpenSearch, embeddings, LLM) via the
``AgenticRAGNodes`` class. Each node is a small async function that reads the
shared ``AgentState`` and returns a partial state update. Routing functions
(``route_*``) decide which edge to follow and return node names as strings.
"""

import logging
import re
from typing import Any, Dict, List

from src.services.agents.prompts import (
    build_grade_messages,
    build_guardrail_messages,
    build_out_of_scope_message,
    build_rewrite_messages,
)
from src.services.agents.state import AgentState, GraphConfig

logger = logging.getLogger(__name__)


class AgenticRAGNodes:
    """Holds service dependencies and exposes the graph node functions."""

    def __init__(self, opensearch_client, embeddings_client, llm_client, config: GraphConfig):
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.llm = llm_client
        self.config = config

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    async def guardrail(self, state: AgentState) -> Dict[str, Any]:
        """Score the query's relevance to the research domain (0-100)."""
        query = state["query"]
        score = 0
        try:
            raw = await self.llm.generate(
                messages=build_guardrail_messages(query),
                model=state.get("model"),
                temperature=self.config.temperature,
            )
            score = self._parse_score(raw)
        except Exception as e:  # never let the guardrail crash the request
            logger.warning(f"Guardrail scoring failed, defaulting to in-scope: {e}")
            score = self.config.guardrail_threshold  # fail open so real questions still work

        out_of_scope = score < self.config.guardrail_threshold
        logger.info(f"Guardrail score={score} (threshold={self.config.guardrail_threshold}) out_of_scope={out_of_scope}")
        return {
            "guardrail_score": score,
            "out_of_scope": out_of_scope,
            "reasoning_steps": [f"Validated query scope (score: {score}/100)"],
        }

    async def out_of_scope(self, state: AgentState) -> Dict[str, Any]:
        """Terminate early with a helpful rejection message."""
        return {
            "answer": build_out_of_scope_message(state["query"]),
            "sources": [],
            "search_mode": "none",
            "reasoning_steps": ["Rejected query as out of scope"],
        }

    async def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Run hybrid/BM25 search for the current query and collect chunks."""
        query = state["query"]
        top_k = state.get("top_k", self.config.top_k)
        use_hybrid = state.get("use_hybrid", True)

        query_embedding = None
        search_mode = "bm25"
        if use_hybrid:
            try:
                query_embedding = await self.embeddings.embed_query(query)
                search_mode = "hybrid"
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to BM25: {e}")

        results = self.opensearch.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            categories=state.get("categories"),
            use_hybrid=use_hybrid and query_embedding is not None,
        )

        chunks, sources = self._extract_chunks_and_sources(results.get("hits", []))
        attempts = state.get("retrieval_attempts", 0) + 1
        logger.info(f"Retrieval attempt {attempts}: {len(chunks)} chunks ({search_mode})")
        return {
            "chunks": chunks,
            "sources": sources,
            "search_mode": search_mode,
            "retrieval_attempts": attempts,
            "reasoning_steps": [f"Retrieved documents ({attempts} attempt(s))"],
        }

    async def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Ask the LLM whether the retrieved chunks can answer the query."""
        chunks = state.get("chunks", [])
        if not chunks:
            return {
                "documents_relevant": False,
                "reasoning_steps": ["Graded documents (none retrieved)"],
            }

        relevant = False
        try:
            raw = await self.llm.generate(
                messages=build_grade_messages(state["query"], chunks),
                model=state.get("model"),
                temperature=self.config.temperature,
            )
            relevant = raw.strip().lower().startswith("y")
        except Exception as e:
            logger.warning(f"Grading failed, assuming relevant: {e}")
            relevant = True  # fail open: better to answer than to loop

        return {
            "documents_relevant": relevant,
            "reasoning_steps": [f"Graded documents ({'relevant' if relevant else 'not relevant'})"],
        }

    async def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """Refine a vague query before retrying retrieval."""
        original = state["query"]
        rewritten = original
        try:
            raw = await self.llm.generate(
                messages=build_rewrite_messages(original),
                model=state.get("model"),
                temperature=self.config.temperature,
            )
            candidate = raw.strip().strip('"').strip()
            if candidate:
                rewritten = candidate
        except Exception as e:
            logger.warning(f"Query rewrite failed, keeping original: {e}")

        logger.info(f"Rewrote query: '{original}' -> '{rewritten}'")
        return {
            "query": rewritten,
            "rewritten_query": rewritten,
            "reasoning_steps": ["Rewritten query for better results"],
        }

    async def generate_answer(self, state: AgentState) -> Dict[str, Any]:
        """Generate the final answer from the retrieved context."""
        chunks = state.get("chunks", [])
        if not chunks:
            return {
                "answer": "I couldn't find any relevant papers to answer your question.",
                "reasoning_steps": ["No relevant documents found"],
            }

        rag_response = await self.llm.generate_rag_answer(
            query=state.get("original_query", state["query"]),
            chunks=chunks,
            model=state.get("model"),
        )
        answer = rag_response.get("answer", "Unable to generate an answer.")
        # Prefer LLM-reported sources, fall back to the ones we built from search hits.
        sources = rag_response.get("sources") or state.get("sources", [])
        return {
            "answer": answer,
            "sources": sources,
            "reasoning_steps": ["Generated answer from context"],
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_score(raw: str) -> int:
        """Extract the first integer 0-100 from the model's reply."""
        match = re.search(r"\d{1,3}", raw or "")
        if not match:
            return 0
        return max(0, min(100, int(match.group())))

    @staticmethod
    def _extract_chunks_and_sources(hits: List[Dict[str, Any]]):
        chunks: List[Dict[str, Any]] = []
        sources_set = set()
        for hit in hits:
            arxiv_id = hit.get("arxiv_id", "")
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )
            if arxiv_id:
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")
        return chunks, list(sources_set)


# ---------------------------------------------------------------------- #
# Routing functions (conditional edges)
# ---------------------------------------------------------------------- #
def route_after_guardrail(state: AgentState) -> str:
    """Block out-of-scope queries, otherwise proceed to retrieval."""
    return "out_of_scope" if state.get("out_of_scope") else "retrieve"


def make_route_after_grading(config: GraphConfig):
    """Build the post-grading router (closes over the attempt budget)."""

    def route_after_grading(state: AgentState) -> str:
        if state.get("documents_relevant"):
            return "generate_answer"
        # Not relevant: retry via rewrite if we still have attempts left.
        if state.get("retrieval_attempts", 0) < config.max_retrieval_attempts:
            return "rewrite_query"
        # Out of attempts: answer with whatever we have (best effort).
        return "generate_answer"

    return route_after_grading
