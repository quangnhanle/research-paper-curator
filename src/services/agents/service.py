"""Service wrapper around the compiled agentic RAG graph."""

import logging
from typing import List, Optional

from src.schemas.api.agentic import AgenticAskRequest, AgenticAskResponse
from src.services.agents.graph import build_agentic_graph
from src.services.agents.nodes import AgenticRAGNodes
from src.services.agents.state import AgentState, GraphConfig

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """High-level entry point: turns a request into a graph invocation."""

    def __init__(self, opensearch_client, embeddings_client, llm_client, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.nodes = AgenticRAGNodes(opensearch_client, embeddings_client, llm_client, self.config)
        self.graph = build_agentic_graph(self.nodes, self.config)

    async def ask(self, request: AgenticAskRequest) -> AgenticAskResponse:
        """Run the agentic workflow end-to-end for a single request."""
        initial_state: AgentState = {
            "query": request.query,
            "original_query": request.query,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": request.categories,
            "model": request.model,
            "retrieval_attempts": 0,
            "reasoning_steps": [],
        }

        logger.info(f"Agentic RAG starting for query: '{request.query}'")
        final_state = await self.graph.ainvoke(initial_state)

        chunks: List[dict] = final_state.get("chunks", [])
        return AgenticAskResponse(
            query=request.query,
            answer=final_state.get("answer", "Unable to generate an answer."),
            sources=final_state.get("sources", []),
            chunks_used=len(chunks),
            search_mode=final_state.get("search_mode", "none"),
            reasoning_steps=final_state.get("reasoning_steps", []),
            retrieval_attempts=final_state.get("retrieval_attempts", 0),
            rewritten_query=final_state.get("rewritten_query"),
        )
