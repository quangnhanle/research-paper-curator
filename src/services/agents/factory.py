import logging

from src.services.agents.service import AgenticRAGService
from src.services.agents.state import GraphConfig

logger = logging.getLogger(__name__)


def make_agentic_rag_service(
    opensearch_client,
    embeddings_client,
    llm_client,
    config: GraphConfig | None = None,
) -> AgenticRAGService:
    """Create the agentic RAG service, reusing the app's existing clients."""
    service = AgenticRAGService(
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        llm_client=llm_client,
        config=config,
    )
    logger.info("Agentic RAG service created successfully")
    return service
