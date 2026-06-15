import logging
from time import time

from fastapi import APIRouter, HTTPException
from src.dependencies import AgenticRAGDep
from src.schemas.api.agentic import AgenticAskRequest, AgenticAskResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agentic"])


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AgenticAskRequest,
    agentic_service: AgenticRAGDep,
) -> AgenticAskResponse:
    """Agentic RAG endpoint.

    Unlike ``/ask`` (which always retrieves), this runs a LangGraph workflow that:

    1. Validates the query is in-scope via a guardrail score (0-100).
    2. Retrieves and grades documents for relevance.
    3. Rewrites the query and retries if the first results are weak.
    4. Generates a grounded answer, returning the full reasoning trace.
    """
    start = time()
    try:
        response = await agentic_service.ask(request)
        logger.info(
            f"Agentic RAG answered '{request.query}' in {time() - start:.1f}s "
            f"({response.retrieval_attempts} attempt(s))"
        )
        return response
    except Exception as e:
        logger.error(f"Agentic RAG failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agentic RAG failed: {str(e)}")
