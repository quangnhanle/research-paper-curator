from typing import List, Optional

from pydantic import BaseModel, Field


class AgenticAskRequest(BaseModel):
    """Request model for agentic RAG question answering."""

    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    top_k: int = Field(3, description="Number of top chunks to retrieve", ge=1, le=10)
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    model: Optional[str] = Field(None, description="LLM model to use (defaults to the configured model)")
    categories: Optional[List[str]] = Field(None, description="Filter by arXiv categories")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are attention mechanisms?",
                "top_k": 3,
                "use_hybrid": True,
                "model": None,
                "categories": None,
            }
        }


class AgenticAskResponse(BaseModel):
    """Response model for agentic RAG question answering.

    Extends the plain RAG response with the agent's decision-making trace so
    callers can see *why* an answer was produced (or rejected).
    """

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer (or out-of-scope rejection message)")
    sources: List[str] = Field(default_factory=list, description="PDF URLs of source papers")
    chunks_used: int = Field(0, description="Number of chunks used for generation")
    search_mode: str = Field("none", description="Search mode used: none, bm25, or hybrid")
    reasoning_steps: List[str] = Field(
        default_factory=list, description="Ordered trace of the agent's decisions"
    )
    retrieval_attempts: int = Field(0, description="Number of retrieval attempts performed (0-2)")
    rewritten_query: Optional[str] = Field(None, description="Query after refinement, if it was rewritten")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are attention mechanisms?",
                "answer": "Attention mechanisms allow models to dynamically focus on...",
                "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
                "chunks_used": 3,
                "search_mode": "hybrid",
                "reasoning_steps": [
                    "Validated query scope (score: 85/100)",
                    "Retrieved documents (1 attempt)",
                    "Graded documents (relevant)",
                    "Generated answer from context",
                ],
                "retrieval_attempts": 1,
                "rewritten_query": None,
            }
        }
