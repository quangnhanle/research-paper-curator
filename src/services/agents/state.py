import operator
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the agentic RAG graph.

    `total=False` because each node only fills in the keys it owns; nodes
    downstream read what previous nodes produced. Keeping this a plain
    ``TypedDict`` (rather than LangChain's ``MessagesState``) lets us drive our
    own OpenAI-compatible LLM client directly without a chat-model adapter.
    """

    # Inputs (set by the service before invoking the graph)
    query: str
    original_query: str
    top_k: int
    use_hybrid: bool
    categories: Optional[List[str]]
    model: Optional[str]

    # Guardrail
    guardrail_score: int
    out_of_scope: bool

    # Retrieval / grading
    chunks: List[Dict[str, Any]]
    sources: List[str]
    search_mode: str
    documents_relevant: bool
    retrieval_attempts: int
    rewritten_query: Optional[str]

    # Output
    answer: str

    # Transparency trace. The ``operator.add`` reducer makes each node's
    # returned list *append* to the running trace instead of overwriting it.
    reasoning_steps: Annotated[List[str], operator.add]


@dataclass(frozen=True)
class GraphConfig:
    """Tunable knobs for the agentic RAG workflow."""

    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60  # minimum 0-100 score to proceed to retrieval
    temperature: float = 0.0
    top_k: int = 3
