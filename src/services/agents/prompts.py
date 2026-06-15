"""Prompt templates for the agentic RAG nodes.

Each builder returns Chat Completions ``messages`` (list of role/content dicts)
so they plug straight into ``ExternalLLMClient.generate``.
"""

from typing import Any, Dict, List

# Domain the curator is scoped to. Used by the guardrail to reject off-topic queries.
DOMAIN_DESCRIPTION = (
    "computer science / machine learning / NLP / AI research papers (e.g. transformers, "
    "attention, embeddings, neural networks, language models, retrieval, training methods)"
)


def build_guardrail_messages(query: str) -> List[Dict[str, str]]:
    """Score 0-100 how relevant a query is to the curator's research domain."""
    system = (
        "You are a scope classifier for a research-paper assistant covering "
        f"{DOMAIN_DESCRIPTION}.\n"
        "Rate how relevant the user's query is to this domain on a scale of 0 to 100, "
        "where 0 = completely unrelated (greetings, weather, general trivia) and "
        "100 = clearly an ML/NLP/AI research question.\n"
        "Respond with ONLY the integer score. No words, no punctuation, no explanation."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {query}\nScore:"},
    ]


def build_grade_messages(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Decide whether retrieved chunks are relevant enough to answer the query."""
    context = "\n\n".join(
        f"[Doc {i}] {chunk.get('chunk_text', chunk.get('abstract', ''))[:600]}"
        for i, chunk in enumerate(chunks, 1)
    )
    system = (
        "You are a grader assessing whether retrieved document excerpts are relevant "
        "to a user's question. Answer with ONLY 'yes' or 'no'. "
        "'yes' if at least one excerpt contains information useful to answer the question, "
        "otherwise 'no'."
    )
    user = f"Question: {query}\n\nRetrieved excerpts:\n{context}\n\nAre these relevant? (yes/no):"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_rewrite_messages(query: str) -> List[Dict[str, str]]:
    """Rewrite a vague query into a focused research search query."""
    system = (
        "You rewrite vague questions into precise search queries for a database of "
        f"{DOMAIN_DESCRIPTION}.\n"
        "Produce a single improved query that surfaces the underlying research intent. "
        "Respond with ONLY the rewritten query text, no preamble or quotes."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Original question: {query}\nRewritten query:"},
    ]


def build_out_of_scope_message(query: str) -> str:
    """Friendly rejection shown when the guardrail blocks a query."""
    return (
        "I'm focused on computer science and machine learning research papers from arXiv, "
        "so I can't help with that question. Try asking about topics like transformers, "
        "attention mechanisms, embeddings, language models, or other ML/NLP research."
    )
