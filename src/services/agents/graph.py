"""LangGraph workflow assembly for agentic RAG.

Topology::

    START
      -> guardrail (LLM scores 0-100)
           |- out_of_scope -> END           (score < threshold)
           '- retrieve                        (score >= threshold)
                -> grade_documents
                     |- generate_answer -> END           (relevant, or out of attempts)
                     '- rewrite_query -> retrieve         (not relevant, attempts remain)
"""

from langgraph.graph import END, START, StateGraph
from src.services.agents.nodes import (
    AgenticRAGNodes,
    make_route_after_grading,
    route_after_guardrail,
)
from src.services.agents.state import AgentState, GraphConfig


def build_agentic_graph(nodes: AgenticRAGNodes, config: GraphConfig):
    """Wire the nodes into a compiled LangGraph runnable."""
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", nodes.guardrail)
    graph.add_node("out_of_scope", nodes.out_of_scope)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate_answer", nodes.generate_answer)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"out_of_scope": "out_of_scope", "retrieve": "retrieve"},
    )
    graph.add_edge("out_of_scope", END)
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        make_route_after_grading(config),
        {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)

    return graph.compile()
