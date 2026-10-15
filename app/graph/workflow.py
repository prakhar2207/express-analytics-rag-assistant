from typing import Literal
from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes import (
    query_analysis_node,
    retrieval_node,
    document_grading_node,
    query_rewrite_node,
    web_fallback_node,
    generation_node
)
from app.core.config import settings

def route_after_grading(state: GraphState) -> Literal["generation", "rewrite_query", "web_fallback"]:
    """
    Conditional routing edge based on document grading results:
    - If relevant docs found -> proceed to generation
    - If no relevant docs and retries available -> rewrite query & loop back
    - If no relevant docs and retries exhausted -> fallback
    """
    relevant_docs = state.get("relevant_docs", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.MAX_RETRIES)

    if len(relevant_docs) > 0:
        return "generation"
    
    if retry_count < max_retries:
        return "rewrite_query"
    
    return "web_fallback"

def build_rag_graph():
    """
    Builds and compiles the Self-Corrective RAG (CRAG) StateGraph.
    """
    workflow = StateGraph(GraphState)

    # 1. Add all nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("document_grading", document_grading_node)
    workflow.add_node("rewrite_query", query_rewrite_node)
    workflow.add_node("web_fallback", web_fallback_node)
    workflow.add_node("generation", generation_node)

    # 2. Add edges
    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "document_grading")

    # Conditional routing after document grading
    workflow.add_conditional_edges(
        "document_grading",
        route_after_grading,
        {
            "generation": "generation",
            "rewrite_query": "rewrite_query",
            "web_fallback": "web_fallback",
        }
    )

    # Loop back from rewrite to retrieval
    workflow.add_edge("rewrite_query", "retrieval")

    # Connect web fallback to generation
    workflow.add_edge("web_fallback", "generation")

    # End after generation
    workflow.add_edge("generation", END)

    return workflow.compile()

# Singleton compiled graph instance
rag_app = build_rag_graph()
