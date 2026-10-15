from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    State tracking object flowing through the LangGraph self-corrective RAG workflow.
    """
    question: str                         # Original user question
    query_type: Optional[str]             # Classified query type (conceptual, how-to, troubleshooting, api_reference)
    transformed_query: str                # Rewritten or expanded query for vector search
    documents: List[Dict[str, Any]]       # Raw documents retrieved from vector store
    relevant_docs: List[Dict[str, Any]]   # Documents graded as relevant by LLM grader
    generation: str                       # Final generated response text
    citations: List[str]                  # Extracted references/sources grounded in context
    retry_count: int                      # Number of re-retrieval / query-rewrite iterations
    max_retries: int                      # Maximum retry ceiling to avoid infinite loops
    is_grounded: Optional[bool]           # Hallucination / groundedness check outcome
    web_fallback_used: bool               # Flag indicating whether web search fallback was used
