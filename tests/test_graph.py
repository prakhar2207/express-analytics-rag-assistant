import pytest
from app.ingestion.chunker import TechnicalDocumentChunker
from app.graph.workflow import route_after_grading
from app.graph.state import GraphState

def test_chunker_header_splitting_and_metadata():
    markdown_sample = """# Main Header
Some intro text.

## Section Alpha
Here is alpha details.

```python
def foo():
    return 'bar'
```

## Section Beta
Here is beta details.
"""
    chunker = TechnicalDocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_markdown_text(markdown_sample, source_name="test_sample.md")
    
    assert len(chunks) >= 2
    for c in chunks:
        assert "source" in c.metadata
        assert c.metadata["source"] == "test_sample.md"
        assert "section_path" in c.metadata
        assert "chunk_index" in c.metadata

    # Check that Python code block remains in Section Alpha
    code_chunk = [c for c in chunks if "def foo():" in c.page_content]
    assert len(code_chunk) > 0
    assert "Section Alpha" in code_chunk[0].metadata.get("section_path", "")

def test_route_after_grading_with_relevant_docs():
    state: GraphState = {
        "question": "How to use dependencies?",
        "query_type": "how-to",
        "transformed_query": "How to use dependencies?",
        "documents": [{"content": "Doc 1"}],
        "relevant_docs": [{"content": "Doc 1"}],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_grounded": None,
        "web_fallback_used": False
    }
    decision = route_after_grading(state)
    assert decision == "generation"

def test_route_after_grading_zero_docs_with_retries_remaining():
    state: GraphState = {
        "question": "Some obscure question",
        "query_type": "conceptual",
        "transformed_query": "Some obscure question",
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_grounded": None,
        "web_fallback_used": False
    }
    decision = route_after_grading(state)
    assert decision == "rewrite_query"

def test_route_after_grading_zero_docs_retries_exhausted():
    state: GraphState = {
        "question": "Out of domain question",
        "query_type": "conceptual",
        "transformed_query": "Out of domain question",
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 2,
        "max_retries": 2,
        "is_grounded": None,
        "web_fallback_used": False
    }
    decision = route_after_grading(state)
    assert decision == "web_fallback"

def test_full_rag_workflow_execution():
    from app.graph.workflow import rag_app
    initial_state: GraphState = {
        "question": "How do I use yield in dependencies for database cleanup?",
        "query_type": None,
        "transformed_query": "How do I use yield in dependencies for database cleanup?",
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_grounded": None,
        "web_fallback_used": False,
    }
    result = rag_app.invoke(initial_state)
    assert "generation" in result
    assert len(result["generation"]) > 0
    assert len(result["relevant_docs"]) > 0
    assert result["query_type"] is not None
