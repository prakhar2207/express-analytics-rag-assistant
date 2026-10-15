"""
End-to-End LangGraph Self-Corrective Workflow (CRAG) Test Script.
Tests:
  1. In-domain technical questions (direct retrieval & generation).
  2. Query expansion and intent classification.
  3. Out-of-domain query triggering document grading rejection and fallback.
"""
import sys
from pathlib import Path

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.graph.workflow import rag_app
from app.core.config import settings

def run_test_scenario(scenario_name: str, question: str):
    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario_name}")
    print(f"Question: \"{question}\"")
    print("=" * 70)

    initial_state = {
        "question": question,
        "query_type": None,
        "transformed_query": question,
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": settings.MAX_RETRIES,
        "is_grounded": None,
        "web_fallback_used": False,
    }

    result = rag_app.invoke(initial_state)

    print(f"[+] Query Type Classified: {result.get('query_type')}")
    print(f"[+] Transformed Query:     {result.get('transformed_query')}")
    print(f"[+] Raw Documents Found:   {len(result.get('documents', []))}")
    print(f"[+] Relevant Docs Graded:  {len(result.get('relevant_docs', []))}")
    print(f"[+] Retries Used:          {result.get('retry_count', 0)} / {settings.MAX_RETRIES}")
    print(f"[+] Web Fallback Used:     {result.get('web_fallback_used', False)}")
    print(f"[+] Citations Extracted:   {result.get('citations')}")
    print("\n[Synthesized Answer]:")
    print(result.get("generation"))
    print("=" * 70)
    return result

def main():
    print("Running Full LangGraph Self-Corrective Workflow Suite...")

    # Test 1: In-domain question
    res1 = run_test_scenario(
        "1. In-Domain Specific Question (Dependency Injection & Yield)",
        "How do I use yield in dependencies to clean up database sessions in FastAPI?"
    )
    assert len(res1["relevant_docs"]) > 0, "Expected relevant documents to be found"
    assert len(res1["citations"]) > 0, "Expected citations to be generated"

    # Test 2: In-domain Security question
    res2 = run_test_scenario(
        "2. Security & OAuth2 Password Bearer",
        "How to authenticate users using OAuth2PasswordBearer in FastAPI?"
    )
    assert len(res2["relevant_docs"]) > 0, "Expected relevant security documents"

    # Test 3: Completely Out-of-Domain question
    res3 = run_test_scenario(
        "3. Out-Of-Domain Question (Triggers Self-Correction & Rejection)",
        "How do I bake sourdough bread with active dry yeast?"
    )
    # The out-of-domain question should trigger query rewrites or web fallback
    assert res3["retry_count"] > 0 or res3["web_fallback_used"], "Expected self-correction or fallback to trigger"

    print("\n[SUCCESS] All 3 End-to-End Workflow Scenarios Passed Successfully!")

if __name__ == "__main__":
    main()
