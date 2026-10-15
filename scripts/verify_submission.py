"""
Submission Quality Assurance Verification Script.
Executes an end-to-end check across all assignment criteria:
  - Corpus integrity
  - Vector Store health & chunk indexing
  - LangGraph self-corrective workflow execution
  - API endpoint integration
  - Unit test suite validation
"""
import sys
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vector_store import get_all_indexed_documents
from app.graph.workflow import rag_app
from fastapi.testclient import TestClient
from app.main import app

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def verify_all():
    scorecard = {}

    # 1. Corpus Verification
    print_header("1. Verifying Technical Document Corpus")
    raw_dir = PROJECT_ROOT / "data" / "raw"
    md_files = list(raw_dir.glob("*.md"))
    print(f"Found {len(md_files)} documentation files in {raw_dir}:")
    for f in md_files:
        print(f"  [OK] {f.name} ({f.stat().st_size} bytes)")
    scorecard["Technical Corpus"] = "PASS" if len(md_files) >= 4 else "FAIL"

    # 2. Vector DB Verification
    print_header("2. Verifying Persistent ChromaDB Index")
    indexed_docs = get_all_indexed_documents()
    total_chunks = sum(d["chunks_count"] for d in indexed_docs)
    print(f"Total Sources Indexed: {len(indexed_docs)}")
    print(f"Total Vector Chunks:   {total_chunks}")
    for d in indexed_docs:
        print(f"  - {d['source']}: {d['chunks_count']} chunks across {len(d['sections'])} sections")
    scorecard["ChromaDB Vector Index"] = "PASS" if total_chunks > 0 else "FAIL"

    # 3. LangGraph Workflow Verification
    print_header("3. Verifying LangGraph CRAG & Self-RAG Workflow")
    test_state = {
        "question": "How do dependencies with yield work for database cleanup in FastAPI?",
        "query_type": None,
        "transformed_query": "How do dependencies with yield work for database cleanup in FastAPI?",
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_grounded": None,
        "web_fallback_used": False,
        "chat_history": None
    }
    result = rag_app.invoke(test_state)
    print(f"  Query Type:       {result.get('query_type')}")
    print(f"  Relevant Chunks:  {len(result.get('relevant_docs', []))}")
    print(f"  Retries Used:     {result.get('retry_count')}")
    print(f"  Grounded Flag:    {result.get('is_grounded')}")
    print(f"  Citations Count:  {len(result.get('citations', []))}")
    print(f"  Answer Snippet:   {result.get('generation', '')[:140]}...")
    workflow_ok = len(result.get("relevant_docs", [])) > 0 and len(result.get("citations", [])) > 0
    scorecard["LangGraph Workflow"] = "PASS" if workflow_ok else "FAIL"

    # 4. API Endpoints Verification
    print_header("4. Verifying FastAPI Endpoints via TestClient")
    client = TestClient(app)
    
    r_root = client.get("/")
    r_docs = client.get("/documents")
    r_query = client.post("/query", json={"question": "What is CORSMiddleware?"})
    r_fb = client.post("/feedback", json={"query": "test", "answer": "test", "rating": "up"})
    
    print(f"  GET /           : HTTP {r_root.status_code}")
    print(f"  GET /documents  : HTTP {r_docs.status_code}")
    print(f"  POST /query     : HTTP {r_query.status_code}")
    print(f"  POST /feedback  : HTTP {r_fb.status_code}")
    
    api_ok = all(code == 200 for code in [r_root.status_code, r_docs.status_code, r_query.status_code, r_fb.status_code])
    scorecard["FastAPI Endpoints"] = "PASS" if api_ok else "FAIL"

    # 5. Automated Pytest Suite
    print_header("5. Running Full Pytest Test Suite")
    python_exe = sys.executable
    proc = subprocess.run([python_exe, "-m", "pytest", "tests/"], capture_output=True, text=True)
    print(proc.stdout)
    scorecard["Pytest Test Suite"] = "PASS" if proc.returncode == 0 else "FAIL"

    # Scorecard Summary
    print_header("SUBMISSION READINESS SCORECARD")
    all_passed = True
    for requirement, status in scorecard.items():
        symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        if status != "PASS":
            all_passed = False
        print(f"  {symbol:<8} {requirement}")

    print("-" * 70)
    if all_passed:
        print("🎉 ALL CRITERIA PASSED! REPOSITORY IS 100% READY FOR SUBMISSION.")
    else:
        print("⚠️ Some checks failed. Review output above.")
    print("=" * 70)

if __name__ == "__main__":
    verify_all()
