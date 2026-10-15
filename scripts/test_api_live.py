"""
Live HTTP verification script for all FastAPI endpoints:
  - GET  /
  - GET  /documents
  - POST /query
  - POST /feedback
  - POST /ingest
"""
import sys
import json
import httpx
from pathlib import Path

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def test_live_api():
    print("=" * 70)
    print("Testing Live FastAPI Endpoints at", BASE_URL)
    print("=" * 70)

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Test GET /
        print("\n[1] GET / (Health & Service Info)")
        r = client.get("/")
        print(f"Status: {r.status_code}")
        print("Response:", r.json())
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

        # 2. Test GET /documents
        print("\n[2] GET /documents (Corpus Inspection)")
        r = client.get("/documents")
        print(f"Status: {r.status_code}")
        docs_data = r.json()
        print(f"Total Documents: {docs_data['total_documents']}, Total Chunks: {docs_data['total_chunks']}")
        for d in docs_data["documents"]:
            print(f"  - {d['source']} ({d['chunks_count']} chunks)")
        assert r.status_code == 200
        assert docs_data["total_documents"] >= 4

        # 3. Test POST /query
        print("\n[3] POST /query (Ask Documentation Assistant)")
        query_payload = {
            "question": "How do I configure CORS middleware in FastAPI with specific origins?"
        }
        r = client.post("/query", json=query_payload)
        print(f"Status: {r.status_code}")
        query_resp = r.json()
        print(f"Query:          {query_resp['question']}")
        print(f"Query Type:     {query_resp.get('query_type')}")
        print(f"Execution Time: {query_resp.get('execution_time_sec')}s")
        print(f"Citations:      {len(query_resp.get('sources', []))} sources")
        for s in query_resp.get("sources", [])[:2]:
            print(f"  * {s['source']} (Section: {s['section']})")
        print("Answer Summary:")
        print(query_resp["answer"][:250] + "...")
        assert r.status_code == 200
        assert len(query_resp["answer"]) > 0

        # 4. Test POST /feedback
        print("\n[4] POST /feedback (Record User Rating)")
        feedback_payload = {
            "query": query_resp["question"],
            "answer": query_resp["answer"],
            "rating": "up",
            "comment": "Accurate explanation of CORSMiddleware and origins list."
        }
        r = client.post("/feedback", json=feedback_payload)
        print(f"Status: {r.status_code}")
        feedback_resp = r.json()
        print("Feedback Response:", feedback_resp)
        assert r.status_code == 200
        assert "feedback_id" in feedback_resp

        # 5. Test POST /ingest (File upload)
        print("\n[5] POST /ingest (Upload New Document)")
        sample_doc_content = b"""# FastAPI Testing Guide
FastAPI applications can be tested seamlessly with pytest and TestClient from starlette.testclient.

## Using TestClient
Import TestClient, pass your app instance, and issue requests like client.get("/").
"""
        files = {
            "files": ("fastapi_testing_guide.md", sample_doc_content, "text/markdown")
        }
        r = client.post("/ingest", files=files)
        print(f"Status: {r.status_code}")
        ingest_resp = r.json()
        print("Ingest Response:", ingest_resp)
        assert r.status_code == 200
        assert ingest_resp["total_chunks_added"] > 0

        # Verify corpus was updated
        r = client.get("/documents")
        updated_docs = r.json()
        print(f"Updated Total Documents: {updated_docs['total_documents']}, Total Chunks: {updated_docs['total_chunks']}")
        assert any(d["source"] == "fastapi_testing_guide.md" for d in updated_docs["documents"])

    print("\n" + "=" * 70)
    print("[SUCCESS] All Live FastAPI Endpoints Verified Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_live_api()
