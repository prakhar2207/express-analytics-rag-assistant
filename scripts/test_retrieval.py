"""
Test script to verify vector store similarity search and metadata extraction.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vector_store import similarity_search_with_metadata

test_queries = [
    "How can I do database cleanup with yield in dependencies?",
    "When should I use Celery instead of FastAPI BackgroundTasks?",
    "How to configure CORSMiddleware with allowed origins?",
    "How does OAuth2PasswordBearer authenticate users with JWT?"
]

def main():
    print("=" * 70)
    print("Verifying ChromaDB Similarity Retrieval & Metadata Extraction")
    print("=" * 70)
    
    for q in test_queries:
        print(f"\n[QUERY]: {q}")
        results = similarity_search_with_metadata(q, k=2)
        assert len(results) > 0, f"No results found for {q}"
        for i, doc in enumerate(results):
            print(f"  Result #{i+1} (Score: {doc['score']:.4f})")
            print(f"    Source:  {doc['source']}")
            print(f"    Section: {doc['section']}")
            print(f"    Snippet: {doc['content'][:120].strip()}...")
    print("\n" + "=" * 70)
    print("All Retrieval Tests Passed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
