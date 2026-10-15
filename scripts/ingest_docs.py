"""
Standalone CLI script to chunk and ingest documentation into ChromaDB.
Usage:
    python scripts/ingest_docs.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.indexer import document_indexer
from app.services.vector_store import get_all_indexed_documents

def main():
    print("=" * 60)
    print("Starting Technical Documentation Ingestion Pipeline")
    print("=" * 60)
    
    result = document_indexer.index_directory()
    print(f"\nIngestion Complete!")
    print(f"Files Indexed: {result['indexed_files']}")
    print(f"Total Chunks Created & Stored: {result['total_chunks']}")
    for f in result["files"]:
        print(f"  - {f['file']}: {f['chunks']} chunks")
        
    print("\nCurrent Vector Store Corpus Summary:")
    docs = get_all_indexed_documents()
    for d in docs:
        print(f"  * {d['source']} ({d['chunks_count']} chunks)")
        for sec in d["sections"][:5]:
            print(f"      - {sec}")
        if len(d["sections"]) > 5:
            print(f"      - ... and {len(d['sections']) - 5} more sections")
    print("=" * 60)

if __name__ == "__main__":
    main()
