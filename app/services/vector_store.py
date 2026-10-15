import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from app.core.config import settings

_vector_store_instance: Optional[Chroma] = None

def get_embeddings():
    """
    Returns embedding function based on configuration.
    Defaults to HuggingFace BAAI or MiniLM (runs 100% locally with no API key).
    """
    if settings.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL or "text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )
    else:
        # Default local embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

def get_vector_store() -> Chroma:
    """
    Returns singleton persistent Chroma vector store.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        embeddings = get_embeddings()
        _vector_store_instance = Chroma(
            collection_name="technical_docs",
            embedding_function=embeddings,
            persist_directory=str(persist_dir)
        )
    return _vector_store_instance

def similarity_search_with_metadata(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Perform similarity search and format documents with standard metadata.
    """
    top_k = k or settings.TOP_K
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=top_k)
    
    formatted = []
    for doc, distance in results:
        # Normalize distance into intuitive similarity score
        similarity = 1.0 / (1.0 + float(distance))
        formatted.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "source": doc.metadata.get("source", "unknown"),
            "section": doc.metadata.get("section_path") or doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or doc.metadata.get("title", ""),
            "score": similarity
        })
    return formatted

def get_all_indexed_documents() -> List[Dict[str, Any]]:
    """
    Extract aggregated summary of indexed documents in Chroma collection.
    """
    store = get_vector_store()
    data = store.get()
    
    sources_map: Dict[str, Dict[str, Any]] = {}
    
    if data and "metadatas" in data and data["metadatas"]:
        for meta in data["metadatas"]:
            if not meta:
                continue
            source = meta.get("source", "unknown")
            section = meta.get("Header 2") or meta.get("Header 1") or ""
            if source not in sources_map:
                sources_map[source] = {
                    "source": source,
                    "chunks_count": 0,
                    "sections": set()
                }
            sources_map[source]["chunks_count"] += 1
            if section:
                sources_map[source]["sections"].add(section)

    result = []
    for s, info in sources_map.items():
        result.append({
            "source": s,
            "chunks_count": info["chunks_count"],
            "sections": sorted(list(info["sections"]))
        })
    return result
