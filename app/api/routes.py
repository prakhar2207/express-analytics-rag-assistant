import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Body, status
from pathlib import Path
import tempfile
import shutil

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    IngestUrlsRequest,
    IngestResponse,
    DocumentsListResponse,
    FeedbackRequest,
    FeedbackResponse
)
from app.graph.workflow import rag_app
from app.ingestion.indexer import document_indexer
from app.services.vector_store import get_all_indexed_documents
from app.services.feedback_service import feedback_service
from app.services.memory_service import memory_service
from app.core.config import settings

router = APIRouter()

@router.post("/query", response_model=QueryResponse, summary="Submit a technical question to the RAG assistant")
async def query_assistant(request: QueryRequest):
    """
    Executes the self-corrective LangGraph pipeline:
    1. Query Analysis & Expansion (with session coreference resolution)
    2. Retrieval from ChromaDB
    3. Document Grading (LLM evaluation)
    4. Adaptive query rewriting (if irrelevant) or Web Fallback
    5. Grounded Answer Generation with explicit citations
    6. Self-RAG Groundedness validation
    """
    start_time = time.perf_counter()
    
    # Retrieve conversation history if session_id provided
    history_str = memory_service.format_history_for_prompt(request.session_id) if request.session_id else ""

    initial_state = {
        "question": request.question,
        "query_type": None,
        "transformed_query": request.question,
        "documents": [],
        "relevant_docs": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": settings.MAX_RETRIES,
        "is_grounded": None,
        "web_fallback_used": False,
        "chat_history": history_str,
    }

    try:
        final_state = rag_app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG workflow: {str(e)}"
        )

    execution_time = round(time.perf_counter() - start_time, 3)
    generated_answer = final_state.get("generation", "No answer generated.")

    # Record in conversational memory if session_id is active
    if request.session_id:
        memory_service.add_interaction(request.session_id, request.question, generated_answer)

    # Format source citations
    citations_output = []
    for doc in final_state.get("relevant_docs", []):
        citations_output.append(SourceCitation(
            source=doc.get("source", "unknown"),
            section=doc.get("section", ""),
            snippet=doc.get("content", "")[:180] + "..." if len(doc.get("content", "")) > 180 else doc.get("content", "")
        ))

    return QueryResponse(
        question=request.question,
        answer=generated_answer,
        sources=citations_output,
        query_type=final_state.get("query_type"),
        retries_used=final_state.get("retry_count", 0),
        web_fallback_used=final_state.get("web_fallback_used", False),
        is_grounded=final_state.get("is_grounded", True),
        session_id=request.session_id,
        execution_time_sec=execution_time
    )

@router.post("/ingest", response_model=IngestResponse, summary="Ingest documents via file upload or URL list")
async def ingest_documents(
    files: Optional[List[UploadFile]] = File(None),
    payload: Optional[IngestUrlsRequest] = Body(None)
):
    """
    Ingests technical documentation from uploaded markdown/text files or external URLs.
    Uses section-aware chunking and embeds chunks into ChromaDB.
    """
    processed = []
    total_chunks = 0

    # Handle file uploads
    if files:
        for file in files:
            if not file.filename.endswith((".md", ".txt", ".markdown")):
                continue
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)
            
            try:
                # Use original filename for metadata tracking
                chunks = document_indexer.chunker.chunk_markdown_text(
                    tmp_path.read_text(encoding="utf-8"),
                    source_name=file.filename
                )
                if chunks:
                    document_indexer.vector_store.add_documents(chunks)
                total_chunks += len(chunks)
                processed.append(file.filename)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    # Handle URL ingestion
    if payload and payload.urls:
        for url in payload.urls:
            try:
                count = document_indexer.index_url(url)
                total_chunks += count
                processed.append(url)
            except Exception as e:
                # Continue other URLs if one fails
                continue

    if not processed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid files (.md, .txt) or URLs provided for ingestion."
        )

    return IngestResponse(
        message=f"Successfully ingested {len(processed)} document(s).",
        documents_processed=processed,
        total_chunks_added=total_chunks
    )

@router.get("/documents", response_model=DocumentsListResponse, summary="List all indexed documents in the corpus")
async def list_documents():
    """
    Inspect what technical documentation is currently indexed in the vector store,
    including chunk counts and documented sections.
    """
    docs_summary = get_all_indexed_documents()
    total_chunks = sum(d["chunks_count"] for d in docs_summary)
    
    return DocumentsListResponse(
        total_documents=len(docs_summary),
        total_chunks=total_chunks,
        documents=docs_summary
    )

@router.post("/feedback", response_model=FeedbackResponse, summary="Submit user feedback on an answer")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Records thumbs up/down user feedback and optional improvement comments.
    """
    if feedback.rating.lower() not in ("up", "down", "thumbs_up", "thumbs_down"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be 'up' or 'down'."
        )
        
    entry = feedback_service.record_feedback(
        query=feedback.query,
        answer=feedback.answer,
        rating=feedback.rating.lower(),
        comment=feedback.comment
    )
    
    return FeedbackResponse(
        message="Feedback successfully recorded. Thank you!",
        feedback_id=entry["id"],
        status="success"
    )
