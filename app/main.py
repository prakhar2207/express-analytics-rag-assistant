import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.core.config import settings
from app.api.routes import router as api_router
from app.ingestion.indexer import document_indexer
from app.services.vector_store import get_all_indexed_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure corpus is indexed if vector DB is empty
    logger.info("Initializing RAG Documentation Assistant...")
    try:
        current_docs = get_all_indexed_documents()
        if not current_docs:
            logger.info("Vector store is empty. Ingesting raw technical corpus from %s...", settings.RAW_DATA_DIR)
            res = document_indexer.index_directory()
            logger.info("Auto-ingestion completed: %d chunks indexed across %d files.", res["total_chunks"], res["indexed_files"])
        else:
            logger.info("Vector store already contains %d document source(s).", len(current_docs))
    except Exception as e:
        logger.warning("Auto-ingestion during startup encountered an issue: %s", e)
    yield
    # Shutdown
    logger.info("Shutting down RAG Documentation Assistant.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Production-ready RAG Assistant with a self-corrective LangGraph workflow. "
        "Evaluates document relevance, rewrites queries adaptively, searches vector store & web fallback, "
        "and generates answers grounded in technical documentation."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router)

@app.get("/", summary="Health and Service Info")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "status": "healthy",
        "docs_url": "/docs",
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "endpoints": {
            "query": "POST /query",
            "ingest": "POST /ingest",
            "documents": "GET /documents",
            "feedback": "POST /feedback"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
