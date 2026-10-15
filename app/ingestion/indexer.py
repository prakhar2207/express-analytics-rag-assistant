import os
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from app.core.config import settings
from app.ingestion.chunker import TechnicalDocumentChunker
from app.services.vector_store import get_vector_store

class DocumentIndexer:
    def __init__(self):
        self.chunker = TechnicalDocumentChunker()
        self.vector_store = get_vector_store()

    def index_file(self, file_path: Path) -> int:
        """
        Chunks and indexes a single markdown or text file into Chroma.
        Returns the number of chunks added.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        chunks = self.chunker.chunk_file(file_path)
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)

    def index_directory(self, dir_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Indexes all markdown files in the specified directory.
        """
        target_dir = dir_path or Path(settings.RAW_DATA_DIR)
        if not target_dir.exists():
            return {"indexed_files": 0, "total_chunks": 0, "files": []}

        md_files = list(target_dir.glob("*.md")) + list(target_dir.glob("*.txt"))
        total_chunks = 0
        processed_files = []

        for f in md_files:
            count = self.index_file(f)
            total_chunks += count
            processed_files.append({"file": f.name, "chunks": count})

        return {
            "indexed_files": len(processed_files),
            "total_chunks": total_chunks,
            "files": processed_files
        }

    def index_url(self, url: str) -> int:
        """
        Fetches markdown/text content from a URL and indexes it.
        """
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        
        source_name = url.split("/")[-1] or "web_doc.md"
        if not source_name.endswith((".md", ".txt")):
            source_name += ".md"
            
        chunks = self.chunker.chunk_markdown_text(response.text, source_name=source_name)
        for chunk in chunks:
            chunk.metadata["url"] = url
            
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)

document_indexer = DocumentIndexer()
