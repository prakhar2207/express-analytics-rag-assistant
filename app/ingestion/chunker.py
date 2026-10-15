import os
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.core.config import settings

class TechnicalDocumentChunker:
    """
    Two-stage chunking strategy specialized for technical documentation:
    1. MarkdownHeaderTextSplitter: preserves conceptual hierarchy and section titles in metadata.
    2. RecursiveCharacterTextSplitter: ensures chunks fit embedding window while respecting code blocks.
    """
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Split headers to capture document hierarchy
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        # Recursive splitter respecting markdown code blocks and paragraphs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n```", "```\n", "\n\n", "\n", " ", ""]
        )

    def chunk_markdown_text(self, text: str, source_name: str) -> List[Document]:
        """
        Takes raw markdown content and returns chunked documents with preserved metadata.
        """
        # Stage 1: Split along markdown headers
        header_splits = self.markdown_splitter.split_text(text)
        
        # Stage 2: Further split oversized sections
        final_splits = self.text_splitter.split_documents(header_splits)
        
        # Augment metadata with source and chunk index
        for idx, doc in enumerate(final_splits):
            doc.metadata["source"] = source_name
            doc.metadata["chunk_index"] = idx
            
            # Construct breadcrumb title for citation grounding
            breadcrumb_parts = []
            for header_key in ["Header 1", "Header 2", "Header 3"]:
                if header_key in doc.metadata:
                    breadcrumb_parts.append(doc.metadata[header_key])
            doc.metadata["section_path"] = " > ".join(breadcrumb_parts) if breadcrumb_parts else "General"
            
        return final_splits

    def chunk_file(self, file_path: Path) -> List[Document]:
        """
        Loads a file from disk and splits it.
        """
        text = file_path.read_text(encoding="utf-8")
        return self.chunk_markdown_text(text, source_name=file_path.name)
