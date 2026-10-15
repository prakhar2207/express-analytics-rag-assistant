import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Technical Documentation RAG Assistant"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    LLM_PROVIDER: str = "groq"  # "groq", "openai", "google"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # API Keys
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    
    # Embedding Settings
    EMBEDDING_PROVIDER: str = "local"  # "local" (sentence-transformers) or "openai"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Storage Paths
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    RAW_DATA_DIR: str = str(BASE_DIR / "data" / "raw")
    FEEDBACK_FILE_PATH: str = str(BASE_DIR / "data" / "feedback.json")
    
    # Graph & Retrieval Config
    TOP_K: int = 4
    MAX_RETRIES: int = 2
    
    # Chunking Strategy Config
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
