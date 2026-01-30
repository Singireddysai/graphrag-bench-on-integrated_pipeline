"""
Configuration module for LightRAG pipeline.
Centralizes all configuration management from environment variables.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration class for LightRAG pipeline."""
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    # LLM Model Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "15000"))
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "baai/bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    
    # Storage Configuration
    WORKING_DIR: str = os.getenv("WORKING_DIR", "./storage")
    WORKSPACE: Optional[str] = os.getenv("WORKSPACE", None)
    
    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "testpassword")
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_KEY: Optional[str] = os.getenv("QDRANT_KEY", None)
    
    # LightRAG Storage Backends
    GRAPH_STORAGE: str = os.getenv("GRAPH_STORAGE", "Neo4JStorage")
    VECTOR_STORAGE: str = os.getenv("VECTOR_STORAGE", "QdrantVectorDBStorage")
    
    # LightRAG Processing Configuration
    CHUNK_TOKEN_SIZE: int = int(os.getenv("CHUNK_TOKEN_SIZE", "3000"))
    CHUNK_OVERLAP_TOKEN_SIZE: int = int(os.getenv("CHUNK_OVERLAP_TOKEN_SIZE", "300"))
    COSINE_THRESHOLD: float = float(os.getenv("COSINE_THRESHOLD", "0.5"))
    ENABLE_LLM_CACHE: bool = os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"
    ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT: bool = (
        os.getenv("ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT", "true").lower() == "true"
    )
    
    # Concurrency Configuration
    MAX_ASYNC: int = int(os.getenv("MAX_ASYNC", "8"))
    MAX_PARALLEL_INSERT: int = int(os.getenv("MAX_PARALLEL_INSERT", "3"))
    EMBEDDING_FUNC_MAX_ASYNC: int = int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "16"))
    EMBEDDING_BATCH_NUM: int = int(os.getenv("EMBEDDING_BATCH_NUM", "20"))
    
    # Query Configuration
    DEFAULT_QUERY_MODE: str = os.getenv("DEFAULT_QUERY_MODE", "hybrid")
    DEFAULT_CHUNK_TOP_K: int = int(os.getenv("DEFAULT_CHUNK_TOP_K", "20"))
    DEFAULT_MAX_TOTAL_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOTAL_TOKENS", "20000"))
    
    # Data Paths
    JSONL_FILE_PATH: str = os.getenv("JSONL_FILE_PATH", "examples/openrouter/document1_paragraph_text.jsonl")
    CSV_DIR: str = os.getenv("CSV_DIR", "examples/openrouter/llm_tables")
    FALLBACK_TEXT_FILE: str = os.getenv("FALLBACK_TEXT_FILE", "./book.txt")
    
    @classmethod
    def setup_environment(cls) -> None:
        """Set up environment variables for LightRAG."""
        os.environ["OPENAI_API_BASE"] = cls.OPENAI_API_BASE
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        
        # Set concurrency environment variables
        os.environ.setdefault("MAX_ASYNC", str(cls.MAX_ASYNC))
        os.environ.setdefault("MAX_PARALLEL_INSERT", str(cls.MAX_PARALLEL_INSERT))
        os.environ.setdefault("EMBEDDING_FUNC_MAX_ASYNC", str(cls.EMBEDDING_FUNC_MAX_ASYNC))
        os.environ.setdefault("EMBEDDING_BATCH_NUM", str(cls.EMBEDDING_BATCH_NUM))
    
    @classmethod
    def ensure_working_dir(cls) -> Path:
        """Ensure working directory exists and return Path object."""
        working_dir = Path(cls.WORKING_DIR)
        working_dir.mkdir(parents=True, exist_ok=True)
        return working_dir
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in your .env file or environment variables."
            )
        return True

