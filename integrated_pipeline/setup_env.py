"""
Helper script to create .env file from .env.example template.
"""
import os
from pathlib import Path


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        response = input(".env file already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing .env file.")
            return
    
    # Create .env.example content if it doesn't exist
    if not env_example.exists():
        env_template = """# LightRAG Configuration
# Fill in your actual values below

# API Configuration
OPENAI_API_KEY=your-openrouter-api-key-here
OPENAI_API_BASE=https://openrouter.ai/api/v1

# LLM Model Configuration
LLM_MODEL=google/gemini-2.0-flash-001
LLM_MAX_TOKENS=15000

# Embedding Model Configuration
EMBEDDING_MODEL=baai/bge-m3
EMBEDDING_DIM=1024

# Storage Configuration
WORKING_DIR=./dickens

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_KEY=

# LightRAG Storage Backends
GRAPH_STORAGE=Neo4JStorage
VECTOR_STORAGE=QdrantVectorDBStorage

# LightRAG Processing Configuration
CHUNK_TOKEN_SIZE=3000
CHUNK_OVERLAP_TOKEN_SIZE=300
COSINE_THRESHOLD=0.5
ENABLE_LLM_CACHE=true
ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT=true

# Concurrency Configuration
MAX_ASYNC=8
MAX_PARALLEL_INSERT=3
EMBEDDING_FUNC_MAX_ASYNC=16
EMBEDDING_BATCH_NUM=20

# Query Configuration
DEFAULT_QUERY_MODE=hybrid
DEFAULT_CHUNK_TOP_K=20
DEFAULT_MAX_TOTAL_TOKENS=20000

# Data Paths
JSONL_FILE_PATH=examples/openrouter/document1_paragraph_text.jsonl
CSV_DIR=examples/openrouter/llm_tables
FALLBACK_TEXT_FILE=./book.txt
"""
        env_example.write_text(env_template)
        print("Created .env.example file.")
    
    # Copy .env.example to .env
    if env_example.exists():
        env_content = env_example.read_text()
        env_file.write_text(env_content)
        print("Created .env file from .env.example")
        print("\n⚠️  IMPORTANT: Please edit .env file and fill in your actual values!")
        print("   - Set your OPENAI_API_KEY")
        print("   - Adjust other settings as needed")
    else:
        print("Error: .env.example not found and could not be created.")


if __name__ == "__main__":
    create_env_file()

