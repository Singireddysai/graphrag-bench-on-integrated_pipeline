# LightRAG Pipeline - Modular Structure

A modular, production-ready implementation of LightRAG for document indexing and querying.

## Project Structure

```
.
├── config.py              # Centralized configuration management
├── lightrag_utils.py      # LightRAG initialization utilities
├── utils.py               # Document loading and processing utilities
├── prompts.py             # Custom prompt definitions
├── train_script.py        # Script to train/index documents
├── query_script.py        # Script to query the RAG system
├── unified_interface.py   # Streamlit web interface (Extract → Index → Query)
├── .env                   # Environment variables (create from .env.example)
```
## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root

Required environment variables:

```env
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

# Neo4j Configuration (if using Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword

# Qdrant Configuration (if using Qdrant)
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
JSONL_FILE_PATH=...
CSV_DIR=...
FALLBACK_TEXT_FILE=...

### 3. Ensure Storage Services are Running

If using Neo4j and Qdrant, make sure they're running:

```bash
# Neo4j (example with Docker)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword \
  neo4j:latest

# Qdrant (example with Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

## Usage

### Web Interface (Recommended)

Launch the unified Streamlit interface for a complete pipeline:

```bash
streamlit run unified_interface.py
```

The interface provides three tabs:
1. **Extract**: Upload PDFs and extract text, tables, and images
2. **Index**: Train the knowledge graph and vector database from extracted content
3. **Query**: Ask questions about your documents interactively

### Command-Line Interface

#### Training (Indexing Documents)

Run the training script to load and index your documents:

```bash
python train_script.py
```

The script will:
1. Load documents from JSONL files
2. Load CSV tables from the specified directory
3. Insert all documents into the RAG system
4. Build the knowledge graph and vector indices

#### Querying

Run the query script to query your indexed documents:

```bash
python query_script.py
```

The script will:
1. Load the existing RAG instance
2. Run predefined queries
3. Display results with proper citations

## Module Documentation

### `config.py`

Centralized configuration management. All environment variables are loaded here and exposed as class attributes.

**Key Features:**
- Type-safe configuration access
- Validation of required settings
- Environment setup helpers

### `lightrag_utils.py`

Utilities for initializing and managing LightRAG instances.

**Key Functions:**
- `initialize_rag()`: Create and configure a LightRAG instance
- `create_embedding_func()`: Set up embedding function
- `create_llm_func()`: Set up LLM function
- `test_embedding_function()`: Verify embedding function works

### `utils.py`

Document processing and loading utilities.

**Key Functions:**
- `load_jsonl_documents()`: Load documents from JSONL files
- `load_csv_tables()`: Load CSV tables from directory
- `load_all_documents()`: Load all available documents
- `convert_csv_to_text()`: Convert CSV to text format
- `clear_existing_data()`: Clear old RAG data

### `prompts.py`

Custom prompt definitions for query responses.

**Key Features:**
- Custom RAG response prompt with citation formatting
- Reference format enforcement
- User prompt helpers

### `unified_interface.py`

Streamlit web interface providing a complete pipeline for document processing.

**Features:**
- PDF extraction with configurable options (text, tables, images)
- Interactive indexing/training interface
- Real-time query interface with adjustable parameters
- Query history and result display
- Integrated with existing codebase utilities (`config.py`, `lightrag_utils.py`, `utils.py`)

## Best Practices

1. **Configuration Management**: All configuration is centralized in `config.py` and loaded from `.env` file
2. **Error Handling**: Proper error handling with informative messages
3. **Type Hints**: All functions include type hints for better code clarity
4. **Modularity**: Code is split into logical modules for maintainability
5. **Documentation**: All modules and functions are documented
6. **Environment Variables**: Sensitive data is stored in `.env` file (not committed to git)

## Customization

### Changing LLM Model

Edit `.env` file:
```env
LLM_MODEL=openai/gpt-4o-mini
```

### Changing Embedding Model

Edit `.env` file:
```env
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

### Changing Storage Backends

Edit `.env` file:
```env
GRAPH_STORAGE=NetworkXStorage  # For in-memory graph
VECTOR_STORAGE=FaissVectorDBStorage  # For FAISS vector storage
```

### Customizing Prompts

Edit `prompts.py` to modify the RAG response prompt format.

## Troubleshooting

### Configuration Errors

If you see "OPENAI_API_KEY is not set":
1. Ensure `.env` file exists in the project root
2. Check that `OPENAI_API_KEY` is set in `.env`
3. Verify the key is valid

### Storage Connection Errors

If Neo4j or Qdrant connection fails:
1. Verify services are running
2. Check connection strings in `.env`
3. Verify credentials are correct

### Import Errors

If you see import errors:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Verify you're in the correct directory
3. Check Python version (requires 3.8+)

## License

This project uses LightRAG, which is licensed under MIT License.

