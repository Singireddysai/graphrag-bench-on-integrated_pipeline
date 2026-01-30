# GraphRAG-Benchmark Integration Setup Guide

This guide explains how to set up and run the integrated_pipeline RAG system with GraphRAG-Benchmark evaluation.

## Prerequisites

1. Docker and Docker Compose installed
2. Python 3.8+ with virtual environment
3. OpenAI API key (or OpenRouter API key)
4. GraphRAG-Benchmark repository cloned

## Step 1: Start Docker Services

The integrated_pipeline uses Neo4j and Qdrant as storage backends. Start them using Docker Compose:

```bash
cd integrated_pipeline
docker-compose up -d
```

Verify services are running:
```bash
docker-compose ps
```

You should see:
- `graphrag_neo4j` running on ports 7474 (HTTP) and 7687 (Bolt)
- `graphrag_qdrant` running on ports 6333 (HTTP) and 6334 (gRPC)

Access Neo4j browser at: http://localhost:7474
Access Qdrant dashboard at: http://localhost:6333/dashboard

## Step 2: Configure Environment Variables

Create a `.env` file in the `integrated_pipeline` directory:

```env
# API Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1

# LLM Model Configuration
LLM_MODEL=gpt-4o-mini
LLM_MAX_TOKENS=15000

# Embedding Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# Storage Configuration
WORKING_DIR=./storage

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
```

## Step 3: Install Dependencies

Activate your virtual environment and install dependencies:

```bash
# Activate virtual environment
.\graphrag_env\Scripts\Activate.ps1  # Windows PowerShell
# or
source graphrag_env/bin/activate  # Linux/Mac

# Install integrated_pipeline dependencies
cd integrated_pipeline
pip install -r requirements.txt

# Make sure LightRAG is installed
pip install lightrag
```

## Step 4: Index GraphRAG-Benchmark Corpus

Index the corpus data into the integrated_pipeline:

```bash
# Index medical subset
python index_graphrag_benchmark.py --subset medical

# Index novel subset
python index_graphrag_benchmark.py --subset novel

# Index specific corpus
python index_graphrag_benchmark.py --subset medical --corpus_name "corpus_1"
```

This will:
1. Load corpus data from `GraphRAG-Benchmark/Datasets/Corpus/`
2. Index each corpus document into LightRAG
3. Store the knowledge graph in Neo4j and vectors in Qdrant
4. Save data to `storage/graphrag_benchmark/{subset}/`

## Step 5: Run GraphRAG-Benchmark Queries

Generate predictions for the benchmark questions:

```bash
# Process all questions for medical subset
python run_graphrag_benchmark.py --subset medical

# Process all questions for novel subset
python run_graphrag_benchmark.py --subset novel

# Sample a subset of questions (for testing)
python run_graphrag_benchmark.py --subset medical --sample 10

# Process specific corpus
python run_graphrag_benchmark.py --subset medical --corpus_name "corpus_1"

# Customize query parameters
python run_graphrag_benchmark.py \
  --subset medical \
  --query_mode hybrid \
  --chunk_top_k 20 \
  --max_total_tokens 20000
```

This will:
1. Load questions from `GraphRAG-Benchmark/Datasets/Questions/`
2. Query the indexed RAG system for each question
3. Generate predictions in the format expected by GraphRAG-Benchmark
4. Save results to `GraphRAG-Benchmark/results/integrated_pipeline/{subset}/predictions_{subset}.json`

## Step 6: Run Evaluation

Evaluate the generated predictions using GraphRAG-Benchmark evaluation scripts:

```bash
cd ../GraphRAG-Benchmark

# Generation evaluation
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/integrated_pipeline/medical/predictions_medical.json \
  --output_file ./results/integrated_pipeline/medical/evaluation_results.json

# Retrieval evaluation
python -m Evaluation.retrieval_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/integrated_pipeline/medical/predictions_medical.json \
  --output_file ./results/integrated_pipeline/medical/retrieval_results.json

# Indexing evaluation
python -m Evaluation.indexing_eval \
  --framework lightrag \
  --base_path ../integrated_pipeline/storage/graphrag_benchmark/medical \
  --folder_name graph_store \
  --output ./results/integrated_pipeline/medical/indexing_metrics.txt
```

## Important Notes

### Context Extraction

The GraphRAG-Benchmark evaluation requires context information from retrieved documents. According to the GraphRAG-Benchmark Examples README, you may need to modify LightRAG source code to return context:

1. In `lightrag/operate.py`, modify `kg_query` to return context:
```python
async def kg_query(...) -> tuple[str, str] | tuple[AsyncIterator[str], str]:
    return response, context
```

2. In `lightrag/lightrag.py`, modify `aquery` to return context:
```python
async def aquery(...):
    ...
    if param.mode in ["local", "global", "hybrid"]:
        response, context = await kg_query(...)
    ...
    return response, context
```

**Note**: The current `run_graphrag_benchmark.py` script includes a workaround, but for accurate context extraction, you should modify LightRAG as described above.

### Storage Backends

The integrated_pipeline uses:
- **Neo4j**: For graph storage (entities and relationships)
- **Qdrant**: For vector storage (embeddings)

Make sure both services are running before indexing or querying.

### Working Directories

- Indexed data: `integrated_pipeline/storage/graphrag_benchmark/{subset}/`
- Predictions: `GraphRAG-Benchmark/results/integrated_pipeline/{subset}/`
- Evaluation results: `GraphRAG-Benchmark/results/integrated_pipeline/{subset}/`

## Troubleshooting

### Docker Services Not Starting

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart

# Stop and remove containers
docker-compose down
docker-compose up -d
```

### Neo4j Connection Issues

- Verify Neo4j is running: `docker ps | grep neo4j`
- Check connection string in `.env`: `NEO4J_URI=bolt://localhost:7687`
- Verify credentials: `NEO4J_USERNAME=neo4j`, `NEO4J_PASSWORD=testpassword`

### Qdrant Connection Issues

- Verify Qdrant is running: `docker ps | grep qdrant`
- Check URL in `.env`: `QDRANT_URL=http://localhost:6333`
- Access dashboard: http://localhost:6333/dashboard

### API Key Issues

- Verify `OPENAI_API_KEY` is set in `.env`
- Check API key is valid
- For OpenRouter, use `OPENAI_API_BASE=https://openrouter.ai/api/v1`

### Import Errors

- Ensure virtual environment is activated
- Install all dependencies: `pip install -r requirements.txt`
- Verify LightRAG is installed: `pip install lightrag`

### Context Not Returned

If context is empty in predictions:
1. Modify LightRAG source code as described above
2. Or check if LightRAG version supports context extraction
3. Verify query mode is set correctly (hybrid mode recommended)

## Complete Workflow Example

```bash
# 1. Start Docker services
cd integrated_pipeline
docker-compose up -d

# 2. Configure .env file (add your API key)

# 3. Index corpus
python index_graphrag_benchmark.py --subset medical

# 4. Run queries
python run_graphrag_benchmark.py --subset medical --sample 10

# 5. Evaluate results
cd ../GraphRAG-Benchmark
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/integrated_pipeline/medical/predictions_medical.json \
  --output_file ./results/integrated_pipeline/medical/evaluation_results.json
```

## Next Steps

1. Review evaluation results in the output JSON files
2. Compare metrics across different query modes
3. Experiment with different chunk_top_k and max_total_tokens values
4. Run full evaluation on both medical and novel subsets

