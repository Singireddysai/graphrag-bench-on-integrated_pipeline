# Complete Evaluation Guide: Integrated Pipeline with GraphRAG-Benchmark

This guide provides step-by-step instructions to replicate the complete evaluation process: indexing the corpus, generating predictions, and running evaluations.

## Prerequisites

1. **Virtual Environment**: Ensure you have the virtual environment activated
   ```powershell
   .\graphrag_env\Scripts\Activate.ps1
   ```

2. **Docker Services**: Ensure Neo4j and Qdrant are running
   ```powershell
   cd integrated_pipeline
   docker-compose up -d
   ```

3. **API Keys**: Set up your `.env` file in `integrated_pipeline/.env` with:
   ```
   OPENAI_API_KEY=your_openrouter_api_key_here
   OPENAI_API_BASE=https://openrouter.ai/api/v1
   LLM_MODEL=openai/gpt-4o-mini
   EMBEDDING_MODEL=baai/bge-m3
   ```

---

## Step 1: Index the Corpus

Index the GraphRAG-Benchmark corpus into the integrated_pipeline knowledge base.

### Command:
```powershell
cd integrated_pipeline
python index_graphrag_benchmark.py --subset medical
```

### Parameters:
- `--subset`: Choose `medical` or `novel`
- `--corpus_name` (optional): Specific corpus name to index
- `--chunk_size` (optional): Chunk size for document processing (default: 1000)
- `--chunk_overlap` (optional): Overlap between chunks (default: 200)

### What it does:
- Loads documents from `GraphRAG-Benchmark/Datasets/Corpus/{subset}/`
- Processes and chunks the documents
- Indexes them into Neo4j (graph) and Qdrant (vector store)
- Saves the indexed data to `storage/graphrag_benchmark/{subset}/`

### Expected Output:
```
[OK] Successfully indexed corpus: medical
Total documents processed: X
Total chunks created: Y
```

---

## Step 2: Generate Predictions

Query the indexed knowledge base and generate answers for benchmark questions.

### Command:
```powershell
cd integrated_pipeline
python run_graphrag_benchmark.py --subset medical
```

### Parameters:
- `--subset`: Choose `medical` or `novel` (required)
- `--corpus_name` (optional): Specific corpus name to process
- `--sample` (optional): Number of questions to sample (e.g., `--sample 5` for testing, omit for all)
- `--query_mode`: Query mode for LightRAG (default: `hybrid`)
  - Options: `naive`, `local`, `global`, `hybrid`
- `--chunk_top_k`: Number of top chunks to retrieve (default: 20)
- `--max_total_tokens`: Maximum total tokens for response (default: 20000)

### Examples:

**Process all questions:**
```powershell
python run_graphrag_benchmark.py --subset medical
```

**Process only 5 questions (for testing):**
```powershell
python run_graphrag_benchmark.py --subset medical --sample 5
```

**Process specific corpus:**
```powershell
python run_graphrag_benchmark.py --subset medical --corpus_name "Medical"
```

### What it does:
- Loads questions from `GraphRAG-Benchmark/Datasets/Questions/{subset}_questions.parquet`
- Queries the integrated_pipeline RAG system for each question
- Generates answers using the indexed knowledge base
- Saves predictions to `GraphRAG-Benchmark/results/integrated_pipeline/{subset}/predictions_{subset}.json`

### Expected Output:
```
Processing GraphRAG-Benchmark Questions: medical
Found X question(s)
[1/X] Processing: Medical-xxxxx
  [OK] Generated answer (XXX chars)
...
Saved X prediction(s) to: .../predictions_medical.json
```

---

## Step 3: Run Evaluation

Evaluate the generated predictions against ground truth answers.

### Command:
```powershell
cd GraphRAG-Benchmark
$env:LLM_API_KEY = "your_openrouter_api_key_here"
$env:HUGGINGFACE_API_KEY = "your_huggingface_key_here"
python -m Evaluation.generation_eval --mode API --model openai/gpt-4o-mini --base_url https://openrouter.ai/api/v1 --embedding_model BAAI/bge-large-en-v1.5 --data_file ./results/integrated_pipeline/medical/predictions_medical.json --output_file ./results/integrated_pipeline/medical/evaluation_results.json
```

### Parameters:
- `--mode`: Evaluation mode (use `API` for OpenRouter/OpenAI)
- `--model`: LLM model name (use `openai/gpt-4o-mini` for OpenRouter)
- `--base_url`: API base URL (use `https://openrouter.ai/api/v1` for OpenRouter)
- `--embedding_model`: Embedding model for semantic similarity (default: `BAAI/bge-large-en-v1.5`)
- `--data_file`: Path to predictions JSON file
- `--output_file`: Path to save evaluation results

### Environment Variables:
Before running, set these in PowerShell:
```powershell
$env:LLM_API_KEY = "sk-or-v1-your-key-here"
$env:HUGGINGFACE_API_KEY = "hf_your-key-here"
```

### What it does:
- Loads predictions from the specified JSON file
- For each prediction:
  - Calculates ROUGE score (text overlap metric)
  - Uses LLM to assess answer correctness
  - Computes semantic similarity using embeddings
- Groups results by question type
- Saves evaluation results to the specified output file

### Expected Output:
```
Loading evaluation data from ./results/integrated_pipeline/medical/predictions_medical.json...
==================================================
Evaluating question type: Fact Retrieval
==================================================
Starting evaluation of X samples...
[OK] Completed sample 1/X - XX.X%
...
Results for Fact Retrieval:
  rouge_score: 0.XXXX
  answer_correctness: 0.XXXX
Saving results to ./results/integrated_pipeline/medical/evaluation_results.json...
Evaluation complete.
```

---

## Complete Workflow Example

Here's the complete workflow from start to finish:

### 1. Activate Virtual Environment
```powershell
.\graphrag_env\Scripts\Activate.ps1
```

### 2. Start Docker Services
```powershell
cd integrated_pipeline
docker-compose up -d
```

### 3. Index the Corpus
```powershell
cd integrated_pipeline
python index_graphrag_benchmark.py --subset medical
```

### 4. Generate Predictions (Test with 5 questions)
```powershell
python run_graphrag_benchmark.py --subset medical --sample 5
```

### 5. Run Evaluation
```powershell
cd ..\GraphRAG-Benchmark
$env:LLM_API_KEY = "sk-or-v1-your-key-here"
$env:HUGGINGFACE_API_KEY = "hf_your-key-here"
python -m Evaluation.generation_eval --mode API --model openai/gpt-4o-mini --base_url https://openrouter.ai/api/v1 --embedding_model BAAI/bge-large-en-v1.5 --data_file ./results/integrated_pipeline/medical/predictions_medical.json --output_file ./results/integrated_pipeline/medical/evaluation_results.json
```

### 6. View Results
Check the evaluation results:
```powershell
cat .\results\integrated_pipeline\medical\evaluation_results.json
```

---

## File Locations

### Input Files:
- **Corpus**: `GraphRAG-Benchmark/Datasets/Corpus/{subset}/`
- **Questions**: `GraphRAG-Benchmark/Datasets/Questions/{subset}_questions.parquet`

### Output Files:
- **Indexed Data**: `integrated_pipeline/storage/graphrag_benchmark/{subset}/`
- **Predictions**: `GraphRAG-Benchmark/results/integrated_pipeline/{subset}/predictions_{subset}.json`
- **Evaluation Results**: `GraphRAG-Benchmark/results/integrated_pipeline/{subset}/evaluation_results.json`

---

## Troubleshooting

### Issue: UnicodeEncodeError
**Solution**: The evaluation script has been fixed to use ASCII characters instead of Unicode. If you encounter this, ensure you're using the updated `generation_eval.py`.

### Issue: API Key Authentication Error
**Solution**: 
- Ensure you're using an OpenRouter API key (starts with `sk-or-v1-`)
- Use the OpenRouter base URL: `https://openrouter.ai/api/v1`
- Set the model name as `openai/gpt-4o-mini` (OpenRouter format)

### Issue: Docker Services Not Running
**Solution**:
```powershell
cd integrated_pipeline
docker-compose up -d
docker-compose ps  # Check status
```

### Issue: No Questions Found
**Solution**: 
- Check that the questions file exists: `GraphRAG-Benchmark/Datasets/Questions/{subset}_questions.parquet`
- Verify the `--subset` parameter matches the file name

### Issue: Working Directory Not Found
**Solution**: Run the indexing step first (`index_graphrag_benchmark.py`) before generating predictions.

---

## Advanced Usage

### Process Multiple Question Types
The evaluation automatically groups by question type. To evaluate specific types, filter the predictions file first.

### Custom Query Modes
Experiment with different query modes:
- `naive`: Simple retrieval
- `local`: Local graph traversal
- `global`: Global graph analysis
- `hybrid`: Combination of local and global (recommended)

### Batch Processing
For large datasets, consider processing in batches:
```powershell
# Process first 100 questions
python run_graphrag_benchmark.py --subset medical --sample 100

# Process next 100 (requires script modification or manual filtering)
```

---

## Notes

- **Indexing** only needs to be done once per corpus
- **Predictions** can be regenerated without re-indexing
- **Evaluation** can be run multiple times on the same predictions file
- The evaluation script uses the LLM as a judge, so API costs apply
- For testing, use `--sample 5` to process only 5 questions quickly

---

## Quick Reference Commands

```powershell
# Index
cd integrated_pipeline
python index_graphrag_benchmark.py --subset medical

# Query (all questions)
python run_graphrag_benchmark.py --subset medical

# Query (sample)
python run_graphrag_benchmark.py --subset medical --sample 5

# Evaluate
cd ..\GraphRAG-Benchmark
$env:LLM_API_KEY = "your-key"
$env:HUGGINGFACE_API_KEY = "your-key"
python -m Evaluation.generation_eval --mode API --model openai/gpt-4o-mini --base_url https://openrouter.ai/api/v1 --embedding_model BAAI/bge-large-en-v1.5 --data_file ./results/integrated_pipeline/medical/predictions_medical.json --output_file ./results/integrated_pipeline/medical/evaluation_results.json
```

