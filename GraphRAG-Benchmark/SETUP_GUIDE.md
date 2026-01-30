# GraphRAG-Benchmark Setup Guide

## ✅ Setup Complete

The GraphRAG-Benchmark evaluation environment has been successfully set up in your local system.

### What's Been Done

1. **Virtual Environment Created**: `graphrag_env` (Python venv)
2. **Repository Cloned**: GraphRAG-Benchmark from GitHub
3. **Dependencies Installed**: All packages from `requirements.txt` are installed
4. **Datasets Available**: Medical and Novel datasets are in `Datasets/` folder

### Repository Structure

```
GraphRAG-Benchmark/
├── Datasets/              # Corpus and Questions datasets
│   ├── Corpus/
│   │   ├── medical.json/parquet
│   │   └── novel.json/parquet
│   └── Questions/
│       ├── medical_questions.json/parquet
│       └── novel_questions.json/parquet
├── Examples/              # Example scripts for different GraphRAG frameworks
│   ├── run_lightrag.py
│   ├── run_fast-graphrag.py
│   ├── run_hipporag2.py
│   └── run_digimon.py
├── Evaluation/            # Evaluation scripts and metrics
│   ├── generation_eval.py
│   ├── retrieval_eval.py
│   └── indexing_eval.py
└── requirements.txt       # Python dependencies (installed)
```

## 🔑 API Keys Configuration

**IMPORTANT**: You need to set the following environment variables before running evaluations:

### For OpenAI API:
```powershell
$env:LLM_API_KEY="your_openai_api_key_here"
# OR
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

### For Ollama (Local):
If using Ollama locally, make sure Ollama is running at `http://localhost:11434` (default)

## 🚀 Next Steps

### 1. Activate Virtual Environment

```powershell
.\graphrag_env\Scripts\Activate.ps1
cd GraphRAG-Benchmark
```

### 2. Install GraphRAG Framework (Choose One)

The benchmark supports multiple GraphRAG frameworks. You need to install the one you want to evaluate:

#### Option A: LightRAG (Recommended for starting)
```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .
cd ..
```

**Note**: According to the README, you may need to modify LightRAG source code to enable context extraction. See `Examples/README.md` for details.

#### Option B: Fast-GraphRAG
```bash
pip install fast-graphrag
```

**Note**: You may need to adapt the library for HuggingFace Embedding support. See `Examples/README.md` for details.

#### Option C: HippoRAG2
```bash
git clone https://github.com/facebookresearch/hipporag.git
cd hipporag
pip install -e .
cd ..
```

**Note**: You may need to add BGE Embedding model support. See `Examples/README.md` for details.

#### Option D: DIGIMON
```bash
git clone https://github.com/JayLZhou/GraphRAG.git
# Follow DIGIMON installation instructions
```

### 3. Run Indexing and Inference

Example for LightRAG:

```powershell
# Set API key
$env:LLM_API_KEY="your_api_key_here"

# Run indexing and inference
python Examples/run_lightrag.py `
  --subset medical `
  --mode API `
  --base_dir ./Examples/lightrag_workspace `
  --model_name gpt-4o-mini `
  --embed_model bge-base-en `
  --retrieve_topk 5 `
  --llm_base_url https://api.openai.com/v1
```

**Parameters:**
- `--subset`: Choose `medical` or `novel`
- `--mode`: `API` (OpenAI) or `ollama` (local)
- `--model_name`: LLM model identifier (e.g., `gpt-4o-mini`, `gpt-4o`)
- `--embed_model`: Embedding model (e.g., `bge-base-en`, `bge-large-en-v1.5`)
- `--retrieve_topk`: Number of top documents to retrieve
- `--sample`: (Optional) Limit number of questions to process
- `--llm_base_url`: API endpoint URL

### 4. Run Evaluation

After generating predictions, evaluate them:

#### Generation Evaluation:
```powershell
python -m Evaluation.generation_eval `
  --mode API `
  --model gpt-4o-mini `
  --base_url https://api.openai.com/v1 `
  --embedding_model BAAI/bge-large-en-v1.5 `
  --data_file ./results/lightrag/medical/predictions_medical.json `
  --output_file ./results/evaluation_results.json
```

#### Retrieval Evaluation:
```powershell
python -m Evaluation.retrieval_eval `
  --mode API `
  --model gpt-4o-mini `
  --base_url https://api.openai.com/v1 `
  --embedding_model BAAI/bge-large-en-v1.5 `
  --data_file ./results/lightrag/medical/predictions_medical.json `
  --output_file ./results/retrieval_results.json
```

#### Indexing Evaluation:
```powershell
python -m Evaluation.indexing_eval `
  --framework lightrag `
  --base_path ./Examples/lightrag_workspace `
  --folder_name graph_store `
  --output ./results/indexing_metrics.txt
```

## 📊 Evaluation Metrics

### Generation Metrics (by Question Type):
- **Fact Retrieval**: ROUGE-L, Answer Correctness
- **Complex Reasoning**: ROUGE-L, Answer Correctness
- **Contextual Summarization**: Answer Correctness, Coverage
- **Creative Generation**: Answer Correctness, Coverage, Faithfulness

### Retrieval Metrics:
- **Context Relevancy**: Relevance of retrieved contexts to questions
- **Evidence Recall**: How well retrieved contexts cover ground truth evidence

### Indexing Metrics:
- Graph structure metrics (density, connectivity, clustering coefficients)
- Entity/relationship distributions

## 📝 Output Format

The evaluation scripts expect predictions in this JSON format:

```json
{
  "id": "question_id",
  "question": "question text",
  "source": "corpus_name",
  "context": ["context1", "context2", ...],
  "evidence": ["evidence1", "evidence2", ...],
  "question_type": "fact_retrieval|complex_reasoning|contextual_summarize|creative_generation",
  "generated_answer": "predicted answer",
  "ground_truth": "correct answer"
}
```

## 🔍 Troubleshooting

1. **Import Errors**: Make sure the virtual environment is activated
2. **API Key Issues**: Verify environment variables are set correctly
3. **Framework Not Found**: Install the specific GraphRAG framework you want to use
4. **Dataset Loading**: Datasets are already in the repository, no download needed

## 📚 Additional Resources

- Main README: `README.md`
- Examples Guide: `Examples/README.md`
- Evaluation Guide: `Evaluation/README.md`
- Paper: https://arxiv.org/abs/2506.05690
- Leaderboard: https://graphrag-bench.github.io/

## ⚠️ Important Notes

1. **Separate Environments**: The README recommends using separate Conda environments for each framework to avoid dependency conflicts. However, you can use this venv if you're only testing one framework at a time.

2. **Framework Modifications**: Some frameworks (LightRAG, fast-graphrag, hipporag2) require source code modifications to work with this benchmark. See `Examples/README.md` for detailed instructions.

3. **API Costs**: Running evaluations will make API calls. Monitor your usage if using paid APIs.

4. **Local Models**: You can use Ollama for local inference to avoid API costs. Make sure Ollama is installed and running.

---

**Setup Date**: $(Get-Date -Format "yyyy-MM-dd")
**Python Version**: Check with `python --version`
**Virtual Environment**: `graphrag_env`

