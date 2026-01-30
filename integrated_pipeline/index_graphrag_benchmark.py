"""
Script to index GraphRAG-Benchmark corpus into integrated_pipeline.
Loads corpus data from GraphRAG-Benchmark Datasets and indexes them.
"""
import asyncio
import os
import sys
import argparse
from pathlib import Path
from datasets import load_dataset
from lightrag import LightRAG
from config import Config
from lightrag_utils import initialize_rag, test_embedding_function

# Add parent directory to path to import from GraphRAG-Benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))


async def index_corpus(subset: str, corpus_name: str = None):
    """
    Index a specific corpus from GraphRAG-Benchmark.
    
    Args:
        subset: Subset name ('medical' or 'novel')
        corpus_name: Specific corpus name to index (if None, indexes all)
    """
    # Path to GraphRAG-Benchmark datasets
    benchmark_dir = Path(__file__).parent.parent / "GraphRAG-Benchmark"
    corpus_path = benchmark_dir / "Datasets" / "Corpus" / f"{subset}.parquet"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    print(f"\n{'='*60}")
    print(f"Indexing GraphRAG-Benchmark Corpus: {subset}")
    print(f"{'='*60}\n")
    
    # Load corpus dataset
    print(f"Loading corpus from: {corpus_path}")
    corpus_dataset = load_dataset("parquet", data_files=str(corpus_path), split="train")
    
    # Filter by corpus_name if specified
    corpus_data = []
    for item in corpus_dataset:
        if corpus_name is None or item["corpus_name"] == corpus_name:
            corpus_data.append({
                "corpus_name": item["corpus_name"],
                "context": item["context"]
            })
    
    if not corpus_data:
        print(f"No corpus data found for subset '{subset}'" + 
              (f" and corpus '{corpus_name}'" if corpus_name else ""))
        return
    
    print(f"Found {len(corpus_data)} corpus document(s)\n")
    
    # Validate configuration
    Config.validate()
    
    # Set working directory per subset
    original_working_dir = Config.WORKING_DIR
    Config.WORKING_DIR = str(Path(Config.WORKING_DIR) / "graphrag_benchmark" / subset)
    
    # Ensure working directory exists
    working_dir = Config.ensure_working_dir()
    print(f"Working directory: {working_dir}\n")
    
    # Initialize RAG instance
    print("Initializing RAG...")
    rag = await initialize_rag()
    
    # Test embedding function
    await test_embedding_function(rag)
    
    # Index each corpus document
    print(f"\n{'='*60}")
    print(f"Indexing {len(corpus_data)} corpus document(s)")
    print(f"{'='*60}\n")
    
    for idx, item in enumerate(corpus_data, 1):
        corpus_name_item = item["corpus_name"]
        context = item["context"]
        
        print(f"[{idx}/{len(corpus_data)}] Indexing: {corpus_name_item}")
        print(f"  Context length: {len(context)} characters")
        
        try:
            await rag.ainsert([context], file_paths=[corpus_name_item])
            print(f"  [OK] Successfully indexed\n")
        except Exception as e:
            print(f"  [ERROR] Error indexing: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Finalize storages
    await rag.finalize_storages()
    
    # Restore original working directory
    Config.WORKING_DIR = original_working_dir
    
    print(f"\n{'='*60}")
    print("Indexing completed successfully!")
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Index GraphRAG-Benchmark corpus into integrated_pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python index_graphrag_benchmark.py --subset medical
  python index_graphrag_benchmark.py --subset novel --corpus_name "corpus_1"
        """
    )
    parser.add_argument(
        "--subset",
        required=True,
        choices=["medical", "novel"],
        help="Subset to index (medical or novel)"
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        default=None,
        help="Specific corpus name to index (if not provided, indexes all)"
    )
    
    args = parser.parse_args()
    
    try:
        await index_corpus(args.subset, args.corpus_name)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

