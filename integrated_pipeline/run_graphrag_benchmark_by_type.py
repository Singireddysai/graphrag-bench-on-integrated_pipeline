"""
Script to run GraphRAG-Benchmark queries by question type.
Samples equal number of questions from each question type and generates predictions.
"""
import asyncio
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from datasets import load_dataset
from lightrag import LightRAG, QueryParam
from config import Config
from lightrag_utils import initialize_rag
from prompts import get_query_user_prompt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def query_rag(
    rag: LightRAG,
    query: str,
    mode: str = "hybrid",
    chunk_top_k: int = 20,
    max_total_tokens: int = 20000
) -> tuple[str, str]:
    """
    Query the RAG system and return response with context.
    
    Args:
        rag: LightRAG instance
        query: Query string
        mode: Query mode (naive, local, global, hybrid)
        chunk_top_k: Number of top chunks to retrieve
        max_total_tokens: Maximum total tokens for response
        
    Returns:
        Tuple of (response, context)
    """
    user_prompt = get_query_user_prompt()
    
    param = QueryParam(
        mode=mode,
        chunk_top_k=chunk_top_k,
        max_total_tokens=max_total_tokens,
        user_prompt=user_prompt,
    )
    
    # Query and get response
    result = await rag.aquery(query, param=param)
    
    # Extract context - this may need adjustment based on LightRAG version
    try:
        if isinstance(result, tuple):
            response, context = result
        else:
            response = result
            context = ""
    except:
        response = result
        context = ""
    
    return response, context


async def process_questions_by_type(
    subset: str,
    corpus_name: str = None,
    questions_per_type: int = 10,
    query_mode: str = "hybrid",
    chunk_top_k: int = 20,
    max_total_tokens: int = 20000
):
    """
    Process questions from GraphRAG-Benchmark by question type.
    Samples equal number of questions from each question type.
    
    Args:
        subset: Subset name ('medical' or 'novel')
        corpus_name: Specific corpus name (if None, processes all)
        questions_per_type: Number of questions to sample per type (default: 10)
        query_mode: Query mode for LightRAG
        chunk_top_k: Number of top chunks to retrieve
        max_total_tokens: Maximum total tokens for response
    """
    # Path to GraphRAG-Benchmark datasets
    benchmark_dir = Path(__file__).parent.parent / "GraphRAG-Benchmark"
    questions_path = benchmark_dir / "Datasets" / "Questions" / f"{subset}_questions.parquet"
    
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    print(f"\n{'='*60}")
    print(f"Processing GraphRAG-Benchmark Questions by Type: {subset}")
    print(f"Sampling {questions_per_type} questions per question type")
    print(f"{'='*60}\n")
    
    # Load questions dataset
    print(f"Loading questions from: {questions_path}")
    questions_dataset = load_dataset("parquet", data_files=str(questions_path), split="train")
    
    # Group questions by type
    questions_by_type = defaultdict(list)
    for item in questions_dataset:
        if corpus_name is None or item["source"] == corpus_name:
            question_data = {
                "id": item["id"],
                "source": item["source"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "evidence": item["evidence"]
            }
            questions_by_type[item["question_type"]].append(question_data)
    
    # Sample questions from each type
    sampled_questions = []
    for q_type, questions in questions_by_type.items():
        if len(questions) < questions_per_type:
            print(f"Warning: Only {len(questions)} questions available for type '{q_type}', using all of them")
            sampled = questions
        else:
            # Sample questions_per_type questions
            import random
            random.seed(42)  # For reproducibility
            sampled = random.sample(questions, questions_per_type)
        
        sampled_questions.extend(sampled)
        print(f"Sampled {len(sampled)} questions from type: {q_type}")
    
    if not sampled_questions:
        print(f"No questions found for subset '{subset}'" + 
              (f" and corpus '{corpus_name}'" if corpus_name else ""))
        return
    
    total_questions = len(sampled_questions)
    print(f"\nTotal questions to process: {total_questions}\n")
    
    # Validate configuration
    Config.validate()
    
    # Set working directory per subset
    original_working_dir = Config.WORKING_DIR
    Config.WORKING_DIR = str(Path(Config.WORKING_DIR) / "graphrag_benchmark" / subset)
    
    # Check if working directory exists
    working_dir = Path(Config.WORKING_DIR)
    if not working_dir.exists():
        raise FileNotFoundError(
            f"Working directory not found: {working_dir}\n"
            "Please run index_graphrag_benchmark.py first to index the corpus."
        )
    
    print(f"Working directory: {working_dir}\n")
    
    # Initialize RAG instance
    print("Loading RAG instance...")
    rag = await initialize_rag()
    print("[OK] RAG instance loaded successfully!\n")
    
    # Process questions
    print(f"{'='*60}")
    print(f"Processing {total_questions} question(s)")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, q in enumerate(sampled_questions, 1):
        print(f"[{idx}/{total_questions}] Processing: {q['id']}")
        print(f"  Question: {q['question'][:100]}...")
        print(f"  Source: {q['source']}")
        print(f"  Type: {q['question_type']}")
        
        try:
            # Query RAG system
            response, context = await query_rag(
                rag=rag,
                query=q["question"],
                mode=query_mode,
                chunk_top_k=chunk_top_k,
                max_total_tokens=max_total_tokens
            )
            
            # Format context as list of strings (as expected by benchmark)
            if isinstance(context, str):
                context_list = [context] if context else []
            elif isinstance(context, list):
                context_list = context
            else:
                context_list = []
            
            # Create result entry
            result = {
                "id": q["id"],
                "question": q["question"],
                "source": q["source"],
                "context": context_list,
                "evidence": q["evidence"],
                "question_type": q["question_type"],
                "generated_answer": str(response),
                "ground_truth": q["answer"]
            }
            
            results.append(result)
            print(f"  [OK] Generated answer ({len(str(response))} chars)\n")
            
        except Exception as e:
            print(f"  [ERROR] Error processing question: {e}\n")
            import traceback
            traceback.print_exc()
            
            # Add error result
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": q["source"],
                "context": [],
                "evidence": q["evidence"],
                "question_type": q["question_type"],
                "generated_answer": f"Error: {str(e)}",
                "ground_truth": q["answer"]
            })
    
    # Save results
    output_dir = benchmark_dir / "results" / "integrated_pipeline" / subset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"predictions_{subset}_by_type"
    if corpus_name:
        output_filename += f"_{corpus_name}"
    output_filename += ".json"
    
    output_path = output_dir / output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary by type
    print(f"{'='*60}")
    print(f"Summary by Question Type:")
    print(f"{'='*60}")
    type_counts = defaultdict(int)
    for r in results:
        type_counts[r["question_type"]] += 1
    for q_type, count in sorted(type_counts.items()):
        print(f"  {q_type}: {count} questions")
    print(f"{'='*60}")
    print(f"Saved {len(results)} prediction(s) to: {output_path}")
    print(f"{'='*60}\n")
    
    # Restore original working directory
    Config.WORKING_DIR = original_working_dir


async def main():
    parser = argparse.ArgumentParser(
        description="Run GraphRAG-Benchmark queries by question type with equal sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_graphrag_benchmark_by_type.py --subset medical --questions_per_type 10
  python run_graphrag_benchmark_by_type.py --subset novel --questions_per_type 5
        """
    )
    parser.add_argument(
        "--subset",
        required=True,
        choices=["medical", "novel"],
        help="Subset to process (medical or novel)"
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        default=None,
        help="Specific corpus name to process (if not provided, processes all)"
    )
    parser.add_argument(
        "--questions_per_type",
        type=int,
        default=10,
        help="Number of questions to sample per question type (default: 10)"
    )
    parser.add_argument(
        "--query_mode",
        type=str,
        default="hybrid",
        choices=["naive", "local", "global", "hybrid"],
        help="Query mode for LightRAG"
    )
    parser.add_argument(
        "--chunk_top_k",
        type=int,
        default=20,
        help="Number of top chunks to retrieve"
    )
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=20000,
        help="Maximum total tokens for response"
    )
    
    args = parser.parse_args()
    
    try:
        await process_questions_by_type(
            subset=args.subset,
            corpus_name=args.corpus_name,
            questions_per_type=args.questions_per_type,
            query_mode=args.query_mode,
            chunk_top_k=args.chunk_top_k,
            max_total_tokens=args.max_total_tokens
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

