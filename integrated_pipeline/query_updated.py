"""
Production-ready query script for LightRAG pipeline.
Queries the RAG system with flexible input/output options.
"""
import asyncio
import argparse
import logging
import signal
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from lightrag import LightRAG, QueryParam
from config import Config
from lightrag_utils import initialize_rag
from prompts import get_query_user_prompt
import prompts  # Import to apply custom prompts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lightrag_query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QueryMetrics:
    """Track query performance metrics."""
    
    def __init__(self):
        self.queries = []
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_duration = 0.0
    
    def add_query(self, query: str, mode: str, duration: float, success: bool, error: str = None):
        """Record a query execution."""
        self.queries.append({
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for storage
            "mode": mode,
            "duration_seconds": round(duration, 3),
            "success": success,
            "error": error
        })
        
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        self.total_duration += duration
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        avg_duration = self.total_duration / self.total_queries if self.total_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": f"{(self.successful_queries / self.total_queries * 100):.1f}%" if self.total_queries > 0 else "0%",
            "total_duration_seconds": round(self.total_duration, 3),
            "average_duration_seconds": round(avg_duration, 3)
        }
    
    def save_to_file(self, filepath: str = "query_metrics.json"):
        """Save metrics to JSON file."""
        metrics_data = {
            "summary": self.get_summary(),
            "queries": self.queries
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Query metrics saved to {filepath}")


class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True


def validate_query_input(query: str) -> str:
    """
    Validate and sanitize query input.
    
    Args:
        query: User query string
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If query is invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    if len(query) < 3:
        raise ValueError("Query must be at least 3 characters long")
    
    if len(query) > 5000:
        raise ValueError("Query exceeds maximum length of 5000 characters")
    
    return query


def load_queries_from_file(filepath: str) -> List[str]:
    """
    Load queries from a text file (one query per line) or JSON file.
    
    Args:
        filepath: Path to queries file
        
    Returns:
        List of query strings
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Queries file not found: {filepath}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(q) for q in data if q]
            elif isinstance(data, dict) and 'queries' in data:
                return [str(q) for q in data['queries'] if q]
            else:
                raise ValueError("JSON file must contain a list or dict with 'queries' key")
    else:
        # Plain text file - one query per line
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            return queries


async def query_rag(
    rag: LightRAG,
    query: str,
    mode: str = "hybrid",
    chunk_top_k: Optional[int] = None,
    max_total_tokens: Optional[int] = None,
    metrics: Optional[QueryMetrics] = None
) -> Optional[str]:
    """
    Query the RAG system with specified parameters.
    
    Args:
        rag: LightRAG instance
        query: Query string
        mode: Query mode (naive, local, global, hybrid, mix)
        chunk_top_k: Number of top chunks to retrieve
        max_total_tokens: Maximum total tokens for response
        metrics: QueryMetrics instance for tracking
        
    Returns:
        Query result string or None if error
    """
    start_time = datetime.now()
    
    try:
        # Validate query
        query = validate_query_input(query)
        
        # Use defaults from config if not provided
        chunk_top_k = chunk_top_k or Config.DEFAULT_CHUNK_TOP_K
        max_total_tokens = max_total_tokens or Config.DEFAULT_MAX_TOTAL_TOKENS
        
        # Get user prompt to enforce reference format
        user_prompt = get_query_user_prompt()
        
        param = QueryParam(
            mode=mode,
            chunk_top_k=chunk_top_k,
            max_total_tokens=max_total_tokens,
            user_prompt=user_prompt,
        )
        
        logger.debug(f"Executing query with mode={mode}, chunk_top_k={chunk_top_k}")
        result = await rag.aquery(query, param=param)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query completed in {duration:.2f}s")
        
        if metrics:
            metrics.add_query(query, mode, duration, success=True)
        
        return result
        
    except ValueError as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Query validation error: {e}")
        if metrics:
            metrics.add_query(query, mode, duration, success=False, error=str(e))
        return None
    except asyncio.TimeoutError:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error("Query timed out")
        if metrics:
            metrics.add_query(query, mode, duration, success=False, error="Timeout")
        return None
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Query execution error: {e}", exc_info=True)
        if metrics:
            metrics.add_query(query, mode, duration, success=False, error=str(e))
        return None


async def validate_environment(workspace: Optional[str] = None):
    """Validate environment and prerequisites."""
    logger.info("Validating environment...")
    
    try:
        # Validate configuration
        Config.validate()
        logger.info(" Configuration validated")
        
        # Check if working directory exists
        working_dir = Path(Config.WORKING_DIR)
        if not working_dir.exists():
            logger.error(f"Working directory '{Config.WORKING_DIR}' not found")
            logger.error("Please run training script first to index your documents")
            return False
        
        logger.info(f" Working directory exists: {working_dir}")
        
        if workspace:
            logger.info(f" Using workspace: {workspace}")
        
        return True
        
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.error("Please ensure your .env file is properly configured.")
        return False
    except Exception as e:
        logger.error(f"Environment validation failed: {e}", exc_info=True)
        return False


def save_results(results: List[dict], output_file: str):
    """
    Save query results to file.
    
    Args:
        results: List of result dictionaries
        output_file: Output file path
    """
    output_path = Path(output_file)
    
    if output_path.suffix == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # Plain text format
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                f.write(f"{'='*80}\n")
                f.write(f"Query {i}: {result['query']}\n")
                f.write(f"Mode: {result['mode']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"{result['result']}\n")
                f.write(f"\n{'-'*80}\n\n")
    
    logger.info(f"Results saved to {output_file}")


async def run_interactive_mode(rag: LightRAG, mode: str, metrics: QueryMetrics):
    """
    Run interactive query mode.
    
    Args:
        rag: LightRAG instance
        mode: Query mode to use
        metrics: QueryMetrics instance
    """
    logger.info("="*60)
    logger.info("INTERACTIVE QUERY MODE")
    logger.info("="*60)
    logger.info("Enter your queries (type 'exit' or 'quit' to stop)")
    logger.info(f"Query mode: {mode}")
    logger.info("="*60)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                logger.info("Exiting interactive mode")
                break
            
            if not query:
                continue
            
            logger.info(f"Processing query: {query}")
            result = await query_rag(rag, query, mode=mode, metrics=metrics)
            
            if result:
                print("\n" + "="*60)
                print("RESULT:")
                print("="*60)
                print(result)
                print("-"*60)
            else:
                print("\n✗ Query failed. Check logs for details.")
                
        except KeyboardInterrupt:
            logger.info("\nInterrupted. Exiting...")
            break
        except EOFError:
            logger.info("\nEOF received. Exiting...")
            break


async def main(
    workspace: Optional[str] = None,
    query: Optional[str] = None,
    queries_file: Optional[str] = None,
    mode: str = "hybrid",
    interactive: bool = False,
    output_file: Optional[str] = None,
    save_metrics: bool = True
):
    """
    Main function to run production query pipeline.
    
    Args:
        workspace: Workspace name for data isolation
        query: Single query to execute
        queries_file: Path to file containing queries
        mode: Query mode (naive, local, global, hybrid, mix)
        interactive: Run in interactive mode
        output_file: Optional file to save results
        save_metrics: Whether to save metrics to file
    """
    rag: Optional[LightRAG] = None
    metrics = QueryMetrics()
    shutdown_handler = GracefulShutdown()
    
    try:
        logger.info("="*60)
        logger.info("LIGHTRAG PRODUCTION QUERY PIPELINE")
        logger.info("="*60)
        
        # Validate environment
        if not await validate_environment(workspace):
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)
        
        # Initialize RAG instance
        logger.info("Initializing RAG instance...")
        rag = await initialize_rag(workspace=workspace)
        logger.info(" RAG instance loaded successfully")
        
        # Determine query source
        results = []
        
        if interactive:
            # Interactive mode
            await run_interactive_mode(rag, mode, metrics)
            
        elif query:
            # Single query from command line
            logger.info(f"Executing single query in {mode} mode")
            logger.info(f"Query: {query}")
            
            result = await query_rag(rag, query, mode=mode, metrics=metrics)
            
            if result:
                print("\n" + "="*60)
                print("RESULT:")
                print("="*60)
                print(result)
                print("-"*60)
                
                results.append({
                    "query": query,
                    "mode": mode,
                    "result": result
                })
            else:
                logger.error("Query execution failed")
                sys.exit(1)
                
        elif queries_file:
            # Batch queries from file
            logger.info(f"Loading queries from file: {queries_file}")
            queries = load_queries_from_file(queries_file)
            
            logger.info(f"Loaded {len(queries)} queries")
            logger.info(f"Executing in {mode} mode")
            logger.info("="*60)
            
            for i, q in enumerate(queries, 1):
                if shutdown_handler.shutdown_requested:
                    logger.warning("Shutdown requested. Stopping query processing.")
                    break
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Query {i}/{len(queries)}")
                logger.info(f"{'='*60}")
                logger.info(f"Query: {q}")
                
                result = await query_rag(rag, q, mode=mode, metrics=metrics)
                
                if result:
                    print("\n" + "-"*60)
                    print("RESULT:")
                    print("-"*60)
                    print(result)
                    print("-"*60)
                    
                    results.append({
                        "query": q,
                        "mode": mode,
                        "result": result
                    })
                else:
                    logger.error(f"Query {i} failed")
        else:
            logger.error("No query source specified. Use --query, --queries-file, or --interactive")
            sys.exit(1)
        
        # Save results if requested
        if output_file and results:
            save_results(results, output_file)
        
        # Print metrics summary
        summary = metrics.get_summary()
        logger.info("\n" + "="*60)
        logger.info("QUERY METRICS SUMMARY")
        logger.info("="*60)
        for key, value in summary.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        logger.info("="*60)
        
        # Save metrics
        if save_metrics:
            metrics.save_to_file()
        
        logger.info("\n Query pipeline completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Query pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        logger.error("Please ensure your .env file is properly configured.")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"Database connection error: {e}")
        logger.error("Please ensure Neo4j and Qdrant are running and accessible.")
        sys.exit(2)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(3)
    except Exception as e:
        logger.critical(f"Unexpected error in query pipeline: {e}", exc_info=True)
        sys.exit(99)
    finally:
        # Cleanup
        if rag:
            try:
                logger.info("Finalizing storage connections...")
                await rag.finalize_storages()
                logger.info("Storage connections finalized")
            except Exception as e:
                logger.error(f"Error finalizing storages: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Production-ready LightRAG query pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query from command line
  python query_updated.py --query "What is NVIDIA's revenue?" --mode hybrid
  
  # Batch queries from file
  python query_updated.py --queries-file my_queries.txt --mode global
  
  # Interactive mode
  python query_updated.py --interactive --mode hybrid
  
  # With workspace isolation
  python query_updated.py --workspace customer_123 --query "Revenue data?"
  
  # Save results to file
  python query_updated.py --queries-file queries.txt --output results.json

Query File Formats:
  - Text file: One query per line (# for comments)
  - JSON file: {"queries": ["query1", "query2", ...]} or ["query1", "query2"]

Notes:
  - Logs saved to: lightrag_query.log
  - Metrics saved to: query_metrics.json
  - Supports graceful shutdown (Ctrl+C)
        """
    )
    
    # Workspace argument
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=None,
        help="Workspace name for data isolation (must match training workspace)"
    )
    
    # Query input options (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to execute"
    )
    query_group.add_argument(
        "--queries-file", "-f",
        type=str,
        help="Path to file containing queries (one per line or JSON)"
    )
    query_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive query mode"
    )
    
    # Query parameters
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["naive", "local", "global", "hybrid", "mix"],
        default="global",
        help="Query mode (default: global)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to file (.txt or .json)"
    )
    
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Don't save metrics to file"
    )
    
    # Logging level
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run main function
    asyncio.run(main(
        workspace=args.workspace,
        query=args.query,
        queries_file=args.queries_file,
        mode=args.mode,
        interactive=args.interactive,
        output_file=args.output,
        save_metrics=not args.no_metrics
    ))

