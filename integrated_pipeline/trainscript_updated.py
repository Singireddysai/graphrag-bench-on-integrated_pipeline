"""
Production-ready training script for LightRAG pipeline.
Loads documents and indexes them using the pipeline approach for robustness.
"""
import asyncio
import argparse
import logging
import signal
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from lightrag import LightRAG
from config import Config
from lightrag_utils import initialize_rag, test_embedding_function
from utils import load_all_documents, clear_existing_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lightrag_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and track training metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_documents = 0
        self.track_id = None
        self.errors = []
    
    def start(self, total_docs: int, track_id: str):
        """Start tracking metrics."""
        self.start_time = datetime.now()
        self.total_documents = total_docs
        self.track_id = track_id
        logger.info(f"Started indexing {total_docs} documents with track_id: {track_id}")
    
    def finish(self):
        """Finish tracking and log summary."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        logger.info("="*60)
        logger.info("INDEXING SUMMARY")
        logger.info("="*60)
        logger.info(f"Track ID: {self.track_id}")
        logger.info(f"Total Documents: {self.total_documents}")
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Average: {duration/self.total_documents:.2f} seconds per document")
        logger.info(f"Errors: {len(self.errors)}")
        logger.info("="*60)
    
    def save_to_file(self, filepath: str = "training_metrics.json"):
        """Save metrics to JSON file."""
        metrics = {
            "track_id": self.track_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            "total_documents": self.total_documents,
            "errors": self.errors
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


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


async def validate_environment():
    """Validate environment and prerequisites."""
    logger.info("Validating environment...")
    
    try:
        # Validate configuration
        Config.validate()
        logger.info("✓ Configuration validated")
        
        # Ensure working directory exists
        working_dir = Config.ensure_working_dir()
        logger.info(f"✓ Working directory ready: {working_dir}")
        
        return True
        
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.error("Please ensure your .env file is properly configured.")
        return False
    except Exception as e:
        logger.error(f"Environment validation failed: {e}", exc_info=True)
        return False


async def load_documents(folder_path: Optional[str] = None) -> tuple[list[str], list[str]]:
    """
    Load documents with error handling.
    
    Args:
        folder_path: Optional path to folder containing documents
        
    Returns:
        Tuple of (text_list, file_paths)
    """
    try:
        logger.info("Loading documents...")
        text_list, file_paths = load_all_documents(folder_path=folder_path)
        
        if not text_list:
            logger.error("No documents loaded!")
            return [], []
        
        csv_count = sum(1 for path in file_paths if path.endswith('.csv') or 'llm_tables' in path)
        text_count = len(text_list) - csv_count
        
        logger.info(f"✓ Loaded {len(text_list)} total entries:")
        logger.info(f"  - Text documents: {text_count}")
        logger.info(f"  - CSV tables: {csv_count}")
        
        return text_list, file_paths
        
    except Exception as e:
        logger.error(f"Failed to load documents: {e}", exc_info=True)
        return [], []


async def index_documents_pipeline(
    rag: LightRAG,
    text_list: list[str],
    file_paths: list[str],
    metrics: MetricsCollector
) -> bool:
    """
    Index documents using production-ready pipeline approach.
    
    Args:
        rag: LightRAG instance
        text_list: List of document texts
        file_paths: List of file paths for citations
        metrics: MetricsCollector instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        total_docs = len(text_list)
        
        # Generate unique track_id for monitoring
        track_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("="*60)
        logger.info(f"INDEXING {total_docs} DOCUMENTS")
        logger.info("="*60)
        logger.info("Using pipeline approach for production robustness:")
        logger.info("  ✓ Duplicate detection enabled")
        logger.info("  ✓ Resume capability enabled")
        logger.info("  ✓ Status tracking enabled")
        logger.info("  ✓ Idempotent operations")
        logger.info("="*60)
        
        # Start metrics tracking
        metrics.start(total_docs, track_id)
        
        # Step 1: Enqueue documents
        logger.info("Step 1/2: Enqueueing documents...")
        returned_track_id = await rag.apipeline_enqueue_documents(
            input=text_list,
            file_paths=file_paths,
            track_id=track_id
        )
        logger.info(f"✓ Documents enqueued successfully")
        logger.info(f"  Track ID: {returned_track_id}")
        
        # Step 2: Process the queue
        logger.info("Step 2/2: Processing enqueued documents...")
        logger.info("This may take a while depending on document count and LLM speed...")
        
        await rag.apipeline_process_enqueue_documents()
        
        logger.info("✓ Document processing completed!")
        
        return True
        
    except asyncio.TimeoutError:
        logger.error("Document indexing timed out", exc_info=True)
        metrics.errors.append({"type": "timeout", "message": "Operation timed out"})
        return False
    except ConnectionError as e:
        logger.error(f"Database connection error during indexing: {e}", exc_info=True)
        metrics.errors.append({"type": "connection", "message": str(e)})
        return False
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {e}", exc_info=True)
        metrics.errors.append({"type": "unexpected", "message": str(e)})
        return False


async def main(
    folder_path: Optional[str] = None,
    workspace: Optional[str] = None,
    clear_existing: bool = False,
    skip_validation: bool = False
):
    """
    Main function to run production training pipeline.
    
    Args:
        folder_path: Optional path to folder containing document subdirectories
        workspace: Workspace name for data isolation
        clear_existing: If True, clear existing data before training
        skip_validation: If True, skip test embedding function validation
    """
    rag: Optional[LightRAG] = None
    metrics = MetricsCollector()
    shutdown_handler = GracefulShutdown()
    
    try:
        logger.info("="*60)
        logger.info("LIGHTRAG PRODUCTION TRAINING PIPELINE")
        logger.info("="*60)
        
        # Validate environment
        if not await validate_environment():
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)
        
        # Handle data clearing if requested
        if clear_existing:
            working_dir = Config.ensure_working_dir()
            logger.warning(f"Clearing existing data in {working_dir}...")
            clear_existing_data(str(working_dir))
            logger.info("✓ Existing data cleared")
        
        # Initialize RAG instance
        logger.info("Initializing RAG instance...")
        if workspace:
            logger.info(f"Using workspace: {workspace}")
        rag = await initialize_rag(workspace=workspace)
        logger.info("✓ RAG instance initialized")
        
        # Optional: Test embedding function
        if not skip_validation:
            await test_embedding_function(rag)
        
        # Check for shutdown signal
        if shutdown_handler.shutdown_requested:
            logger.warning("Shutdown requested before loading documents. Exiting gracefully.")
            return
        
        # Load documents
        text_list, file_paths = await load_documents(folder_path=folder_path)
        
        if not text_list:
            logger.error("No documents to index. Exiting.")
            sys.exit(1)
        
        # Check for shutdown signal
        if shutdown_handler.shutdown_requested:
            logger.warning("Shutdown requested before indexing. Exiting gracefully.")
            return
        
        # Index documents using pipeline
        success = await index_documents_pipeline(rag, text_list, file_paths, metrics)
        
        # Finish metrics tracking
        metrics.finish()
        metrics.save_to_file()
        
        if success:
            logger.info("="*60)
            logger.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info("Next steps:")
            logger.info("  1. Run query_script.py to test queries")
            logger.info("  2. Check training_metrics.json for performance data")
            logger.info("  3. Review lightrag_training.log for detailed logs")
            logger.info("="*60)
            sys.exit(0)
        else:
            logger.error("Training completed with errors. Check logs for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        sys.exit(130)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        logger.error("Please ensure your .env file is properly configured.")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"Database connection error: {e}")
        logger.error("Please ensure Neo4j and Qdrant are running and accessible.")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"Unexpected error in training pipeline: {e}", exc_info=True)
        sys.exit(99)
    finally:
        # Cleanup
        if rag:
            try:
                logger.info("Finalizing storage connections...")
                await rag.finalize_storages()
                logger.info("✓ Storage connections finalized")
            except Exception as e:
                logger.error(f"Error finalizing storages: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Production-ready LightRAG training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default folder
  python trainscript_updated.py
  
  # Specify custom document folder
  python trainscript_updated.py --folder multiple_docs_op
  
  # Clear existing data before training
  python trainscript_updated.py --clear-existing
  
  # Skip embedding validation (faster startup)
  python trainscript_updated.py --skip-validation
  
  # Full production run with all options
  python trainscript_updated.py --folder my_docs --clear-existing --skip-validation

Notes:
  - Uses pipeline approach for robustness and resume capability
  - Logs saved to: lightrag_training.log
  - Metrics saved to: training_metrics.json
  - Supports graceful shutdown (Ctrl+C)
  - Idempotent: safe to re-run if interrupted
        """
    )
    
    parser.add_argument(
        "--folder", "-f",
        type=str,
        default=None,
        help="Path to folder containing document subdirectories (e.g., multiple_docs_op)"
    )
    
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=None,
        help="Workspace name for data isolation (e.g., customer_123, project_abc)"
    )
    
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing indexed data before training (WARNING: destructive)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip embedding function validation test (faster startup)"
    )
    
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
        folder_path=args.folder,
        workspace=args.workspace,
        clear_existing=args.clear_existing,
        skip_validation=args.skip_validation
    ))

