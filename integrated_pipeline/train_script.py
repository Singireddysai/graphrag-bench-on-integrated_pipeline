"""
Training script for LightRAG pipeline.
Loads documents and inserts them into the RAG system.
"""
import asyncio
import argparse
from pathlib import Path
from lightrag import LightRAG
from config import Config
from lightrag_utils import initialize_rag, test_embedding_function
from utils import load_all_documents, clear_existing_data


async def main(folder_path: str = None):
    """
    Main function to train the LightRAG system.
    
    Args:
        folder_path: Optional path to folder containing document subdirectories
    """
    rag: LightRAG = None
    
    try:
        # Validate configuration
        Config.validate()
        
        # Ensure working directory exists
        working_dir = Config.ensure_working_dir()
        
        # Optional: Clear old data files
        clear_existing = input("\nClear existing data? (y/N): ").strip().lower() == 'y'
        
        if clear_existing:
            clear_existing_data(str(working_dir))
        
        # Initialize RAG instance
        print("\nInitializing RAG...")
        rag = await initialize_rag()
        
        # Test embedding function
        await test_embedding_function(rag)
        
        # Load all documents (from folder or config)
        text_list, file_paths = load_all_documents(folder_path=folder_path)
        
        # Insert all documents into RAG
        if text_list:
            total_docs = len(text_list)
            csv_count = sum(1 for path in file_paths if path.endswith('.csv') or 'llm_tables' in path)
            
            print(f"\n{'='*50}")
            print(f"Inserting {total_docs} document(s) into RAG")
            print(f"  - Text entries: {total_docs - csv_count}")
            print(f"  - CSV tables: {csv_count}")
            print(f"{'='*50}")
            print("This may take a while...")
            
            await rag.ainsert(text_list, file_paths=file_paths)
            print("✓ All documents inserted successfully!")
        else:
            print("Error: No documents to insert!")
            return
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure your .env file is properly configured.")
        print("You can copy .env.example to .env and fill in your values.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LightRAG system with documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_script.py                           # Defaults to 'multiple_docs_op' folder
  python train_script.py --folder multiple_docs_op # Load from folder structure
        """
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        default=None,
        help="Path to folder containing document subdirectories (e.g., multiple_docs_op)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(folder_path=args.folder))
    print("\nDone!")
