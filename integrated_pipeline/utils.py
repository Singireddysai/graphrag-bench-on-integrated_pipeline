"""
Utility functions for document processing and data loading.
"""
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from config import Config


def convert_csv_to_text(csv_path: str) -> Optional[str]:
    """
    Convert CSV file to readable text format for RAG ingestion.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Formatted text string or None if error occurs
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            return None
        
        # Format as a readable table
        text_parts = []
        csv_filename = os.path.basename(csv_path)
        text_parts.append(f"\n[TABLE: {csv_filename}]\n")
        
        # Process rows
        for idx, row in enumerate(rows):
            # Clean up row data (remove newlines within cells)
            cleaned_row = [str(cell).replace('\n', ' ').replace('\r', '') for cell in row]
            text_parts.append(" | ".join(cleaned_row))
            
            # Add separator after header row
            if idx == 0 and len(rows) > 1:
                separator = "-" * min(80, sum(len(str(cell)) for cell in cleaned_row) + len(cleaned_row) * 3)
                text_parts.append(separator)
        
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None


def load_jsonl_documents(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """
    Load documents from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        Tuple of (text_list, file_paths)
    """
    text_list = []
    file_paths = []
    
    if not os.path.exists(jsonl_path):
        return text_list, file_paths
    
    print(f"\nLoading JSONL file: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                if "text" in json_obj and json_obj["text"]:
                    text_list.append(json_obj["text"])
                    # Format file path for citation
                    doc_id = json_obj.get("doc_id", "unknown")
                    page_no = json_obj.get("page_no", "unknown")
                    file_paths.append(f'Document Name: {doc_id}, Page Number: {page_no}')
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    if text_list:
        print(f"Loaded {len(text_list)} text entries from JSONL file")
    
    return text_list, file_paths


def load_csv_tables(csv_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load CSV tables from directory.
    
    Args:
        csv_dir: Directory containing CSV files
        
    Returns:
        Tuple of (text_list, file_paths)
    """
    text_list = []
    file_paths = []
    csv_count = 0
    
    if not os.path.exists(csv_dir):
        print(f"\nWarning: CSV directory '{csv_dir}' not found")
        return text_list, file_paths
    
    print(f"\nLoading CSV tables from: {csv_dir}")
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    csv_files.sort()  # Sort for consistent ordering
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        csv_text = convert_csv_to_text(csv_path)
        
        if csv_text:
            text_list.append(csv_text)
            # Use CSV filename as file path for citation
            file_paths.append(f"{csv_dir}/{csv_file}")
            csv_count += 1
            print(f"  ✓ Loaded: {csv_file}")
    
    if csv_count > 0:
        print(f"Loaded {csv_count} CSV table(s)")
    else:
        print("No valid CSV files found in llm_tables folder")
    
    return text_list, file_paths


def load_fallback_text(fallback_path: str) -> Tuple[List[str], List[str]]:
    """
    Load fallback text file if JSONL is not available.
    
    Args:
        fallback_path: Path to fallback text file
        
    Returns:
        Tuple of (text_list, file_paths)
    """
    text_list = []
    file_paths = []
    
    if os.path.exists(fallback_path):
        print(f"\nJSONL file not found, using fallback: {fallback_path}")
        with open(fallback_path, "r", encoding="utf-8") as f:
            text_list.append(f.read())
            file_paths.append(os.path.basename(fallback_path))
    else:
        print(f"Warning: Fallback file '{fallback_path}' not found!")
    
    return text_list, file_paths


def load_all_documents(folder_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Load all documents (JSONL, CSV tables, and fallback text).
    
    Args:
        folder_path: Optional path to folder containing document subdirectories
                    (e.g., "multiple_docs_op"). If provided, scans all document
                    subdirectories for llm_text/*_paragraph_text.jsonl and llm_tables/.
                    If None, defaults to "multiple_docs_op" and prompts user if not found.
    
    Returns:
        Tuple of (text_list, file_paths)
    """
    text_list = []
    file_paths = []
    
    # If no folder_path provided, default to "multiple_docs_op"
    if folder_path is None:
        folder_path = "multiple_docs_op"
    
    # Check if folder exists, if not prompt user
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"\nFolder '{folder_path}' does not exist or is not a directory.")
        folder_path = input("Please provide the exact extracted output folder_path: ").strip()
        if not folder_path:
            raise ValueError("No folder path provided. Cannot proceed.")
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Folder '{folder_path}' does not exist or is not a directory")
    
    # Load from folder structure
    print(f"\n{'='*50}")
    print(f"Loading documents from folder: {folder_path}")
    print(f"{'='*50}")
    
    # Find all document subdirectories
    document_dirs = sorted([d for d in folder.iterdir() if d.is_dir()])
    
    if not document_dirs:
        raise ValueError(f"No document subdirectories found in '{folder_path}'")
    
    print(f"Found {len(document_dirs)} document subdirectory(ies)\n")
    
    for doc_dir in document_dirs:
        doc_name = doc_dir.name
        print(f"Processing document: {doc_name}")
        
        # Find *_paragraph_text.jsonl in llm_text/ subdirectory
        llm_text_dir = doc_dir / "llm_text"
        jsonl_files = []
        
        if llm_text_dir.exists() and llm_text_dir.is_dir():
            jsonl_files = list(llm_text_dir.glob("*_paragraph_text.jsonl"))
        
        # If not found in llm_text, check the document directory itself
        if not jsonl_files:
            jsonl_files = list(doc_dir.glob("*_paragraph_text.jsonl"))
        
        # Load JSONL files
        for jsonl_file in jsonl_files:
            print(f"  Loading JSONL: {jsonl_file.name}")
            jsonl_texts, jsonl_paths = load_jsonl_documents(str(jsonl_file))
            text_list.extend(jsonl_texts)
            file_paths.extend(jsonl_paths)
        
        # Find and load CSV tables from llm_tables/ directory
        llm_tables_dir = doc_dir / "llm_tables"
        if llm_tables_dir.exists() and llm_tables_dir.is_dir():
            print(f"  Loading tables from: {llm_tables_dir.name}/")
            csv_texts, csv_paths = load_csv_tables(str(llm_tables_dir))
            text_list.extend(csv_texts)
            file_paths.extend(csv_paths)
        else:
            print(f"  Warning: No llm_tables directory found in {doc_name}")
    
    print(f"\n{'='*50}")
    print(f"Total loaded: {len(text_list)} entries")
    print(f"{'='*50}")
    
    return text_list, file_paths


def get_files_to_delete(working_dir: str) -> List[str]:
    """
    Get list of files to delete when clearing existing data.
    
    Args:
        working_dir: Working directory path
        
    Returns:
        List of file paths to delete
    """
    files_to_delete = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
        "kv_store_entity_chunks.json",
        "kv_store_relation_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    ]
    
    return [os.path.join(working_dir, file) for file in files_to_delete]


def clear_existing_data(working_dir: str) -> None:
    """
    Clear existing RAG data files.
    
    Args:
        working_dir: Working directory path
    """
    files_to_delete = get_files_to_delete(working_dir)
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted old file: {file_path}")

