#!/usr/bin/env python3
"""
Streamlit UI for LLM-Ready PDF Extraction Pipeline
A user-friendly web interface for extracting tables, images, and text from PDFs, optimized for LLM consumption.
"""

import streamlit as st
import os
import sys
import json
import time
import zipfile
import copy
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import tempfile
import shutil
import base64
from PIL import Image
import io

# Add the llm_ready_pipeline to the path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.llm_runner import run_llm_pipeline

# Page configuration
st.set_page_config(
    page_title="LLM-Ready PDF Extraction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default configuration
DEFAULT_LLM_CONFIG = {
    "llm_tables": {
        "strategy": "hi_res",
        "min_rows": 1,
        "min_cols": 2,
        "extract_page_headers": True,
        "extract_sections": True,
        "skip_empty_rows": True,
        "max_table_nesting_level": 5,
        "use_pymupdf4llm": True,
        "include_context": True,
        "generate_markdown": True,
        "generate_json": True
    },
    "llm_images": {
        "min_size": 50,
        "dpi": 300,
        "include_captions": True,
        "merge_tolerance": 20,
        "extract_vector_graphics": True,
        "use_pymupdf4llm": True,
        "generate_base64": True,
        "include_context": True,
        "extract_ocr_text": True
    },
    "llm_text": {
        "use_pymupdf4llm": True,
        "extract_headers": True,
        "extract_footers": True,
        "extract_captions": True,
        "extract_tables_as_markdown": True,
        "extract_images_with_captions": True,
        "preserve_formatting": True,
        "use_ocr": False,
        "min_confidence": 0.5,
        "write_images": True,
        "embed_images": False,
        "preprocess_text": True,
        "normalize_unicode": True,
        "convert_ellipsis": True
    }
}

def initialize_session_state():
    """Initialize session state variables."""
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'extracted_content' not in st.session_state:
        st.session_state.extracted_content = None
    if 'pdf_page_ranges' not in st.session_state:
        st.session_state.pdf_page_ranges = {}
    if 'master_page_range_enabled' not in st.session_state:
        st.session_state.master_page_range_enabled = False
    if 'use_same_range_for_all' not in st.session_state:
        st.session_state.use_same_range_for_all = False
    if 'extract_tables' not in st.session_state:
        st.session_state.extract_tables = True
    if 'extract_images' not in st.session_state:
        st.session_state.extract_images = True
    if 'extract_text' not in st.session_state:
        st.session_state.extract_text = True

def clear_all_content():
    """Clear all previously extracted content and reset session state."""
    try:
        # Clear the output directory
        output_dir = os.path.join(os.getcwd(), "streamlit_output_llm")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        # Clear session state
        st.session_state.results = None
        st.session_state.output_dir = None
        st.session_state.extracted_content = None
        st.session_state.processing_status = "idle"
        
        return True
    except Exception as e:
        st.error(f"Error clearing content: {str(e)}")
        return False

def get_extracted_files(output_dir: str, file_types: tuple) -> List[Dict[str, Any]]:
    """Get all extracted files of specific types with metadata."""
    files_found = []
    if not os.path.exists(output_dir):
        return files_found
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith(file_types):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, output_dir)
                
                try:
                    file_info = {
                        'path': file_path,
                        'name': file,
                        'relative_path': rel_path,
                        'size': os.path.getsize(file_path),
                        'icon': get_file_icon(file)
                    }
                    
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        with Image.open(file_path) as img:
                            file_info['dimensions'] = f"{img.width}x{img.height}"
                            file_info['format'] = img.format
                    elif file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        file_info['rows'] = len(df)
                        file_info['columns'] = len(df.columns)
                        file_info['dataframe'] = df
                    elif file.endswith(('.txt', '.json', '.md', '.jsonl')):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        file_info['content'] = content
                        file_info['lines'] = len(content.splitlines())
                        file_info['characters'] = len(content)
                        
                    files_found.append(file_info)
                except Exception as e:
                    files_found.append({
                        'path': file_path,
                        'name': file,
                        'relative_path': rel_path,
                        'size': os.path.getsize(file_path),
                        'error': str(e)
                    })
    
    return sorted(files_found, key=lambda x: x['name'])

def display_image_gallery(images: List[Dict[str, Any]]):
    """Display extracted images in a gallery format."""
    if not images:
        st.info("No images were extracted.")
        return
    
    st.subheader(f"🖼️ Extracted Images ({len(images)} found)")
    
    # Create columns for image display
    cols_per_row = 3
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(images):
                img_info = images[i + j]
                
                with col:
                    try:
                        # Display image
                        with open(img_info['path'], 'rb') as f:
                            img_data = f.read()
                        
                        st.image(img_data, caption=img_info['name'], use_column_width=True)
                        
                        # Image metadata
                        st.caption(f"📏 {img_info['dimensions']} | 📦 {img_info['size']:,} bytes")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download",
                            data=img_data,
                            file_name=img_info['name'],
                            mime="image/png",
                            key=f"img_download_gallery_{i+j}_{img_info['name']}"
                        )
                        
                    except Exception as e:
                        st.error(f"Error loading {img_info['name']}: {str(e)}")

def display_tables(tables: List[Dict[str, Any]]):
    """Display extracted tables in an interactive format."""
    if not tables:
        st.info("No tables were extracted.")
        return
    
    st.subheader(f"📊 Extracted Tables ({len(tables)} found)")
    
    for i, table_info in enumerate(tables):
        with st.expander(f"📋 {table_info['name']} ({table_info['rows']} rows × {table_info['columns']} columns)", expanded=False):
            try:
                if 'dataframe' in table_info:
                    # Display the table
                    st.dataframe(table_info['dataframe'], use_container_width=True)
                    
                    # Table statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", table_info['rows'])
                    with col2:
                        st.metric("Columns", table_info['columns'])
                    with col3:
                        st.metric("Size", f"{table_info['size']:,} bytes")
                    
                    # Download options
                    col1, col2, col3 = st.columns(3)
                    
                    # CSV download
                    with open(table_info['path'], 'rb') as f:
                        csv_data = f.read()
                    
                    with col1:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=table_info['name'],
                            mime="text/csv",
                            key=f"csv_download_gallery_{i}_{table_info['name']}"
                        )
                    
                    # JSON download (if exists)
                    json_path = table_info['path'].replace('.csv', '.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'rb') as f:
                            json_data = f.read()
                        
                        with col2:
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_data,
                                file_name=table_info['name'].replace('.csv', '.json'),
                                mime="application/json",
                                key=f"json_download_gallery_{i}_{table_info['name']}"
                            )
                    
                    # Markdown download (if exists)
                    md_path = table_info['path'].replace('.csv', '.md')
                    if os.path.exists(md_path):
                        with open(md_path, 'rb') as f:
                            md_data = f.read()
                        
                        with col3:
                            st.download_button(
                                label="📥 Download Markdown",
                                data=md_data,
                                file_name=table_info['name'].replace('.csv', '.md'),
                                mime="text/markdown",
                                key=f"md_download_gallery_{i}_{table_info['name']}"
                            )
                
                else:
                    st.error(f"Error loading table: {table_info.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error displaying table {table_info['name']}: {str(e)}")

def display_text_content(text_files: List[Dict[str, Any]]):
    """Display extracted text content with syntax highlighting."""
    if not text_files:
        st.info("No text files were extracted.")
        return
    
    st.subheader(f"📝 Extracted Text Content ({len(text_files)} files)")
    
    for i, text_info in enumerate(text_files):
        with st.expander(f"📄 {text_info['name']} ({text_info.get('lines', 0)} lines, {text_info.get('characters', 0):,} chars)", expanded=False):
            try:
                if 'content' in text_info:
                    # Display text content
                    st.text_area(
                        "Content Preview",
                        value=text_info['content'][:2000] + ("..." if len(text_info['content']) > 2000 else ""),
                        height=300,
                        disabled=True,
                        key=f"text_preview_{i}"
                    )
                    
                    # Text statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lines", text_info.get('lines', 0))
                    with col2:
                        st.metric("Characters", text_info.get('characters', 0))
                    with col3:
                        st.metric("Size", f"{text_info['size']:,} bytes")
                    
                    # Download button
                    with open(text_info['path'], 'rb') as f:
                        text_data = f.read()
                    
                    st.download_button(
                        label="📥 Download Text File",
                        data=text_data,
                        file_name=text_info['name'],
                        mime="text/plain",
                        key=f"text_download_gallery_{i}_{text_info['name']}"
                    )
                
                else: 
                    st.error(f"Error loading text file: {text_info.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error displaying text file {text_info['name']}: {str(e)}")

def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.header("🔧 LLM-Ready Configuration")
    
    # Advanced settings only - extraction types moved to main page
    st.sidebar.subheader("⚙️ Advanced Settings")
    
    with st.sidebar.expander("📊 Table Settings", expanded=False):
        # Note: pymupdf4llm option removed for tables - now using only unstructured.partition_pdf
        use_pymupdf_tables = False  # Deprecated - kept for compatibility
        st.markdown("**🔍 Extraction Engine:** unstructured.partition_pdf")
        st.caption("Table of contents pages are automatically filtered out")
        table_strategy = st.selectbox(
            "Strategy",
            ["hi_res", "fast"],
            index=0,
            help="High resolution for better accuracy, fast for speed",
            key="table_strategy_selectbox"
        )
        generate_markdown_tables = st.checkbox("Generate Markdown", value=True, key="generate_markdown_tables_checkbox")
        generate_json_tables = st.checkbox("Generate JSON", value=True, key="generate_json_tables_checkbox")
        include_table_context = st.checkbox("Include Context", value=True, key="include_table_context_checkbox")
        st.caption("💡 Table cell text is automatically preprocessed (Unicode normalization, special characters)")

    with st.sidebar.expander("🖼️ Image Settings", expanded=False):
        use_pymupdf_images = st.checkbox("Use pymupdf4llm for Images", value=True, key="use_pymupdf_images_checkbox")
        generate_base64 = st.checkbox("Generate Base64", value=True, key="generate_base64_checkbox")
        extract_ocr_text = st.checkbox("Extract OCR Text", value=True, key="extract_ocr_text_checkbox")
        include_image_context = st.checkbox("Include Context", value=True, key="include_image_context_checkbox")

    with st.sidebar.expander("📝 Text Settings", expanded=False):
        use_pymupdf_text = st.checkbox("Use pymupdf4llm for Text", value=True, key="use_pymupdf_text_checkbox")
        extract_tables_as_md = st.checkbox("Tables as Markdown", value=True, key="extract_tables_as_md_checkbox")
        extract_images_with_captions = st.checkbox("Images with Captions", value=True, key="extract_images_with_captions_checkbox")
        st.markdown("**🔬 Hybrid Classification:**")
        st.caption("Unstructured + PyMuPDF classification automatically improves text block labeling (title, heading, paragraph, header, footer, caption)")
        preprocess_text_enabled = st.checkbox("Enhanced Preprocessing", value=True, help="Enable Unicode normalization and special character handling for text and table cells", key="preprocess_text_enabled_checkbox")
        normalize_unicode = st.checkbox("Normalize Unicode", value=True, help="Convert Unicode escape sequences (\\u2212, \\u03c1, etc.) and normalize special characters (Greek letters, mathematical symbols)", key="normalize_unicode_checkbox")
        convert_ellipsis = st.checkbox("Convert Ellipsis", value=True, help="Convert ellipsis characters to regular dots", key="convert_ellipsis_checkbox")
    
    # Build configuration
    config = DEFAULT_LLM_CONFIG.copy()
    config["llm_tables"]["use_pymupdf4llm"] = use_pymupdf_tables
    config["llm_tables"]["strategy"] = table_strategy
    config["llm_tables"]["generate_markdown"] = generate_markdown_tables
    config["llm_tables"]["generate_json"] = generate_json_tables
    config["llm_tables"]["include_context"] = include_table_context
    
    config["llm_images"]["use_pymupdf4llm"] = use_pymupdf_images
    config["llm_images"]["generate_base64"] = generate_base64
    config["llm_images"]["extract_ocr_text"] = extract_ocr_text
    config["llm_images"]["include_context"] = include_image_context
    
    config["llm_text"]["use_pymupdf4llm"] = use_pymupdf_text
    config["llm_text"]["extract_tables_as_markdown"] = extract_tables_as_md
    config["llm_text"]["extract_images_with_captions"] = extract_images_with_captions
    config["llm_text"]["preprocess_text"] = preprocess_text_enabled
    config["llm_text"]["normalize_unicode"] = normalize_unicode
    config["llm_text"]["convert_ellipsis"] = convert_ellipsis
    
    # Note: selected_extractions will be set in main() based on session state
    config["selected_extractions"] = []
    
    return config

def get_file_icon(file_path: str) -> str:
    """Get appropriate icon for file type."""
    ext = os.path.splitext(file_path)[1].lower()
    icon_map = {
        '.pdf': '📄',
        '.csv': '📊',
        '.json': '📋',
        '.txt': '📝',
        '.md': '📝',
        '.png': '🖼️',
        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.zip': '📦',
        '.jsonl': '📋'
    }
    return icon_map.get(ext, '📄')

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

@st.cache_data(ttl=300)  # Cache for 5 minutes to avoid re-scanning on every rerun
def _get_document_dirs(output_dir: str):
    """Get document directories. Cached to prevent re-scanning on reruns."""
    if not os.path.exists(output_dir):
        return []
    
    document_dirs = []
    
    # Check root level for document directories
    # Each PDF creates a directory named after the PDF file (without extension)
    # Each such directory contains subdirectories: llm_tables/, llm_images/, llm_text/ (or tables/, images/, text/)
    try:
        root_items = os.listdir(output_dir)
        for item in root_items:
            item_path = os.path.join(output_dir, item)
            # Check if it's a directory
            if os.path.isdir(item_path):
                # Check if this directory has the structure of a document directory
                # (contains images/, tables/, text/ or llm_tables/, llm_images/, llm_text/ subdirectories)
                subdirs = []
                try:
                    subdirs = [d for d in os.listdir(item_path) 
                              if os.path.isdir(os.path.join(item_path, d))]
                except (PermissionError, OSError):
                    continue
                
                # If it has images, tables, or text subdirectories (standard or LLM), it's a document directory
                if any(subdir in ['images', 'tables', 'text', 'llm_tables', 'llm_images', 'llm_text', 'figures'] for subdir in subdirs):
                    document_dirs.append(item_path)
    except (PermissionError, OSError) as e:
        st.warning(f"Error scanning output directory: {e}")
    
    # If no document directories found, check if files are directly in output_dir
    # (this happens when there's only one PDF or structure is different)
    if not document_dirs:
        # Check if output_dir itself contains images/, tables/, or text/
        try:
            subdirs = [d for d in os.listdir(output_dir) 
                      if os.path.isdir(os.path.join(output_dir, d))]
            if any(subdir in ['images', 'tables', 'text', 'llm_tables', 'llm_images', 'llm_text'] for subdir in subdirs):
                document_dirs = [output_dir]
            else:
                # Last resort: treat output_dir as a single document
                document_dirs = [output_dir]
        except (PermissionError, OSError):
            document_dirs = [output_dir]
    
    # Sort by directory name for consistent display order
    document_dirs.sort(key=lambda x: os.path.basename(x).lower())
    
    return document_dirs

@st.cache_data(ttl=3600)
def _load_image_cached(img_path: str) -> bytes:
    """Cache image loading to avoid repeated file reads."""
    with open(img_path, 'rb') as f:
        return f.read()

@st.cache_data(ttl=3600)
def _load_table_data_cached(table_path: str):
    """Cache table loading to avoid repeated file reads."""
    return pd.read_csv(table_path)

@st.cache_data(ttl=3600)
def _load_json_cached(json_path: str):
    """Cache JSON loading."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(ttl=1800)
def _load_text_file_cached(text_path: str, max_bytes: int = 50000) -> str:
    """Cache text file loading with size limit."""
    try:
        with open(text_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(max_bytes)
    except Exception:
        return "Error reading file"

def display_extracted_content(output_dir: str, page_range_enabled: bool = False):
    """Display all extracted content organized by document/source with collapsible sections."""
    if not os.path.exists(output_dir):
        st.error("❌ Output directory does not exist.")
        return
    
    # Get all content organized by document
    documents = {}
    total_files = 0
    total_size = 0
    
    # Get document directories (cached to prevent re-scanning on reruns)
    document_dirs = _get_document_dirs(output_dir)
    
    if not document_dirs:
        st.info("No documents found in the output directory.")
        return
    
    # Debug: Show found documents (can be removed later)
    if len(document_dirs) > 0:
        st.caption(f"Found {len(document_dirs)} document(s): {', '.join([os.path.basename(d) for d in document_dirs])}")
    
    # Process each document directory
    for doc_dir in document_dirs:
        # Use the directory name as the document name
        # The directory name is the PDF filename (without extension)
        doc_name = os.path.basename(doc_dir)
        # Only change if it's the same as output_dir (meaning files are directly in output_dir)
        if doc_name == os.path.basename(output_dir) or doc_dir == output_dir:
            # Try to get a better name from manifest or files
            manifest_path = os.path.join(doc_dir, 'llm_manifest.json')  # LLM uses llm_manifest.json
            if not os.path.exists(manifest_path):
                manifest_path = os.path.join(doc_dir, 'manifest.json')  # Fallback
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        if 'pdf' in manifest:
                            source_name = os.path.basename(manifest['pdf'])
                            doc_name = os.path.splitext(source_name)[0] if source_name else "Extracted Content"
                        else:
                            doc_name = "Extracted Content"
                except:
                    doc_name = "Extracted Content"
            else:
                doc_name = "Extracted Content"
        
        # Store the directory path with the document for manifest lookup
        if doc_name not in documents:
            documents[doc_name] = {
                'images': [],
                'tables': [],
                'text_files': [],
                'other_files': [],
                '_doc_dir': doc_dir  # Store directory path for manifest lookup
            }
        else:
            # If duplicate name, append directory info
            documents[doc_name]['_doc_dir'] = doc_dir
        
        # Walk through all files in this document directory
        for root, dirs, files in os.walk(doc_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                file_info = {
                    'name': file,
                    'path': file_path,
                    'size': file_size,
                    'size_formatted': format_file_size(file_size),
                    'icon': get_file_icon(file)
                }
                
                # Note: Page range filtering is now handled during extraction, not in display
                # All files in output_dir are already filtered by their respective page ranges
                
                total_files += 1
                total_size += file_size
                
                # Categorize by file type
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    documents[doc_name]['images'].append(file_info)
                elif file.endswith('.csv'):
                    documents[doc_name]['tables'].append(file_info)
                elif file.endswith('.json'):
                    # Skip JSON files - these are table metadata, not text files
                    # They're automatically shown with their corresponding CSV tables
                    continue
                elif file.endswith('.md'):
                    # Markdown files for LLM tables - skip here, shown with CSV
                    continue
                elif file.endswith(('.txt', '.jsonl')):
                    # Note: Text files are already filtered by page range during extraction
                    documents[doc_name]['text_files'].append(file_info)
                else:
                    documents[doc_name]['other_files'].append(file_info)
    
    # Display content for each document with collapsible sections
    for doc_name, content in documents.items():
        # Calculate document statistics
        doc_total_files = len(content['images']) + len(content['tables']) + len(content['text_files']) + len(content['other_files'])
        doc_total_size = (
            sum(img.get('size', 0) for img in content['images']) +
            sum(tbl.get('size', 0) for tbl in content['tables']) +
            sum(txt.get('size', 0) for txt in content['text_files']) +
            sum(oth.get('size', 0) for oth in content['other_files'])
        )
        
        # Get total page count from manifest if available
        total_pages = None
        manifest_path = None
        # Use the stored doc_dir if available, otherwise try to find it
        doc_dir_for_manifest = content.get('_doc_dir')
        if not doc_dir_for_manifest:
            # Try to find the directory by name
            for d in document_dirs:
                if os.path.basename(d) == doc_name:
                    doc_dir_for_manifest = d
                    break
        
        if doc_dir_for_manifest and os.path.isdir(doc_dir_for_manifest):
            manifest_path = os.path.join(doc_dir_for_manifest, 'llm_manifest.json')
            if not os.path.exists(manifest_path):
                manifest_path = os.path.join(doc_dir_for_manifest, 'manifest.json')
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        # Get total page count from metadata
                        if 'metadata' in manifest and 'page_range' in manifest['metadata']:
                            pr = manifest['metadata']['page_range']
                            if not pr.get('enabled'):
                                # When page range is disabled, end_page is the total pages in the PDF
                                total_pages = pr.get('end_page', None)
                            else:
                                # When page range is enabled, end_page only shows last extracted page
                                # Count unique pages from paragraph_text.jsonl to get total pages extracted
                                total_pages = None
                        elif 'metadata' in manifest and 'page_count' in manifest['metadata']:
                            total_pages = manifest['metadata']['page_count']
                except Exception:
                    pass
        
        # If still no page count found, count unique pages from paragraph_text.jsonl
        # This works as a fallback and also for when page range is enabled
        if not total_pages:
            page_nums = set()
            for txt_file in content['text_files']:
                if 'paragraph_text.jsonl' in txt_file['name']:
                    try:
                        with open(txt_file['path'], 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    page_no = data.get('page_no')
                                    if page_no:
                                        page_nums.add(page_no)
                                except:
                                    pass
                    except:
                        pass
            if page_nums:
                # Get the maximum page number (total pages extracted/available)
                total_pages = max(page_nums)
        
        # Create collapsible document section
        with st.expander(f"📄 **{doc_name}** - {len(content['images'])} images, {len(content['tables'])} tables", expanded=False):
            # Document summary metrics - show Total Number of Pages and Total Size
            doc_metrics_col1, doc_metrics_col2 = st.columns(2)
            with doc_metrics_col1:
                if total_pages:
                    st.metric("📑 Total No. of Pages", total_pages)
                else:
                    st.metric("📑 Total No. of Pages", "N/A")
            with doc_metrics_col2:
                st.metric("📦 Total Size", format_file_size(doc_total_size))
            
            # Sort files by path to maintain folder order (same as in folder structure)
            content['images'] = sorted(content['images'], key=lambda x: os.path.relpath(x['path'], output_dir))
            content['tables'] = sorted(content['tables'], key=lambda x: os.path.relpath(x['path'], output_dir))
            content['text_files'] = sorted(content['text_files'], key=lambda x: os.path.relpath(x['path'], output_dir))
            content['other_files'] = sorted(content['other_files'], key=lambda x: os.path.relpath(x['path'], output_dir))
            
            # Display all content with tabs for each document
            if any(content.values()):  # Only show tabs if there's content
                # Create tabs for different content types within each document
                tab_images, tab_tables, tab_text = st.tabs([
                    f"🖼️ Images ({len(content['images'])})",
                    f"📊 Tables ({len(content['tables'])})", 
                    f"📝 Text ({len(content['text_files'])})"
                ])
                
                with tab_images:
                    if content['images']:
                        # Load manifest to get captions for images
                        image_captions = {}
                        image_captions_by_filename = {}  # Fallback: match by filename
                        manifest_path = None
                        if doc_dir_for_manifest and os.path.isdir(doc_dir_for_manifest):
                            manifest_path = os.path.join(doc_dir_for_manifest, 'llm_manifest.json')
                            if not os.path.exists(manifest_path):
                                manifest_path = os.path.join(doc_dir_for_manifest, 'manifest.json')
                            
                            if manifest_path and os.path.exists(manifest_path):
                                try:
                                    with open(manifest_path, 'r', encoding='utf-8') as f:
                                        manifest = json.load(f)
                                        # Create mapping from image path to caption
                                        # Handle both old structure (list) and new structure (dict with 'images' key)
                                        images_data = manifest.get('images', [])
                                        if isinstance(images_data, dict):
                                            images_data = images_data.get('images', [])
                                        
                                        for img_entry in images_data:
                                            img_filename = img_entry.get('filename')
                                            caption = img_entry.get('caption')
                                            if img_filename and caption:
                                                # Try matching by filename
                                                img_basename = os.path.basename(str(img_filename))
                                                image_captions_by_filename[img_basename] = caption
                                except Exception as e:
                                    pass  # Silently fail if manifest can't be loaded
                        
                        page_images = content['images']
                        
                        # Display images in a grid
                        cols_per_row = 3
                        for i in range(0, len(page_images), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j, col in enumerate(cols):
                                if i + j < len(page_images):
                                    img_info = page_images[i + j]
                                    global_img_idx = i + j
                                    
                                    with col:
                                        try:
                                            # Get caption if available
                                            img_caption = None
                                            img_filename = os.path.basename(img_info['path'])
                                            if img_filename in image_captions_by_filename:
                                                img_caption = image_captions_by_filename[img_filename]
                                            
                                            # Display image (cached)
                                            img_data = _load_image_cached(img_info['path'])
                                            
                                            # Use extracted caption if available, otherwise use filename
                                            display_caption = img_caption if img_caption else img_info['name']
                                            st.image(img_data, caption=display_caption)
                                            
                                            # Show caption separately if available (for better visibility)
                                            if img_caption:
                                                st.info(f"📝 **Caption:** {img_caption}")
                                            
                                            # Image metadata
                                            st.caption(f"📏 {img_info['size_formatted']}")
                                            
                                            # Download button
                                            st.download_button(
                                                label="📥 Download",
                                                data=img_data,
                                                file_name=img_info['name'],
                                                mime="image/png",
                                                key=f"img_download_{doc_name}_{global_img_idx}_{img_info['name']}"
                                            )
                                            
                                        except Exception as e:
                                            st.error(f"Error loading {img_info['name']}: {str(e)}")
                    else:
                        st.info("No images found in this document.")
            
                with tab_tables:
                    if content['tables']:
                        # Load manifest to get captions for tables
                        table_captions = {}
                        table_captions_by_filename = {}  # Fallback: match by filename
                        manifest_path = None
                        if doc_dir_for_manifest and os.path.isdir(doc_dir_for_manifest):
                            manifest_path = os.path.join(doc_dir_for_manifest, 'llm_manifest.json')
                            if not os.path.exists(manifest_path):
                                manifest_path = os.path.join(doc_dir_for_manifest, 'manifest.json')
                            
                            if manifest_path and os.path.exists(manifest_path):
                                try:
                                    with open(manifest_path, 'r', encoding='utf-8') as f:
                                        manifest = json.load(f)
                                        # Create mapping from CSV path to caption
                                        # Handle both old structure (list) and new structure (dict with 'tables' key)
                                        if 'tables' in manifest:
                                            tables_data = manifest['tables']
                                            if isinstance(tables_data, dict):
                                                tables_list = tables_data.get('tables', [])
                                            else:
                                                tables_list = tables_data
                                            
                                            for table_entry in tables_list:
                                                # Get csv_path from file_paths object or fallback to csv_path field
                                                file_paths = table_entry.get('file_paths', {})
                                                csv_path = file_paths.get('csv') if file_paths else table_entry.get('csv_path')
                                                caption = table_entry.get('caption')
                                                if csv_path and caption:
                                                    # Normalize path for comparison
                                                    normalized_csv_path = os.path.normpath(csv_path)
                                                    table_captions[normalized_csv_path] = caption
                                                    
                                                    # Also create mapping by filename for fallback matching
                                                    csv_filename = os.path.basename(csv_path)
                                                    table_captions_by_filename[csv_filename] = caption
                                except Exception as e:
                                    pass  # Silently fail if manifest can't be loaded
                        
                        # Display all tables - lazy loading ensures no lag (content only loads when expander opened)
                        for i, table_info in enumerate(content['tables']):
                            with st.expander(f"📊 {table_info['name']} ({table_info['size_formatted']})", expanded=False):
                                try:
                                    # Display caption if available
                                    table_caption = None
                                    # Try matching by full path first
                                    normalized_table_path = os.path.normpath(table_info['path'])
                                    if normalized_table_path in table_captions:
                                        table_caption = table_captions[normalized_table_path]
                                    else:
                                        # Fallback: match by filename
                                        table_filename = os.path.basename(table_info['path'])
                                        if table_filename in table_captions_by_filename:
                                            table_caption = table_captions_by_filename[table_filename]
                                    
                                    if table_caption:
                                        st.info(f"📝 **Caption:** {table_caption}")
                                    
                                    # Lazy load table (cached)
                                    df = _load_table_data_cached(table_info['path'])
                                
                                    # Limit rows for display
                                    max_display_rows = 1000
                                    if len(df) > max_display_rows:
                                        st.warning(f"⚠️ Table has {len(df)} rows. Showing first {max_display_rows} rows.")
                                        st.dataframe(df.head(max_display_rows), width='stretch')
                                    else:
                                        st.dataframe(df, width='stretch')
                                    
                                    # Table statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Rows", len(df))
                                    with col2:
                                        st.metric("Columns", len(df.columns))
                                    with col3:
                                        st.metric("Size", table_info['size_formatted'])
                                    
                                    # Check for merged cells in JSON (cached)
                                    json_path = table_info['path'].replace('.csv', '.json')
                                    merged_cells_count = 0
                                    json_data = None
                                    if os.path.exists(json_path):
                                        try:
                                            json_data = _load_json_cached(json_path)
                                            # Check for merged cells in metadata or structured_data
                                            if isinstance(json_data, dict):
                                                merged_cells = json_data.get('merged_cells', [])
                                                if not merged_cells and 'metadata' in json_data:
                                                    merged_cells = json_data['metadata'].get('merged_cells', [])
                                                if not merged_cells and 'structured_data' in json_data:
                                                    merged_cells = json_data['structured_data'].get('merged_cells', [])
                                                merged_cells_count = len(merged_cells) if merged_cells else 0
                                        except Exception:
                                            pass
                                    
                                    with col4:
                                        st.metric("Merged Cells", merged_cells_count if merged_cells_count > 0 else "None")
                                    
                                    # Display merged cells info if available
                                    if merged_cells_count > 0 and json_data:
                                        st.markdown("**🔗 Merged Cells Information:**")
                                        if isinstance(json_data, dict):
                                            merged_cells = json_data.get('merged_cells', [])
                                            if not merged_cells and 'metadata' in json_data:
                                                merged_cells = json_data['metadata'].get('merged_cells', [])
                                            if not merged_cells and 'structured_data' in json_data:
                                                merged_cells = json_data['structured_data'].get('merged_cells', [])
                                            if merged_cells:
                                                merged_df = pd.DataFrame(merged_cells)
                                                if 'row' in merged_df.columns and 'col' in merged_df.columns:
                                                    st.dataframe(merged_df[['row', 'col', 'rowspan', 'colspan']], width='stretch', hide_index=True)
                                    
                                    # Display extraction metadata (engine, etc.)
                                    if json_data and isinstance(json_data, dict):
                                        metadata = json_data.get('metadata', {})
                                        extraction_method = metadata.get('extraction_method', 'unknown')
                                        
                                        # Display extraction engine info
                                        if extraction_method:
                                            engine_info_cols = st.columns(3)
                                            
                                            with engine_info_cols[0]:
                                                # Display engine name with icon
                                                engine_icons = {
                                                    'camelot': '🐫',
                                                    'pdfplumber': '📄',
                                                    'pymupdf4llm': '🤖',
                                                    'partition_pdf': '📊'
                                                }
                                                engine_icon = engine_icons.get(extraction_method, '⚙️')
                                                st.markdown(f"**{engine_icon} Engine:** {extraction_method.title()}")
                                            
                                            # Display source type (vector vs image)
                                            source = json_data.get('source') or metadata.get('source')
                                            if source:
                                                with engine_info_cols[1]:
                                                    source_icon = "🖼️" if source == "image" else "📐"
                                                    st.markdown(f"**{source_icon} Source:** {source.title()}")
                                            
                                            # Show preprocessing info
                                            with engine_info_cols[2]:
                                                st.caption("✨ Cell text preprocessed")
                                    
                                    # Download buttons
                                    download_col1, download_col2, download_col3 = st.columns(3)
                                    
                                    with download_col1:
                                        with open(table_info['path'], 'rb') as f:
                                            csv_data = f.read()
                                        st.download_button(
                                            label="📥 Download CSV",
                                            data=csv_data,
                                            file_name=table_info['name'],
                                            mime="text/csv",
                                            key=f"csv_download_{doc_name}_{i}_{table_info['name']}"
                                        )
                                    
                                    # JSON download (if exists)
                                    json_path = table_info['path'].replace('.csv', '.json')
                                    if os.path.exists(json_path):
                                        with download_col2:
                                            with open(json_path, 'rb') as f:
                                                json_data_bytes = f.read()
                                            st.download_button(
                                                label="📥 Download JSON",
                                                data=json_data_bytes,
                                                file_name=table_info['name'].replace('.csv', '.json'),
                                                mime="application/json",
                                                key=f"json_download_{doc_name}_{i}_{table_info['name']}"
                                            )
                                    
                                    # Markdown download (if exists) - LLM-specific
                                    md_path = table_info['path'].replace('.csv', '.md')
                                    if os.path.exists(md_path):
                                        with download_col3:
                                            with open(md_path, 'rb') as f:
                                                md_data = f.read()
                                            st.download_button(
                                                label="📥 Download Markdown",
                                                data=md_data,
                                                file_name=table_info['name'].replace('.csv', '.md'),
                                                mime="text/markdown",
                                                key=f"md_download_{doc_name}_{i}_{table_info['name']}"
                                            )
                                    
                                except Exception as e:
                                    st.error(f"Error loading table {table_info['name']}: {str(e)}")
                    else:
                        st.info("No tables found in this document.")
            
                with tab_text:
                    if content['text_files']:
                        # Display all text files
                        for i, text_info in enumerate(content['text_files']):
                            global_text_idx = i
                            with st.expander(f"📝 {text_info['name']} ({text_info['size_formatted']})", expanded=False):
                                try:
                                    # Lazy load text content (cached, limited size)
                                    content_text = _load_text_file_cached(text_info['path'], max_bytes=50000)
                                    
                                    # Check if this is a JSONL file (paragraph_text.jsonl) and show hybrid info
                                    is_jsonl = text_info['name'].endswith('.jsonl')
                                    hybrid_info_shown = False
                                    
                                    if is_jsonl:
                                        try:
                                            # Parse JSONL to show hybrid classification summary
                                            hybrid_count = 0
                                            provenance_counts = {}
                                            text_type_counts = {}
                                            title_count = 0
                                            total_blocks = 0
                                            with open(text_info['path'], 'r', encoding='utf-8') as f:
                                                for line in f:
                                                    try:
                                                        data = json.loads(line)
                                                        total_blocks += 1
                                                        if data.get('hybrid_label'):
                                                            hybrid_count += 1
                                                        prov = data.get('provenance') or data.get('metadata', {}).get('provenance')
                                                        if prov:
                                                            provenance_counts[prov] = provenance_counts.get(prov, 0) + 1
                                                        text_type = data.get('region_type', 'unknown')
                                                        text_type_counts[text_type] = text_type_counts.get(text_type, 0) + 1
                                                        if text_type == 'Title':
                                                            title_count += 1
                                                    except:
                                                        pass
                                            
                                            # Show classification summary
                                            if total_blocks > 0:
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.metric("Total Blocks", total_blocks)
                                                with col2:
                                                    st.metric("Titles Detected", title_count)
                                            
                                            if hybrid_count > 0:
                                                st.info(f"🔬 **Hybrid Classification:** {hybrid_count} text blocks were relabeled using Unstructured + PyMuPDF classification")
                                                hybrid_info_shown = True
                                            
                                            # Show text type breakdown
                                            if text_type_counts:
                                                type_breakdown = ", ".join([f"{k}: {v}" for k, v in sorted(text_type_counts.items())])
                                                st.caption(f"📊 **Text Types:** {type_breakdown}")
                                            
                                            if provenance_counts:
                                                prov_text = ", ".join([f"{k}: {v}" for k, v in provenance_counts.items()])
                                                st.caption(f"🔍 **Provenance:** {prov_text}")
                                            
                                            # Show preprocessing info
                                            st.caption("✨ **Text Preprocessing:** All text blocks and table cells are preprocessed (Unicode normalization, special characters)")
                                        except:
                                            pass
                                    
                                    # Display text content (limit preview for performance)
                                    preview_size = 10000
                                    if len(content_text) > preview_size:
                                        preview_text = content_text[:preview_size] + f"\n\n... [Content truncated, showing first {preview_size:,} characters. Use download to get full content]"
                                        st.warning(f"⚠️ Showing first {preview_size:,} characters of {len(content_text):,} total.")
                                    else:
                                        preview_text = content_text
                                    
                                    st.text_area(
                                        "Content",
                                        value=preview_text,
                                        height=300,
                                        disabled=True,
                                        key=f"text_content_{doc_name}_{global_text_idx}"
                                    )
                                    
                                    # Text statistics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Characters", len(content_text))
                                    with col2:
                                        st.metric("Lines", len(content_text.splitlines()))
                                    
                                    # Download button
                                    with open(text_info['path'], 'rb') as f:
                                        text_data = f.read()
                                    
                                    st.download_button(
                                        label="📥 Download Text File",
                                        data=text_data,
                                        file_name=text_info['name'],
                                        mime="text/plain",
                                        key=f"text_download_{doc_name}_{global_text_idx}_{text_info['name']}"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error loading text file {text_info['name']}: {str(e)}")
                    else:
                        st.info("No text content found in this document.")
            else:
                st.info("No content found for this document.")
    
    if not documents:
        st.info("No documents found in the output directory.")

def display_results(results: Dict[str, Any], output_dir: str):
    """Display extraction results with content previews."""
    if not results or not results.get("success"):
        st.error("❌ Extraction failed or no results available.")
        return
    
    # Get page range info from session state (new multi-PDF structure)
    page_range_info = st.session_state.get('page_range_info', {})
    page_range_enabled = page_range_info.get('enabled', False)
    pdf_ranges = page_range_info.get('pdf_ranges', {})
    
    # Show page range info if enabled
    if page_range_enabled:
        enabled_ranges = []
        for idx, pdf_range in pdf_ranges.items():
            if pdf_range.get('enabled'):
                enabled_ranges.append(f"PDF {idx+1}: pages {pdf_range.get('start_page')}-{pdf_range.get('end_page')}")
        
        if enabled_ranges:
            if len(enabled_ranges) == len(pdf_ranges) and page_range_info.get('use_same_range'):
                # Same range for all
                first_range = list(pdf_ranges.values())[0]
                st.info(f"📑 Showing content from pages {first_range.get('start_page')} to {first_range.get('end_page')} for all documents")
            else:
                # Different ranges
                st.info(f"📑 Page ranges applied to {len(enabled_ranges)} document(s)")
    
    # Summary metrics
    col1 = st.columns(1)[0]
    
    with col1:
        st.metric("📄 PDFs Processed", f"{results['processed']}/{results['total']}")
    
    # Performance metrics
    if 'total_extraction_times' in results:
        perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
        
        with perf_col1:
            st.metric("Total Time", f"{results['total_extraction_times']['total_seconds']:.2f}s")
        
        with perf_col2:
            table_time = results['total_extraction_times'].get('tables_seconds', 0)
            st.metric("Tables Time", f"{table_time:.2f}s")
        
        with perf_col3:
            image_time = results['total_extraction_times'].get('images_seconds', 0)
            st.metric("Images Time", f"{image_time:.2f}s")
        
        with perf_col4:
            text_time = results['total_extraction_times'].get('text_seconds', 0)
            st.metric("Text Time", f"{text_time:.2f}s")
        
        with perf_col5:
            total_pages = results.get('total_pages', 0)
            st.metric("Total Pages", total_pages)
    
    # Content display tabs
    if os.path.exists(output_dir):
        # Note: Page range filtering is now handled during extraction, not in display
        # All content in output_dir is already filtered by their respective page ranges
        
        # Display all extracted content with heading
        st.markdown("### 📂 All Content")
        display_extracted_content(output_dir, page_range_enabled)
        
        # Download options
        st.markdown("---")
        zip_path = os.path.join(output_dir, "extraction_results.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if not file.endswith('.zip'):  # Don't include the zip file itself
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arc_path)
        
        # Download button - use session state counter to ensure unique key
        if 'zip_download_counter' not in st.session_state:
            st.session_state.zip_download_counter = 0
        st.session_state.zip_download_counter += 1
        zip_key = f"zip_download_{st.session_state.zip_download_counter}"
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="📦 Download All Results (ZIP)",
                data=f.read(),
                file_name="extraction_results.zip",
                mime="application/zip",
                key=zip_key
            )

def process_pdfs(uploaded_files: List, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process uploaded PDF files."""
    # Create persistent output directory
    output_dir = os.path.join(os.getcwd(), "streamlit_output_llm")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for input files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files
        pdf_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(file_path)
        
        # Process each PDF
        results = []
        successful = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get page range info from config (new multi-PDF structure)
        page_range_info = config.get("page_range", {})
        master_page_range_enabled = page_range_info.get("master_enabled", False)
        pdf_ranges = page_range_info.get("pdf_ranges", {})
        
        for i, pdf_path in enumerate(pdf_paths):
            status_text.text(f"Processing {os.path.basename(pdf_path)}...")
            
            # Apply page range for this specific PDF if enabled
            # Use deepcopy to ensure nested dicts are not shared
            pdf_config = copy.deepcopy(config)
            
            # Ensure selected_extractions is properly copied and defaults to all if missing
            if "selected_extractions" not in pdf_config or not pdf_config["selected_extractions"]:
                # Default to all extraction types if none selected
                pdf_config["selected_extractions"] = ['tables', 'images', 'text']
            else:
                # Ensure it's a new list, not a reference
                pdf_config["selected_extractions"] = list(pdf_config["selected_extractions"])
            
            # Ensure all required config keys exist with defaults
            if "llm_tables" not in pdf_config:
                pdf_config["llm_tables"] = DEFAULT_LLM_CONFIG["llm_tables"].copy()
            if "llm_images" not in pdf_config:
                pdf_config["llm_images"] = DEFAULT_LLM_CONFIG["llm_images"].copy()
            if "llm_text" not in pdf_config:
                pdf_config["llm_text"] = DEFAULT_LLM_CONFIG["llm_text"].copy()
            
            if master_page_range_enabled and i in pdf_ranges:
                pdf_range = pdf_ranges[i]
                if pdf_range.get('enabled', False):
                    # Apply page range to this PDF
                    pdf_config["page_range"] = {
                        "enabled": True,
                        "start_page": pdf_range.get("start_page"),
                        "end_page": pdf_range.get("end_page")
                    }
                else:
                    # Page range disabled for this specific PDF
                    pdf_config["page_range"] = {"enabled": False}
            else:
                # No page range for this PDF
                pdf_config["page_range"] = {"enabled": False}
            
            try:
                # Validate config before processing
                if not pdf_config.get("selected_extractions"):
                    pdf_config["selected_extractions"] = ['tables', 'images', 'text']
                
                # Process single PDF
                result = run_llm_pipeline(
                    source=pdf_path,
                    out_root=output_dir,
                    config=pdf_config
                )
                
                if result and result.get("success"):
                    results.append(result)
                    successful += 1
                else:
                    error_msg = result.get('message', 'Unknown error') if result else 'No result returned'
                    st.warning(f"⚠️ Failed to process {os.path.basename(pdf_path)}: {error_msg}")
                
                progress_bar.progress((i + 1) / len(pdf_paths))
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error processing {os.path.basename(pdf_path)}: {error_msg}")
                progress_bar.progress((i + 1) / len(pdf_paths))
        
        status_text.text("Processing complete!")
        
        # Combine results
        if results:
            combined_results = {
                "success": True,
                "processed": successful,
                "total": len(pdf_paths),
                "pdfs": []
            }
            
            # Aggregate timing data
            total_table_time = 0
            total_image_time = 0
            total_text_time = 0
            total_pages = 0
            
            for result in results:
                combined_results["pdfs"].extend(result.get("pdfs", []))
                if result.get("total_extraction_times"):
                    total_table_time += result["total_extraction_times"].get("tables_seconds", 0)
                    total_image_time += result["total_extraction_times"].get("images_seconds", 0)
                    total_text_time += result["total_extraction_times"].get("text_seconds", 0)
                total_pages += result.get("total_pages", 0)
            
            combined_results["total_table_extraction_seconds"] = total_table_time
            combined_results["total_image_extraction_seconds"] = total_image_time
            combined_results["total_text_extraction_seconds"] = total_text_time
            combined_results["total_extraction_seconds"] = total_table_time + total_image_time + total_text_time
            combined_results["total_pages"] = total_pages
            combined_results["total_extraction_times"] = {
                "tables_seconds": total_table_time,
                "images_seconds": total_image_time,
                "text_seconds": total_text_time,
                "total_seconds": total_table_time + total_image_time + total_text_time
            }
            
            return combined_results, output_dir
        else:
            return {"success": False, "message": "No PDFs processed successfully"}, None

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Add CSS following Apple Human Interface Guidelines
    st.markdown("""
    <style>
        /* Apple HIG: Clarity - Clear visual hierarchy */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        
        h1 {
            margin-bottom: 0.5rem;
        }
        
        h2 {
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }
        
        h3 {
            margin-top: 1.25rem;
            margin-bottom: 0.5rem;
        }
        
        /* Apple HIG: Deference - Content first, minimal spacing */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            max-width: 1200px;
        }
        
        /* Apple HIG: Subtle spacing for better readability */
        .element-container {
            margin-bottom: 1rem;
        }
        
        /* Apple HIG: Depth - Subtle shadows and rounded corners */
        [data-testid="stMetric"] {
            padding: 0.75rem;
            border-radius: 8px;
        }
        
        /* Apple HIG: Clear visual hierarchy for buttons */
        [data-testid="column"] {
            padding: 0.25rem;
        }
        
        /* Apple HIG: Subtle spacing between sections */
        hr {
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 LLM-Ready PDF Extraction")
    st.markdown("An advanced PDF extraction system optimized for RAG and LLM applications.")
    
    # Feature highlights
    with st.expander("✨ Latest Features", expanded=False):
        st.markdown("""
        **🔬 Hybrid Text Classification**
        - Combines Unstructured layout parsing with PyMuPDF font analysis
        - Automatically improves text block labeling (title, heading, paragraph, header, footer, caption)
        - Smart title detection prevents misclassification of entire page text
        
        **✨ Enhanced Text Preprocessing**
        - Unicode normalization (handles \\u2212, \\u03c1, etc.)
        - Special character handling (Greek letters, mathematical symbols, dashes, quotes)
        - Applied to both text blocks and table cell values
        
        **🚫 Smart Filtering**
        - Table of contents pages automatically filtered out
        - Duplicate and superset blocks removed for cleaner extraction
        
        **📊 Table Extraction**
        - Uses unstructured.partition_pdf for accurate table detection
        - All table cell text is preprocessed for consistency
        """)
    
    # Sidebar configuration
    config = create_sidebar()
    
    # Main content area
    st.header("📁 Upload PDF Files")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files for LLM-ready extraction"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF file(s) uploaded successfully")
        
        # Show file details
        with st.expander("📋 Uploaded Files", expanded=False):
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size:,} bytes)")
        
        # Page range selection feature - Enhanced for multiple PDFs
        # Initialize page range data structure in session state
        if 'pdf_page_ranges' not in st.session_state:
            st.session_state.pdf_page_ranges = {}
        if 'master_page_range_enabled' not in st.session_state:
            st.session_state.master_page_range_enabled = False
        if 'use_same_range_for_all' not in st.session_state:
            st.session_state.use_same_range_for_all = False
        
        # Master toggle for page range
        master_page_range_enabled = st.checkbox(
            "Enable page range selection",
            value=st.session_state.master_page_range_enabled,
            help="⚡ Faster extraction! Extract only specific pages. Uses optimized pdfplumber (skips slow unstructured.partition_pdf)",
            key="master_page_range_toggle"
        )
        st.session_state.master_page_range_enabled = master_page_range_enabled
        
        # Get page counts for all PDFs
        pdf_page_counts = {}
        if master_page_range_enabled:
            import fitz
            import tempfile
            
            for idx, pdf_file in enumerate(uploaded_files):
                try:
                    # Save uploaded file temporarily to get page count
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(pdf_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    doc = fitz.open(tmp_path)
                    pdf_page_counts[idx] = len(doc)
                    doc.close()
                    os.unlink(tmp_path)
                    
                    # Initialize page range for this PDF if not exists
                    if idx not in st.session_state.pdf_page_ranges:
                        st.session_state.pdf_page_ranges[idx] = {
                            'enabled': False,
                            'start_page': 1,
                            'end_page': pdf_page_counts[idx]
                        }
                except Exception as e:
                    st.warning(f"Could not get page count for {pdf_file.name}: {str(e)}")
                    pdf_page_counts[idx] = None
        
        # For multiple PDFs: show per-PDF options
        if len(uploaded_files) > 1 and master_page_range_enabled:
            # Option to use same range for all
            use_same_range = st.checkbox(
                "Use same page range for all documents",
                value=st.session_state.use_same_range_for_all,
                help="Apply the same page range to all uploaded PDFs",
                key="use_same_range_checkbox"
            )
            st.session_state.use_same_range_for_all = use_same_range
            
            if use_same_range:
                # Single range input that applies to all
                st.markdown("**📄 Page Range (applies to all documents):**")
                col1, col2 = st.columns(2)
                
                # Use first PDF's page count as reference (but allow any valid range)
                reference_pages = max(pdf_page_counts.values()) if pdf_page_counts else 1000
                
                with col1:
                    global_start_page = st.number_input(
                        "Start Page (all documents)",
                        min_value=1,
                        max_value=reference_pages,
                        value=st.session_state.pdf_page_ranges.get(0, {}).get('start_page', 1) if st.session_state.pdf_page_ranges else 1,
                        help=f"First page to extract from all PDFs (1-{reference_pages})",
                        key="global_start_page"
                    )
                
                with col2:
                    global_end_page = st.number_input(
                        "End Page (all documents)",
                        min_value=1,
                        max_value=reference_pages,
                        value=st.session_state.pdf_page_ranges.get(0, {}).get('end_page', reference_pages) if st.session_state.pdf_page_ranges else reference_pages,
                        help=f"Last page to extract from all PDFs (1-{reference_pages})",
                        key="global_end_page"
                    )
                
                if global_start_page > global_end_page:
                    st.error("❌ Start page must be less than or equal to end page")
                    master_page_range_enabled = False
                else:
                    # Apply to all PDFs
                    for idx in range(len(uploaded_files)):
                        max_pages = pdf_page_counts.get(idx, reference_pages) or reference_pages
                        st.session_state.pdf_page_ranges[idx] = {
                            'enabled': True,
                            'start_page': min(global_start_page, max_pages),
                            'end_page': min(global_end_page, max_pages)
                        }
                    st.success(f"✅ Will extract pages {global_start_page} to {global_end_page} from all {len(uploaded_files)} documents")
            else:
                # Per-PDF page range selection
                st.markdown("**📄 Page Range for Each Document:**")
                for idx, pdf_file in enumerate(uploaded_files):
                    with st.expander(f"📑 {pdf_file.name} (Pages: {pdf_page_counts.get(idx, 'Unknown')})", expanded=False):
                        pdf_max_pages = pdf_page_counts.get(idx)
                        
                        if pdf_max_pages:
                            # Initialize if not exists
                            if idx not in st.session_state.pdf_page_ranges:
                                st.session_state.pdf_page_ranges[idx] = {
                                    'enabled': False,
                                    'start_page': 1,
                                    'end_page': pdf_max_pages
                                }
                            
                            # Enable/disable toggle for this PDF
                            pdf_enabled = st.checkbox(
                                f"Enable page range for this document",
                                value=st.session_state.pdf_page_ranges[idx].get('enabled', False),
                                key=f"pdf_range_enabled_{idx}"
                            )
                            st.session_state.pdf_page_ranges[idx]['enabled'] = pdf_enabled
                            
                            if pdf_enabled:
                                col1, col2 = st.columns(2)
                                with col1:
                                    start_page = st.number_input(
                                        "Start Page",
                                        min_value=1,
                                        max_value=pdf_max_pages,
                                        value=st.session_state.pdf_page_ranges[idx].get('start_page', 1),
                                        help=f"First page to extract (1-{pdf_max_pages})",
                                        key=f"pdf_start_{idx}"
                                    )
                                    st.session_state.pdf_page_ranges[idx]['start_page'] = start_page
                                
                                with col2:
                                    end_page = st.number_input(
                                        "End Page",
                                        min_value=1,
                                        max_value=pdf_max_pages,
                                        value=st.session_state.pdf_page_ranges[idx].get('end_page', pdf_max_pages),
                                        help=f"Last page to extract (1-{pdf_max_pages})",
                                        key=f"pdf_end_{idx}"
                                    )
                                    st.session_state.pdf_page_ranges[idx]['end_page'] = end_page
                                
                                if start_page > end_page:
                                    st.error("❌ Start page must be less than or equal to end page")
                                    st.session_state.pdf_page_ranges[idx]['enabled'] = False
                                else:
                                    st.success(f"✅ Will extract pages {start_page} to {end_page}")
                        else:
                            st.warning(f"⚠️ Could not determine page count for {pdf_file.name}. Page range disabled for this document.")
                            st.session_state.pdf_page_ranges[idx] = {'enabled': False}
        elif len(uploaded_files) == 1 and master_page_range_enabled:
            # Single PDF: simple range selection
            pdf_file = uploaded_files[0]
            pdf_max_pages = pdf_page_counts.get(0)
            
            if pdf_max_pages:
                st.info(f"📄 Total pages in document: **{pdf_max_pages}**")
                
                # Initialize if not exists
                if 0 not in st.session_state.pdf_page_ranges:
                    st.session_state.pdf_page_ranges[0] = {
                        'enabled': True,
                        'start_page': 1,
                        'end_page': pdf_max_pages
                    }
                
                col1, col2 = st.columns(2)
                with col1:
                    start_page = st.number_input(
                        "Start Page",
                        min_value=1,
                        max_value=pdf_max_pages,
                        value=st.session_state.pdf_page_ranges[0].get('start_page', 1),
                        help=f"First page to extract (1-{pdf_max_pages})",
                        key="single_pdf_start"
                    )
                    st.session_state.pdf_page_ranges[0]['start_page'] = start_page
                
                with col2:
                    end_page = st.number_input(
                        "End Page",
                        min_value=1,
                        max_value=pdf_max_pages,
                        value=st.session_state.pdf_page_ranges[0].get('end_page', pdf_max_pages),
                        help=f"Last page to extract (1-{pdf_max_pages})",
                        key="single_pdf_end"
                    )
                    st.session_state.pdf_page_ranges[0]['end_page'] = end_page
                
                if start_page > end_page:
                    st.error("❌ Start page must be less than or equal to end page")
                    master_page_range_enabled = False
                else:
                    st.success(f"✅ Will extract pages {start_page} to {end_page} from {pdf_file.name}")
                    st.session_state.pdf_page_ranges[0]['enabled'] = True
            else:
                st.warning("⚠️ Could not determine page count. Page range selection disabled.")
                master_page_range_enabled = False
        
        # Summary display for multiple PDFs
        if len(uploaded_files) > 1 and master_page_range_enabled:
            st.markdown("**📋 Page Range Summary:**")
            summary_data = []
            for idx, pdf_file in enumerate(uploaded_files):
                pdf_range = st.session_state.pdf_page_ranges.get(idx, {})
                if pdf_range.get('enabled'):
                    summary_data.append({
                        'Document': pdf_file.name,
                        'Page Range': f"{pdf_range.get('start_page', '?')}-{pdf_range.get('end_page', '?')}",
                        'Status': '✅ Enabled'
                    })
                else:
                    summary_data.append({
                        'Document': pdf_file.name,
                        'Page Range': 'All pages',
                        'Status': '⏸️ Disabled'
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Extraction type selection - moved to main page with toggle buttons
        # Create toggle buttons with visual feedback
        extract_col1, extract_col2, extract_col3 = st.columns(3)
        
        with extract_col1:
            button_label = "📊 **Tables**" if st.session_state.extract_tables else "📊 Tables"
            button_type = "primary" if st.session_state.extract_tables else "secondary"
            
            if st.button(
                button_label,
                type=button_type,
                help="Extract tables in multiple formats for LLM consumption (CSV, Markdown, JSON)",
                disabled=st.session_state.processing_status == "processing",
                key="toggle_tables_llm"
            ):
                st.session_state.extract_tables = not st.session_state.extract_tables
                st.rerun()
            
            # Visual status indicator
            if st.session_state.extract_tables:
                st.markdown("<p style='color: green; font-weight: bold;'>✓ Selected</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: gray;'>Not selected</p>", unsafe_allow_html=True)
        
        with extract_col2:
            button_label = "🖼️ **Images**" if st.session_state.extract_images else "🖼️ Images"
            button_type = "primary" if st.session_state.extract_images else "secondary"
            
            if st.button(
                button_label,
                type=button_type,
                help="Extract images with Base64 encoding and OCR text",
                disabled=st.session_state.processing_status == "processing",
                key="toggle_images_llm"
            ):
                st.session_state.extract_images = not st.session_state.extract_images
                st.rerun()
            
            # Visual status indicator
            if st.session_state.extract_images:
                st.markdown("<p style='color: green; font-weight: bold;'>✓ Selected</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: gray;'>Not selected</p>", unsafe_allow_html=True)
        
        with extract_col3:
            button_label = "📝 **Text**" if st.session_state.extract_text else "📝 Text"
            button_type = "primary" if st.session_state.extract_text else "secondary"
            
            if st.button(
                button_label,
                type=button_type,
                help="Extract text in LLM-ready Markdown format",
                disabled=st.session_state.processing_status == "processing",
                key="toggle_text_llm"
            ):
                st.session_state.extract_text = not st.session_state.extract_text
                st.rerun()
            
            # Visual status indicator
            if st.session_state.extract_text:
                st.markdown("<p style='color: green; font-weight: bold;'>✓ Selected</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: gray;'>Not selected</p>", unsafe_allow_html=True)
        
        # Update config with selected extractions
        config["selected_extractions"] = []
        if st.session_state.extract_tables:
            config["selected_extractions"].append("tables")
        if st.session_state.extract_images:
            config["selected_extractions"].append("images")
        if st.session_state.extract_text:
            config["selected_extractions"].append("text")
        
        # Visual separator
        st.markdown("---")
        
        # Processing button
        if st.button("🚀 Start LLM-Ready Extraction", type="primary", disabled=st.session_state.processing_status == "processing" or not config["selected_extractions"]):
            if not config["selected_extractions"]:
                st.error("❌ Please select at least one extraction type above.")
                st.stop()
            
            # Automatically clear previous session output before starting
            st.info("🧹 Clearing previous session output...")
            if clear_all_content():
                st.success("✅ Previous session output cleared successfully!")
            else:
                st.warning("⚠️ Could not clear all previous output, but continuing with extraction...")
            
            st.session_state.processing_status = "processing"
            
            # Build page range config from session state
            # Store page range info in config for processing
            # For processing, we'll handle per-PDF ranges in process_pdfs function
            config["page_range"] = {
                "master_enabled": master_page_range_enabled,
                "use_same_range": st.session_state.use_same_range_for_all if len(uploaded_files) > 1 else False,
                "pdf_ranges": st.session_state.pdf_page_ranges.copy() if master_page_range_enabled else {}
            }
            
            # Store for display
            st.session_state.page_range_info = {
                "enabled": master_page_range_enabled,
                "use_same_range": st.session_state.use_same_range_for_all if len(uploaded_files) > 1 else False,
                "pdf_ranges": st.session_state.pdf_page_ranges.copy() if master_page_range_enabled else {}
            }
            
            # Process files
            with st.spinner("🔄 Processing PDFs for LLM-ready extraction..."):
                results, output_dir = process_pdfs(uploaded_files, config)
                
                st.session_state.results = results
                st.session_state.output_dir = output_dir
                st.session_state.processing_status = "completed"
            
            # Store extracted content in session state
            if results and results.get("success"):
                st.session_state.extracted_content = {
                    'images': get_extracted_files(output_dir, ('.png', '.jpg', '.jpeg')),
                    'tables': get_extracted_files(output_dir, ('.csv',)),
                    'text_files': get_extracted_files(output_dir, ('.txt', '.md', '.json', '.jsonl'))
                }
            
            # Display results
            if results and results.get("success"):
                st.success("🎉 LLM-Ready Extraction Completed!")
                display_results(results, output_dir)
            else:
                st.error(f"❌ Extraction failed: {results.get('message', 'Unknown error')}")

    # Footer
    st.markdown("---")
    st.markdown("**LLM-Ready PDF Extraction Pipeline** - Built with Streamlit")
    
    # Help section
    with st.expander("❓ Help & Tips", expanded=False):
        st.markdown("""
        ### How to Use:
        1. **Upload PDFs**: Use the file uploader to select one or more PDF files
        2. **Configure Settings**: Use the sidebar to adjust extraction parameters
        3. **Select Extraction Types**: Choose what to extract (tables, images, text)
        4. **Start Processing**: Click the "Start LLM-Ready Extraction" button
        5. **Download Results**: Use the download buttons to get your extracted content
        
        **Tips:**
        - **Tables**: Outputs in CSV, Markdown, and JSON formats optimized for LLM consumption. Table of contents pages are automatically filtered out.
        - **Images**: Includes Base64 encoding and OCR text extraction
        - **Text**: Structured Markdown format with hybrid Unstructured + PyMuPDF classification for improved accuracy
        - **Performance**: Use page range selection for faster extraction on large documents
        
        **Key Features:**
        - **🔬 Hybrid Text Classification**: Combines Unstructured layout parsing with PyMuPDF font analysis for accurate text block labeling (title, heading, paragraph, header, footer, caption)
        - **✨ Text Preprocessing**: Automatic Unicode normalization and special character handling for both text blocks and table cells (handles Greek letters, mathematical symbols, etc.)
        - **🎯 Smart Title Detection**: Prevents entire page text from being misclassified as titles using multiple validation checks
        - **🚫 Table of Contents Filtering**: Automatically detects and skips table of contents pages during table extraction
        - **🔄 Duplicate Block Removal**: Removes duplicate and superset blocks to ensure clean, accurate extraction
        
        **Output Formats:**
        - **Tables**: CSV, Markdown, and JSON files (JSON includes merged cell metadata). All cell text is preprocessed.
        - **Images**: PNG/JPEG files with Base64 encoding and OCR text
        - **Text**: Structured Markdown and JSONL files optimized for RAG systems with hybrid classification and preprocessing
        """)

if __name__ == "__main__":
    main()
