import os
import asyncio
import streamlit as st
import nest_asyncio
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Import existing codebase utilities FIRST (before adding llm_ready_pipeline to path)
from config import Config
from lightrag_utils import initialize_rag
from prompts import get_query_user_prompt
import prompts  # Import to apply custom prompts

# Import root-level utils using importlib to avoid binding 'utils' as a module name
# This prevents conflicts with llm_ready_pipeline/utils package
_root_utils_path = Path(__file__).parent / "utils.py"
_spec = importlib.util.spec_from_file_location("_root_utils", _root_utils_path)
_root_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_utils)
load_all_documents = _root_utils.load_all_documents
clear_existing_data = _root_utils.clear_existing_data

# Add paths for imports (after importing root-level utils)
sys.path.insert(0, str(Path(__file__).parent / "llm_ready_pipeline"))

from lightrag import LightRAG, QueryParam

# Import extraction pipeline
from pipeline.llm_runner import run_llm_pipeline

# Apply nest_asyncio to allow nested event loops (needed for Streamlit)
nest_asyncio.apply()

load_dotenv()

# Default extraction config (from streamlit_app.py)
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


# Helper function to run async code in Streamlit
def run_async(coro):
    """Run an async coroutine in Streamlit's event loop"""
    if 'event_loop' not in st.session_state:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply(loop)
        st.session_state.event_loop = loop
    else:
        loop = st.session_state.event_loop
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply(loop)
            st.session_state.event_loop = loop
        else:
            try:
                current_loop = asyncio.get_event_loop()
                if current_loop is not loop and not current_loop.is_closed():
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                asyncio.set_event_loop(loop)
    
    try:
        nest_asyncio.apply(loop)
    except:
        pass
    
    try:
        running_loop = asyncio.get_running_loop()
        if running_loop is not loop:
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=300)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
        st.error(f"Error running async operation: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise




async def train_rag(extraction_output_dir: str, clear_existing: bool = False):
    """Train RAG with extracted data using existing utilities"""
    if not os.path.exists(extraction_output_dir):
        raise ValueError(f"Extraction output directory not found: {extraction_output_dir}")
    
    # Validate configuration
    Config.validate()
    
    # Ensure working directory exists
    working_dir = Config.ensure_working_dir()
    
    # Clear existing data if requested
    if clear_existing:
        clear_existing_data(str(working_dir))
    
    # Initialize RAG instance using existing utility
    rag = await initialize_rag()
    
    # Load all documents from the extraction output directory using existing utility
    text_list, file_paths = load_all_documents(folder_path=extraction_output_dir)
    
    # Insert all documents into RAG
    if text_list:
        total_docs = len(text_list)
        await rag.ainsert(text_list, file_paths=file_paths)
        return rag, total_docs
    else:
        raise ValueError("No extracted data found to train on")


async def query_rag(rag: LightRAG, query: str, mode: str = "hybrid", chunk_top_k: int = None, max_total_tokens: int = None) -> str:
    """
    Query the RAG system with specified parameters using existing utilities.
    
    Args:
        rag: LightRAG instance
        query: Query string
        mode: Query mode (naive, local, global, hybrid, mix)
        chunk_top_k: Number of top chunks to retrieve
        max_total_tokens: Maximum total tokens for response
        
    Returns:
        Query result string
    """
    # Use defaults from config if not provided
    chunk_top_k = chunk_top_k or Config.DEFAULT_CHUNK_TOP_K
    max_total_tokens = max_total_tokens or Config.DEFAULT_MAX_TOTAL_TOKENS
    
    # Get user prompt to enforce reference format using existing utility
    user_prompt = get_query_user_prompt()
    
    param = QueryParam(
        mode=mode,
        chunk_top_k=chunk_top_k,
        max_total_tokens=max_total_tokens,
        user_prompt=user_prompt,
    )
    
    result = await rag.aquery(query, param=param)
    return result


# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
    st.session_state.rag_initialized = False
    st.session_state.extraction_output_dir = None
    st.session_state.training_completed = False
    st.session_state.query_history = []

# Streamlit UI
st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 PDF ChatBot")
st.markdown("Complete pipeline: Extract → Index → Query")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OPENAI_API_KEY environment variable is not set.")
    st.stop()

# Create tabs for different stages
tab1, tab2, tab3 = st.tabs(["📄 1. Extract", "🎓 2. Index", "🔍 3. Query"])

# ==================== TAB 1: EXTRACTION ====================
with tab1:
    st.header("📄 PDF Extraction")
    st.markdown("Extract text, tables, and images from PDF files")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files for extraction"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} PDF file(s) uploaded")
        
        # Extraction type selection
        st.subheader("Extraction Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extract_tables = st.checkbox("📊 Extract Tables", value=True)
        with col2:
            extract_images = st.checkbox("🖼️ Extract Images", value=True)
        with col3:
            extract_text = st.checkbox("📝 Extract Text", value=True)
        
        selected_extractions = []
        if extract_tables:
            selected_extractions.append("tables")
        if extract_images:
            selected_extractions.append("images")
        if extract_text:
            selected_extractions.append("text")
        
        if not selected_extractions:
            st.warning("⚠️ Please select at least one extraction type")
        else:
            if st.button("🚀 Start Extraction", type="primary"):
                with st.spinner("🔄 Extracting content from PDFs..."):
                    # Create temporary directory for uploaded files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            pdf_paths.append(file_path)
                        
                        # Create output directory (use multiple_docs_op as standard location)
                        output_dir = os.path.join(os.getcwd(), "multiple_docs_op")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Build config
                        config = DEFAULT_LLM_CONFIG.copy()
                        config["selected_extractions"] = selected_extractions
                        config["page_range"] = {"enabled": False}
                        
                        # Process PDFs
                        if len(pdf_paths) == 1:
                            results = run_llm_pipeline(pdf_paths[0], output_dir, config)
                        else:
                            # For multiple PDFs, process each one
                            all_results = []
                            for pdf_path in pdf_paths:
                                result = run_llm_pipeline(pdf_path, output_dir, config)
                                if result and result.get("success"):
                                    all_results.append(result)
                            
                            # Combine results
                            if all_results:
                                results = {
                                    "success": True,
                                    "processed": len(all_results),
                                    "total": len(pdf_paths),
                                    "pdfs": [pdf for r in all_results for pdf in r.get("pdfs", [])]
                                }
                            else:
                                results = {"success": False, "message": "No PDFs processed successfully"}
                        
                        if results and results.get("success"):
                            st.session_state.extraction_output_dir = output_dir
                            st.success("🎉 Extraction completed successfully!")
                            
                            # Show summary
                            st.subheader("Extraction Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("PDFs Processed", f"{results['processed']}/{results['total']}")
                            with col2:
                                total_tables = sum(pdf.get('tables', 0) for pdf in results.get('pdfs', []))
                                st.metric("Tables Extracted", total_tables)
                            with col3:
                                total_text = sum(pdf.get('text_blocks', 0) for pdf in results.get('pdfs', []))
                                st.metric("Text Blocks", total_text)
                            
                            st.info(f"📁 Output directory: `{output_dir}`")
                        else:
                            st.error(f"❌ Extraction failed: {results.get('message', 'Unknown error')}")
    
    # Show existing extraction output if available
    if st.session_state.extraction_output_dir and os.path.exists(st.session_state.extraction_output_dir):
        st.divider()
        st.subheader("Existing Extraction Output")
        st.info(f"📁 Found extraction output at: `{st.session_state.extraction_output_dir}`")
        
        # List extracted documents
        extraction_path = Path(st.session_state.extraction_output_dir)
        document_dirs = [d for d in extraction_path.iterdir() if d.is_dir()]
        
        if document_dirs:
            st.write(f"Found {len(document_dirs)} document(s):")
            for doc_dir in document_dirs:
                with st.expander(f"📄 {doc_dir.name}"):
                    # Count files
                    jsonl_files = list((doc_dir / "llm_text").glob("*.jsonl")) if (doc_dir / "llm_text").exists() else []
                    csv_files = list((doc_dir / "llm_tables").glob("*.csv")) if (doc_dir / "llm_tables").exists() else []
                    st.write(f"- Text blocks: {len(jsonl_files)}")
                    st.write(f"- Tables: {len(csv_files)}")

# ==================== TAB 2: TRAINING ====================
with tab2:
    st.header("🎓 Indexing")
    st.markdown("Train the knowledge graph and vector database from extracted content")
    
    # Check if extraction output exists (default to multiple_docs_op)
    extraction_dir = st.session_state.extraction_output_dir or os.path.join(os.getcwd(), "multiple_docs_op")
    
    if not os.path.exists(extraction_dir):
        st.warning("⚠️ No extraction output found. Please run extraction first.")
        st.info(f"Expected directory: `{extraction_dir}`")
    else:
        st.success(f"✅ Found extraction output at: `{extraction_dir}`")
        
        # Training options
        st.subheader("Training Options")
        clear_existing = st.checkbox("Clear existing knowledge graph and vector database", value=False)
        
        # Working directory info (from config)
        st.info(f"📁 Working directory: `{Config.WORKING_DIR}`")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("🔄 Training RAG system (this may take a while)..."):
                try:
                    rag, doc_count = run_async(train_rag(extraction_dir, clear_existing=clear_existing))
                    st.session_state.rag = rag
                    st.session_state.rag_initialized = True
                    st.session_state.training_completed = True
                    st.success(f"🎉 Training completed successfully! Processed {doc_count} documents.")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        # Show training status
        if st.session_state.training_completed:
            st.success("✅ Training completed! You can now proceed to the Query tab.")
        elif st.session_state.rag_initialized:
            st.info("ℹ️ RAG instance is initialized. You can query or retrain.")

# ==================== TAB 3: QUERYING ====================
with tab3:
    st.header("🔍 Query Interface")
    st.markdown("Ask anything about the documents")
    
    # Check if RAG is initialized
    if not st.session_state.rag_initialized:
        st.warning("Use the existing data to perform query or custom Index using the Extract and Index tabs")
        
        # Option to initialize without training (if working dir exists)
        if os.path.exists(Config.WORKING_DIR):
            if st.button("🔄 Initialize RAG (use existing data)"):
                with st.spinner("Initializing RAG instance..."):
                    try:
                        st.session_state.rag = run_async(initialize_rag())
                        st.session_state.rag_initialized = True
                        st.success("✅ RAG initialized successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Initialization failed: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
    else:
        # Query settings
        with st.sidebar:
            st.subheader("Query Settings")
            mode = st.selectbox(
                "Query Mode",
                ["global", "hybrid", "local", "naive", "mix"],
                index=1,  # Default to hybrid (from Config.DEFAULT_QUERY_MODE)
                help="Select the query mode"
            )
            
            chunk_top_k = st.slider(
                "Chunk Top K",
                min_value=5,
                max_value=50,
                value=Config.DEFAULT_CHUNK_TOP_K,
                help="Number of top document chunks to retrieve"
            )
            
            max_total_tokens = st.slider(
                "Max Total Tokens",
                min_value=5000,
                max_value=50000,
                value=Config.DEFAULT_MAX_TOTAL_TOKENS,
                step=5000,
                help="Maximum tokens for the response"
            )
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is the value of superficial gas velocity used for simulation?",
            help="Type your question here and click 'Query' to get an answer"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            query_button = st.button("Query", type="primary", use_container_width=True)
        
        # Process query
        if query_button and query:
            with st.spinner("Processing query..."):
                try:
                    result = run_async(
                        query_rag(
                            st.session_state.rag,
                            query,
                            mode=mode,
                            chunk_top_k=chunk_top_k,
                            max_total_tokens=max_total_tokens
                        )
                    )
                    
                    # Store in history
                    st.session_state.query_history.insert(0, {
                        "query": query,
                        "result": result,
                        "mode": mode
                    })
                    
                    # Display result
                    st.divider()
                    st.subheader("Answer")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        # Display query history
        if st.session_state.query_history:
            st.divider()
            st.header("Query History")
            
            for idx, item in enumerate(st.session_state.query_history[:5]):
                with st.expander(f"Query {idx + 1}: {item['query'][:60]}... (Mode: {item['mode']})"):
                    st.markdown("**Query:**")
                    st.write(item['query'])
                    st.markdown("**Answer:**")
                    st.markdown(item['result'])

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>LightRAG Unified Interface | Powered by OpenRouter</small>
    </div>
    """,
    unsafe_allow_html=True
)

