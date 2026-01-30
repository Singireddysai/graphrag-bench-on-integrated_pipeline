#!/usr/bin/env python3
"""
LLM-Ready PDF Extraction CLI
Extract tables, images, and text from PDF files optimized for LLM/RAG consumption.
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
import os

# Suppress cryptography deprecation warnings from pypdf dependency
warnings.filterwarnings('ignore', category=DeprecationWarning, module='cryptography')

# Add the llm_pipeline to the path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.llm_runner import run_llm_pipeline, update_all_manifests
from utils.logging import set_log_level

# Default configuration for LLM-ready extraction
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
    },
    "logging": {
        "level": "INFO"
    }
}

def interactive_mode():
    """Interactive mode for user-friendly LLM-ready PDF extraction."""
    print("🤖 LLM-Ready PDF Extraction - Interactive Mode")
    print("=" * 60)
    print("📊 Table Extraction: unstructured.partition_pdf")
    print("📝 Text Extraction: Hybrid Unstructured + PyMuPDF classification")
    print("=" * 60)
    
    # Get PDF input
    print("\n📄 PDF Input Selection:")
    print("1. Single PDF file")
    print("2. Folder containing PDFs")
    
    while True:
        choice = input("\nChoose input type (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ Invalid choice. Please enter 1 or 2.")
    
    # Get PDF path
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        if choice == '1':
            pdf_path = input("\n📁 Enter path to PDF file: ").strip()
        else:
            pdf_path = input("\n📁 Enter path to folder containing PDFs: ").strip()
        
        # Convert to absolute path and check if exists
        pdf_path = os.path.abspath(pdf_path)
        if os.path.exists(pdf_path):
            break
        
        # Try relative to parent directory (for paths like llm_pipeline/sample_pdfs/document1.pdf)
        parent_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), pdf_path))
        if os.path.exists(parent_path):
            pdf_path = parent_path
            break
            
        attempts += 1
        print(f"❌ Path not found: {pdf_path}")
        print("💡 Tip: Use relative paths like 'sample_pdfs/document1.pdf' or 'llm_pipeline/sample_pdfs/document1.pdf'")
        print("💡 Current directory:", os.getcwd())
        if attempts < max_attempts:
            print(f"🔄 Attempt {attempts}/{max_attempts}. Please try again.")
        else:
            print("❌ Maximum attempts reached. Exiting.")
            sys.exit(1)
    
    # Get output directory
    output_dir = input("\n📤 Enter output directory (default: multiple_docs_op): ").strip()
    if not output_dir:
        output_dir = "multiple_docs_op"
    
    # Convert output directory to absolute path
    output_dir = os.path.abspath(output_dir)
    
    # Get extraction options
    print("\n🎯 LLM Pipeline - Extraction Options:")
    print("Available extractions:")
    print("📊 Tables (CSV + Markdown + JSON) - Extract tables in multiple formats for LLM consumption")
    print("🖼️  Images (PNG/JPEG + Base64 + OCR) - Extract images with Base64 encoding and OCR text")
    print("📝 Text (Structured + Markdown) - Extract text with hybrid Unstructured + PyMuPDF classification")
    print("🔧 All extractions - Extract everything in LLM-ready formats")
    
    extractions = []
    while True:
        print("\nWhat would you like to extract?")
        print("Available options: tables, images, text, all")
        print("You can select multiple (e.g., 'tables,images' or 'all')")
        
        user_input = input("Enter your choice: ").strip().lower()
        
        if user_input == 'all':
            extractions = ['tables', 'images', 'text']
            break
        elif user_input:
            valid_options = ['tables', 'images', 'text']
            selected = [opt for opt in user_input.split(',') if opt.strip() in valid_options]
            if selected:
                extractions = selected
                break
            else:
                print("❌ Invalid options. Please choose from: tables, images, text, all")
        else:
            print("❌ Please make a selection.")
    
    # Optional: Page range selection
    page_range_enabled = False
    start_page = None
    end_page = None
    try:
        resp = input("\n📑 Enable page range extraction? (y/n, default: n): ").strip().lower()
        page_range_enabled = resp == 'y'
        if page_range_enabled:
            # If single PDF, try to fetch page count for validation
            total_pages = None
            try:
                if os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
                    import fitz  # PyMuPDF
                    with fitz.open(pdf_path) as d:
                        total_pages = len(d)
            except Exception:
                total_pages = None

            # Prompt for start/end
            while True:
                try:
                    start_page = int(input("   ▶ Start page (1-indexed): ").strip())
                    end_page = int(input("   ▶ End page (>= start): ").strip())
                    if start_page <= 0 or end_page < start_page:
                        print("   ❌ Invalid range. Start must be >=1 and end >= start.")
                        continue
                    if total_pages is not None and end_page > total_pages:
                        print(f"   ❌ End page exceeds total pages ({total_pages}).")
                        continue
                    break
                except ValueError:
                    print("   ❌ Please enter valid integers for pages.")
    except KeyboardInterrupt:
        print("\n❌ Cancelled.")
        sys.exit(1)
    
    # Build configuration based on selections
    config = DEFAULT_LLM_CONFIG.copy()
    
    # Store selected extractions for processing logic
    config["selected_extractions"] = extractions
    # Store page range info
    if page_range_enabled:
        config["page_range"] = {
            "enabled": True,
            "start_page": start_page,
            "end_page": end_page,
        }
    else:
        config["page_range"] = {"enabled": False}
    
    # Simple confirmation
    print(f"\n✅ Configuration Summary:")
    print(f"   📁 Input: {pdf_path}")
    print(f"   📤 Output: {output_dir}")
    print(f"   🎯 Extractions: {', '.join(extractions)}")
    if page_range_enabled:
        print(f"   📑 Page range: {start_page}-{end_page}")
    
    confirm = input("\nProceed with extraction? (y/n, default: y): ").strip().lower()
    if confirm == 'n':
        print("❌ Extraction cancelled.")
        sys.exit(0)
    
    return pdf_path, output_dir, config, extractions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-Ready PDF Extraction - Extract tables, images, and text optimized for LLM/RAG consumption. Text extraction uses hybrid Unstructured + PyMuPDF classification."
    )
    
    parser.add_argument(
        "source",
        nargs='?',
        help="PDF file or directory containing PDFs (optional for interactive mode)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="multiple_docs_op",
        help="Output directory (default: multiple_docs_op)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # LLM-specific options
    parser.add_argument(
        "--no-pymupdf4llm",
        action="store_false",
        dest="use_pymupdf4llm",
        help="Disable pymupdf4llm (use fallback methods)"
    )
    
    parser.add_argument(
        "--no-base64",
        action="store_false",
        dest="generate_base64",
        help="Don't generate base64 encoded images"
    )
    
    parser.add_argument(
        "--no-markdown",
        action="store_false",
        dest="generate_markdown",
        help="Don't generate markdown tables"
    )
    
    parser.add_argument(
        "--no-json",
        action="store_false",
        dest="generate_json",
        help="Don't generate JSON tables"
    )
    
    parser.add_argument(
        "--no-context",
        action="store_false",
        dest="include_context",
        help="Don't extract surrounding context"
    )
    
    parser.add_argument(
        "--no-ocr",
        action="store_false",
        dest="extract_ocr_text",
        help="Don't extract OCR text from images"
    )
    
    # Table extraction options
    parser.add_argument(
        "--table-strategy",
        choices=["hi_res", "fast"],
        default=DEFAULT_LLM_CONFIG["llm_tables"]["strategy"],
        help="Table extraction strategy (default: hi_res)"
    )
    
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_LLM_CONFIG["llm_tables"]["min_rows"],
        help="Minimum rows to consider a valid table (default: 1)"
    )
    
    parser.add_argument(
        "--min-cols",
        type=int,
        default=DEFAULT_LLM_CONFIG["llm_tables"]["min_cols"],
        help="Minimum columns to consider a valid table (default: 2)"
    )
    
    # Note: Camelot-specific options removed - now using only unstructured.partition_pdf for table extraction
    
    parser.add_argument(
        "--show-table-stats",
        action="store_true",
        help="Show detailed table extraction statistics (engine breakdown, confidence scores)"
    )
    
    # Image extraction options
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=DEFAULT_LLM_CONFIG["llm_images"]["min_size"],
        help="Minimum image dimension in pixels (default: 50)"
    )
    
    parser.add_argument(
        "--image-dpi",
        type=int,
        default=DEFAULT_LLM_CONFIG["llm_images"]["dpi"],
        help="DPI for image extraction (default: 300)"
    )
    
    parser.add_argument(
        "--no-captions",
        action="store_false",
        dest="include_captions",
        help="Don't try to extract image captions"
    )
    
    parser.add_argument(
        "--no-vector-graphics",
        action="store_false",
        dest="extract_vector_graphics",
        help="Don't extract vector graphics (charts, diagrams)"
    )
    
    parser.add_argument(
        "--merge-tolerance",
        type=int,
        default=DEFAULT_LLM_CONFIG["llm_images"]["merge_tolerance"],
        help="Tolerance for merging nearby image rectangles (default: 20)"
    )
    
    # Text extraction options (with hybrid Unstructured + PyMuPDF classification)
    parser.add_argument(
        "--no-headers",
        action="store_false",
        dest="extract_headers",
        help="Don't extract headers separately"
    )
    
    parser.add_argument(
        "--no-footers",
        action="store_false",
        dest="extract_footers",
        help="Don't extract footers separately"
    )
    
    parser.add_argument(
        "--no-captions-text",
        action="store_false",
        dest="extract_captions",
        help="Don't extract captions separately"
    )
    
    parser.add_argument(
        "--no-formatting",
        action="store_false",
        dest="preserve_formatting",
        help="Don't preserve text formatting"
    )
    
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR for scanned documents"
    )
    
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=DEFAULT_LLM_CONFIG["llm_text"]["min_confidence"],
        help="Minimum confidence for OCR results (default: 0.5)"
    )
    
    parser.add_argument(
        "--no-preprocess",
        action="store_false",
        dest="preprocess_text",
        help="Disable enhanced text preprocessing (Unicode normalization, ellipsis conversion)"
    )
    
    parser.add_argument(
        "--no-unicode-normalize",
        action="store_false",
        dest="normalize_unicode",
        help="Disable Unicode character normalization"
    )
    
    parser.add_argument(
        "--no-ellipsis-convert",
        action="store_false",
        dest="convert_ellipsis",
        help="Disable ellipsis to dots conversion"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LLM_CONFIG["logging"]["level"],
        help="Set logging level (default: INFO)"
    )

    # Page range option (e.g., 2-5)
    parser.add_argument(
        "--page-range",
        type=str,
        help="Optional page range to process, e.g., 2-5 (1-indexed)"
    )
    
    # Update manifests option
    parser.add_argument(
        "--update-manifests",
        type=str,
        metavar="OUTPUT_DIR",
        help="Update all existing manifest files in the specified output directory to the new structure with folder paths and file locations"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the LLM-ready extraction CLI."""
    args = parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    set_log_level(log_level)
    
    # Handle manifest update command
    if args.update_manifests:
        print(f"🔄 Updating all manifest files in: {args.update_manifests}")
        result = update_all_manifests(args.update_manifests)
        
        if result["success"]:
            print(f"\n✅ Manifest Update Completed!")
            print("=" * 60)
            print(f"📊 Update Summary:")
            print(f"   📁 Output directory: {Path(args.update_manifests).resolve()}")
            print(f"   📄 Total manifests found: {result['total']}")
            print(f"   ✅ Successfully updated: {result['updated']}")
            if result['failed'] > 0:
                print(f"   ❌ Failed: {result['failed']}")
            print("\n✨ All manifests now include folder paths and file locations!")
        else:
            print(f"\n❌ Manifest update failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
        
        return
    
    # Check if interactive mode or no source provided
    if args.interactive or not args.source:
        pdf_path, output_dir, config, extractions = interactive_mode()
        
        print(f"\n🎯 Selected extractions: {', '.join(extractions)}")
        print(f"📁 Input: {pdf_path}")
        print(f"📤 Output: {output_dir}")
        # Show page range if enabled
        page_range_info = config.get("page_range", {})
        if page_range_info.get("enabled", False):
            print(f"📑 Page range: {page_range_info.get('start_page')}-{page_range_info.get('end_page')}")
        
        # Run LLM pipeline
        summary = run_llm_pipeline(
            source=pdf_path,
            out_root=output_dir,
            config=config
        )
        
        if summary["success"]:
            print(f"\n🎉 LLM-Ready PDF Extraction Completed Successfully!")
            print("=" * 60)
            print(f"📊 Processing Summary:")
            print(f"   ✅ Successfully processed: {summary['processed']}/{summary['total']} PDFs")
            print(f"   📁 Output directory: {Path(output_dir).resolve()}")
            # Show page range if enabled
            page_range_info = config.get("page_range", {})
            if page_range_info.get("enabled", False):
                print(f"   📑 Page range: {page_range_info.get('start_page')}-{page_range_info.get('end_page')}")
            
            # Show extraction results based on what was selected
            if 'tables' in extractions:
                total_tables = sum(pdf.get('tables', 0) for pdf in summary.get('pdfs', []))
                print(f"   📊 LLM-ready tables extracted: {total_tables} (CSV + Markdown + JSON)")
                
                # Show per-PDF table extraction details with engine information
                print(f"\n   📋 Table Extraction Details by PDF:")
                for pdf_info in summary.get('pdfs', []):
                    pdf_name = Path(pdf_info.get('path', '')).name
                    pdf_tables = pdf_info.get('tables', 0)
                    if pdf_tables > 0:
                        engine_info = pdf_info.get('table_engines', {})
                        partition_count = engine_info.get('partition_pdf', pdf_tables)
                        print(f"      📄 {pdf_name}: {pdf_tables} table(s) - 📊 unstructured.partition_pdf: {partition_count}")
                
                # Show table extraction engine statistics
                if 'table_stats' in summary:
                    stats = summary['table_stats']
                    if 'engine_breakdown' in stats:
                        engine_breakdown = stats['engine_breakdown']
                        partition_count = engine_breakdown.get('partition_pdf', 0)
                        if partition_count > 0:
                            print(f"\n   📈 Table Extraction Engine:")
                            print(f"      📊 unstructured.partition_pdf: {partition_count} tables (100%)")
            if 'images' in extractions:
                total_images = sum(pdf.get('images', 0) for pdf in summary.get('pdfs', []))
                print(f"   🖼️  LLM-ready images extracted: {total_images} (PNG/JPEG + Base64 + OCR)")
            if 'text' in extractions:
                total_text = sum(pdf.get('text_blocks', 0) for pdf in summary.get('pdfs', []))
                print(f"   📝 LLM-ready text blocks extracted: {total_text} (Hybrid classification: Unstructured + PyMuPDF)")
            
            # Show extraction times if available
            if 'total_extraction_times' in summary:
                times = summary['total_extraction_times']
                print(f"\n⏱️  Performance Metrics:")
                print(f"   🕐 Total extraction time: {times['total_seconds']:.2f} seconds")
                if 'tables' in extractions:
                    print(f"   📊 Tables processing: {times['tables_seconds']:.2f}s")
                if 'images' in extractions:
                    print(f"   🖼️  Images processing: {times['images_seconds']:.2f}s")
                if 'text' in extractions:
                    print(f"   📝 Text processing: {times['text_seconds']:.2f}s")
                # Show page range in performance metrics if enabled
                page_range_info = config.get("page_range", {})
                if page_range_info.get("enabled", False):
                    print(f"   📑 Pages processed: {page_range_info.get('start_page')}-{page_range_info.get('end_page')} (range)")
            
            print(f"\n📂 Output Structure:")
            print(f"   📁 {Path(output_dir).resolve()}/")
            if 'tables' in extractions:
                print(f"   ├── 📊 tables/ (CSV + Markdown + JSON files)")
            if 'images' in extractions:
                print(f"   ├── 🖼️  images/ (PNG/JPEG + Base64 + OCR files)")
            if 'text' in extractions:
                print(f"   ├── 📝 text/ (Structured + Markdown + Hybrid classification)")
            print(f"   └── 📄 manifest.json (LLM-ready metadata)")
            
            return 0
        else:
            print(f"\n❌ LLM-ready extraction failed: {summary['message']}")
            return 1
    
    # Traditional command line mode
    print("🤖 LLM-Ready PDF Extraction Pipeline")
    print("=" * 50)
    print("📊 Table Extraction: unstructured.partition_pdf")
    print("📝 Text Extraction: Hybrid Unstructured + PyMuPDF classification")
    print("=" * 50)
    
    # Convert paths to absolute paths
    source_path = os.path.abspath(args.source)
    output_path = os.path.abspath(args.output)
    
    print(f"📁 Source: {source_path}")
    print(f"📤 Output: {output_path}")
    if args.page_range:
        print(f"📑 Page range: {args.page_range}")
    
    # Build config from arguments
    config = DEFAULT_LLM_CONFIG.copy()
    
    # LLM-specific options
    config["llm_tables"]["use_pymupdf4llm"] = args.use_pymupdf4llm
    config["llm_tables"]["include_context"] = args.include_context
    config["llm_tables"]["generate_markdown"] = args.generate_markdown
    config["llm_tables"]["generate_json"] = args.generate_json
    
    config["llm_images"]["use_pymupdf4llm"] = args.use_pymupdf4llm
    config["llm_images"]["generate_base64"] = args.generate_base64
    config["llm_images"]["include_context"] = args.include_context
    config["llm_images"]["extract_ocr_text"] = args.extract_ocr_text
    
    config["llm_text"]["use_pymupdf4llm"] = args.use_pymupdf4llm
    config["llm_text"]["extract_tables_as_markdown"] = args.generate_markdown
    config["llm_text"]["extract_images_with_captions"] = args.include_captions
    
    # Standard options
    config["llm_tables"]["strategy"] = args.table_strategy
    config["llm_tables"]["min_rows"] = args.min_rows
    config["llm_tables"]["min_cols"] = args.min_cols
    
    # Note: Camelot options removed - using only unstructured.partition_pdf
    config["llm_images"]["min_size"] = args.min_image_size
    config["llm_images"]["dpi"] = args.image_dpi
    config["llm_images"]["include_captions"] = args.include_captions
    config["llm_images"]["extract_vector_graphics"] = args.extract_vector_graphics
    config["llm_images"]["merge_tolerance"] = args.merge_tolerance
    config["llm_text"]["extract_headers"] = args.extract_headers
    config["llm_text"]["extract_footers"] = args.extract_footers
    config["llm_text"]["extract_captions"] = args.extract_captions
    config["llm_text"]["preserve_formatting"] = args.preserve_formatting
    config["llm_text"]["use_ocr"] = args.use_ocr
    config["llm_text"]["min_confidence"] = args.ocr_confidence
    config["llm_text"]["preprocess_text"] = args.preprocess_text
    config["llm_text"]["normalize_unicode"] = args.normalize_unicode
    config["llm_text"]["convert_ellipsis"] = args.convert_ellipsis
    # Optional page range
    if args.page_range:
        try:
            parts = args.page_range.split("-")
            if len(parts) == 2:
                start = int(parts[0])
                end = int(parts[1])
                if start > 0 and end >= start:
                    config["page_range"] = {
                        "enabled": True,
                        "start_page": start,
                        "end_page": end,
                    }
                else:
                    config["page_range"] = {"enabled": False}
            else:
                config["page_range"] = {"enabled": False}
        except Exception:
            # Ignore malformed input; proceed without page range
            config["page_range"] = {"enabled": False}
    else:
        config["page_range"] = {"enabled": False}
    
    # Run LLM pipeline
    summary = run_llm_pipeline(
        source=source_path,
        out_root=output_path,
        config=config
    )
    
    if summary["success"]:
        print(f"\n🎉 LLM-Ready PDF Extraction Completed Successfully!")
        print("=" * 60)
        print(f"📊 Processing Summary:")
        print(f"   ✅ Successfully processed: {summary['processed']}/{summary['total']} PDFs")
        print(f"   📁 Output directory: {Path(output_path).resolve()}")
        # Show page range if enabled
        page_range_info = config.get("page_range", {})
        if page_range_info.get("enabled", False):
            print(f"   📑 Page range: {page_range_info.get('start_page')}-{page_range_info.get('end_page')}")
        
        # Show extraction results
        total_tables = sum(pdf.get('tables', 0) for pdf in summary.get('pdfs', []))
        total_images = sum(pdf.get('images', 0) for pdf in summary.get('pdfs', []))
        total_text = sum(pdf.get('text_blocks', 0) for pdf in summary.get('pdfs', []))
        
        print(f"   📊 LLM-ready tables extracted: {total_tables} (CSV + Markdown + JSON)")
        print(f"   🖼️  LLM-ready images extracted: {total_images} (PNG/JPEG + Base64 + OCR)")
        print(f"   📝 LLM-ready text blocks extracted: {total_text} (Hybrid classification: Unstructured + PyMuPDF)")
        
        # Show per-PDF table extraction details with engine information
        if total_tables > 0:
            print(f"\n   📋 Table Extraction Details by PDF:")
            for pdf_info in summary.get('pdfs', []):
                pdf_name = Path(pdf_info.get('path', '')).name
                pdf_tables = pdf_info.get('tables', 0)
                if pdf_tables > 0:
                    engine_info = pdf_info.get('table_engines', {})
                    partition_count = engine_info.get('partition_pdf', pdf_tables)
                    print(f"      📄 {pdf_name}: {pdf_tables} table(s) - 📊 unstructured.partition_pdf: {partition_count}")
        
        # Show table extraction engine statistics
        if 'table_stats' in summary:
            stats = summary['table_stats']
            if 'engine_breakdown' in stats:
                engine_breakdown = stats['engine_breakdown']
                partition_count = engine_breakdown.get('partition_pdf', 0)
                if partition_count > 0:
                    print(f"\n   📈 Table Extraction Engine:")
                    print(f"      📊 unstructured.partition_pdf: {partition_count} tables (100%)")
        
        # Show extraction times if available
        if 'total_extraction_times' in summary:
            times = summary['total_extraction_times']
            print(f"\n⏱️  Performance Metrics:")
            print(f"   🕐 Total extraction time: {times['total_seconds']:.2f} seconds")
            print(f"   📊 Tables processing: {times['tables_seconds']:.2f}s")
            print(f"   🖼️  Images processing: {times['images_seconds']:.2f}s")
            print(f"   📝 Text processing: {times['text_seconds']:.2f}s")
            # Show page range in performance metrics if enabled
            page_range_info = config.get("page_range", {})
            if page_range_info.get("enabled", False):
                print(f"   📑 Pages processed: {page_range_info.get('start_page')}-{page_range_info.get('end_page')} (range)")
        
        print(f"\n📂 Output Structure:")
        print(f"   📁 {Path(output_path).resolve()}/")
        print(f"   ├── 📊 tables/ (CSV + Markdown + JSON files)")
        print(f"   ├── 🖼️  images/ (PNG/JPEG + Base64 + OCR files)")
        print(f"   ├── 📝 text/ (Structured + Markdown + Hybrid classification)")
        print(f"   └── 📄 manifest.json (LLM-ready metadata)")
        
        return 0
    else:
        print(f"\n❌ LLM-ready extraction failed: {summary['message']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
