# LLM-Ready PDF Extraction Pipeline

An advanced PDF extraction system optimized for RAG (Retrieval-Augmented Generation) and LLM (Large Language Model) applications. Extracts content in formats specifically designed for AI/ML consumption.

## 🎯 Overview

The LLM-Ready Pipeline provides AI-optimized PDF extraction with:
- **Tables**: CSV + Markdown + JSON formats with caption extraction
- **Images**: PNG/JPEG + Base64 encoding + OCR text with caption extraction
- **Text**: Structured + Markdown format
- **Metadata**: LLM-ready JSON manifests with caption information

## 🚀 Quick Start

### Streamlit Web UI (Recommended for Beginners)
```bash
cd llm_ready_pipeline
python run_streamlit.py
# Or directly:
streamlit run streamlit_app.py
```
Access the web interface at `http://localhost:8501` for an interactive GUI experience.

### Interactive CLI Mode
```bash
cd llm_ready_pipeline
python cli.py --interactive
```

### Command Line Mode
```bash
# Single PDF
python cli.py sample_pdfs/document1.pdf --output llm_output

# Multiple PDFs from folder
python cli.py sample_pdfs/ --output llm_output

# Using short form for output
python cli.py sample_pdfs/document1.pdf -o llm_output
```

## 📋 Features

### Interactive Mode Features
- **📄 PDF Input Selection**: Choose single file or folder
- **🎯 Extraction Options**: Select tables, images, text, or all
- **📁 Path Handling**: Supports relative and absolute paths
- **✅ Confirmation**: Review settings before processing
- **📊 Progress Tracking**: Real-time extraction progress

### LLM-Optimized Extraction
- **📊 Tables**: Multiple formats (CSV + Markdown + JSON) with automatic caption extraction
  - Caption extraction supports hyphen format (`Table-5.1`), space format (`Table 5.1`), and dot format (`Table.5.1`)
  - Captions stored in JSON files, CSV files, and LLM manifest JSON
- **🖼️ Images**: Base64 encoding + OCR text extraction with automatic caption extraction
  - Caption extraction supports hyphen format (`Fig-2.1`, `Figure-2.1`), space format (`Fig 2.1`), and dot format (`Fig.2.1`)
  - Captions stored in LLM manifest JSON and image metadata
- **📝 Text**: Markdown format for LLM consumption
- **🔗 Context**: Surrounding context for better understanding
- **🤖 AI-Ready**: Optimized for RAG and LLM applications

## 🛠️ Installation

```bash
cd llm_ready_pipeline
pip install -r requirements.txt
```

### M1/M2 Mac GPU Support
For M1/M2 Mac users with GPU acceleration:
```bash
pip install -r requirements-m1-gpu.txt
```

### Dependencies
- **PyMuPDF (fitz)**: Fast PDF processing
- **pymupdf4llm**: LLM-optimized extraction
- **pdfplumber**: Table and text extraction
- **unstructured**: Advanced table detection
- **Pillow**: Image processing
- **pytesseract**: OCR capabilities
- **markdown**: Markdown processing
- **pandas**: Data processing for provenance files
- **pyarrow**: Parquet file support
- **streamlit**: Web UI framework (optional, for GUI)

## 📖 Usage Examples

### Interactive Mode
```bash
python cli.py --interactive
```

**Example Session:**
```
🤖 LLM-Ready PDF Extraction - Interactive Mode
============================================================

📄 PDF Input Selection:
1. Single PDF file
2. Folder containing PDFs

Choose input type (1 or 2): 1

📁 Enter path to PDF file: sample_pdfs/document1.pdf

📤 Enter output directory (default: llm_output): 

🎯 LLM Pipeline - Extraction Options:
Available extractions:
📊 Tables (CSV + Markdown + JSON) - Extract tables in multiple formats for LLM consumption
🖼️  Images (PNG/JPEG + Base64 + OCR) - Extract images with Base64 encoding and OCR text
📝 Text (Structured + Markdown) - Extract text in LLM-ready Markdown format
🔧 All extractions - Extract everything in LLM-ready formats

What would you like to extract?
Available options: tables, images, text, all
Enter your choice: all

✅ Configuration Summary:
   📁 Input: /path/to/sample_pdfs/document1.pdf
   📤 Output: /path/to/llm_output
   🎯 Extractions: tables, images, text

Proceed with extraction? (y/n, default: y): y
```

### Command Line Mode
```bash
# Basic LLM extraction
python cli.py sample_pdfs/document1.pdf

# Custom output directory
python cli.py sample_pdfs/document1.pdf --output my_llm_output

# Process entire folder
python cli.py sample_pdfs/ --output batch_llm_output

# Disable pymupdf4llm (fallback mode)
python cli.py sample_pdfs/document1.pdf --no-pymupdf4llm

# Customize LLM settings
python cli.py sample_pdfs/document1.pdf \
    --no-base64 \
    --no-markdown \
    --no-json \
    --no-context

# Table extraction options
python cli.py sample_pdfs/document1.pdf \
    --table-strategy hi_res \
    --min-rows 2 \
    --min-cols 2

# Image extraction options
python cli.py sample_pdfs/document1.pdf \
    --min-image-size 100 \
    --image-dpi 300 \
    --no-captions \
    --no-vector-graphics \
    --merge-tolerance 20

# Text extraction options
python cli.py sample_pdfs/document1.pdf \
    --no-headers \
    --no-footers \
    --no-captions-text \
    --no-formatting \
    --use-ocr \
    --ocr-confidence 0.7 \
    --no-preprocess \
    --no-unicode-normalize \
    --no-ellipsis-convert

# Set logging level
python cli.py sample_pdfs/document1.pdf --log-level DEBUG
```

## 📊 Output Structure

```
llm_output/
├── document1/
│   ├── llm_tables/
│   │   ├── table_p1_i1.csv
│   │   ├── table_p1_i1.md
│   │   ├── table_p1_i1.json  # Contains caption field
│   │   └── ...
│   ├── figures/
│   │   ├── document1_p1_f1.png
│   │   ├── document1_p2_f1.png
│   │   └── ...
│   ├── llm_text/
│   │   ├── document1_llm_caption.txt
│   │   ├── document1_llm_combined.txt
│   │   ├── document1_llm_footer.txt
│   │   ├── document1_llm_header.txt
│   │   ├── document1_llm_main_text.txt
│   │   ├── document1_llm_summary.txt
│   │   └── document1_paragraph_text.jsonl
│   ├── document1_table_provenance.csv  # Contains caption column
│   ├── document1_table_provenance.parquet  # Contains caption field
│   ├── document1_table_provenance.html  # Displays caption in table
│   ├── document1_figure_text.jsonl  # Contains image captions
│   └── llm_manifest.json  # Contains caption in tables[] and images[]
└── llm_index.json
```

**LLM Manifest JSON Structure:**
```json
{
  "pdf": "/absolute/path/to/document.pdf",
  "hash": "sha256_hash_of_pdf",
  "extraction_type": "llm_ready",
  "tables_count": 6,
  "images_count": 2,
  "text_blocks_count": 150,
  "metadata": { ... },
  "tables": [
    {
      "page": 5,
      "index": 1,
      "rows": 10,
      "cols": 4,
      "caption": "Table-5.1: Sample data",
      "has_markdown": true,
      "has_json": true,
      "has_context": true,
      "engine": "partition_pdf",
      "engine_version": "...",
      "confidence": 0.95,
      "csv_path": "/absolute/path/to/llm_tables/table_p5_i1.csv",
      "markdown_path": "/absolute/path/to/llm_tables/table_p5_i1.md",
      "json_path": "/absolute/path/to/llm_tables/table_p5_i1.json",
      "file_paths": {
        "csv": "/absolute/path/to/llm_tables/table_p5_i1.csv",
        "markdown": "/absolute/path/to/llm_tables/table_p5_i1.md",
        "json": "/absolute/path/to/llm_tables/table_p5_i1.json"
      }
    }
  ],
  "images": [
    {
      "page": 2,
      "filename": "p2_f1_figure.png",
      "caption": "Fig-2.1: System architecture",
      "hash": "image_hash",
      "has_base64": true,
      "has_context": true,
      "image_path": "/absolute/path/to/llm_images/p2_f1_figure.png",
      "file_paths": {
        "image": "/absolute/path/to/llm_images/p2_f1_figure.png",
        "figure_text_jsonl": "/absolute/path/to/document_figure_text.jsonl"
      }
    }
  ],
  "text_blocks": [ ... ],
  "file_paths": {
    "tables": {
      "provenance": {
        "csv": "/absolute/path/to/document_table_provenance.csv",
        "parquet": "/absolute/path/to/document_table_provenance.parquet",
        "html": "/absolute/path/to/document_table_provenance.html"
      }
    },
    "images": {
      "figure_text_jsonl": "/absolute/path/to/document_figure_text.jsonl"
    },
    "text": {
      "paragraph_text_jsonl": "/absolute/path/to/llm_text/document_paragraph_text.jsonl",
      "combined_text": "/absolute/path/to/llm_text/document_llm_combined.txt",
      "summary_text": "/absolute/path/to/llm_text/document_llm_summary.txt",
      "text_type_files": {
        "main_text": "/absolute/path/to/llm_text/document_llm_main_text.txt",
        "header": "/absolute/path/to/llm_text/document_llm_header.txt",
        ...
      }
    }
  }
}
```

## ⚙️ Configuration Options

### LLM Table Extraction
- **Strategy**: `hi_res` (high resolution) or `fast` - `--table-strategy`
- **pymupdf4llm**: Use LLM-optimized extraction - `--no-pymupdf4llm` to disable
- **Min Rows**: Minimum rows to consider a valid table - `--min-rows` (default: 1)
- **Min Cols**: Minimum columns to consider a valid table - `--min-cols` (default: 2)
- **Caption Extraction**: Automatically extracts table captions
  - Supports hyphen format: `Table-5.1`, `Table-5.2`
  - Supports space/dot format: `Table 5.1`, `Table.5.1`
  - Searches above tables first (typical location), then below
  - Position-based mapping with confidence scoring
  - Stored in JSON files, CSV files, and LLM manifest JSON
- **Markdown**: Generate Markdown tables - `--no-markdown` to disable
- **JSON**: Generate JSON table data - `--no-json` to disable
- **Context**: Include surrounding context - `--no-context` to disable
- **Provenance**: Automatically generates CSV, Parquet, and HTML provenance files with caption column

### LLM Image Extraction
- **Base64**: Generate Base64 encoded images - `--no-base64` to disable
- **OCR**: Extract text from images - `--no-ocr` to disable
- **Context**: Include surrounding context - `--no-context` to disable
- **Vector graphics**: Extract charts and diagrams - `--no-vector-graphics` to disable
- **Caption Extraction**: Automatically extracts image captions - `--no-captions` to disable
  - Supports hyphen format: `Fig-2.1`, `Figure-2.1`
  - Supports space/dot format: `Fig 2.1`, `Fig.2.1`
  - Searches below images (within 150 pixels) - typical caption location
  - Position-based mapping with confidence scoring
  - OCR fallback: Performs OCR if no text found in text layer
  - Stored in LLM manifest JSON and image metadata
- **Min Size**: Minimum image dimension in pixels - `--min-image-size` (default: 50)
- **DPI**: DPI for image extraction - `--image-dpi` (default: 300)
- **Merge Tolerance**: Tolerance for merging nearby image rectangles - `--merge-tolerance` (default: 20)
- **Figures**: Extracts figures with embedded text to `figures/` directory
- **Figure Text**: Generates `{doc}_figure_text.jsonl` with figure metadata including captions

### LLM Text Extraction
- **pymupdf4llm**: Use LLM-optimized text extraction - `--no-pymupdf4llm` to disable
- **Markdown**: Generate Markdown format (automatic)
- **Tables as Markdown**: Convert tables to Markdown - controlled by `--no-markdown`
- **Images with captions**: Include image references - controlled by `--no-captions`
- **Headers**: Extract headers separately - `--no-headers` to disable
- **Footers**: Extract footers separately - `--no-footers` to disable
- **Captions**: Extract captions separately - `--no-captions-text` to disable
- **Formatting**: Preserve text formatting - `--no-formatting` to disable
- **OCR**: Use OCR for scanned documents - `--use-ocr` to enable
- **OCR Confidence**: Minimum confidence for OCR - `--ocr-confidence` (default: 0.5)
- **Preprocessing**: Enhanced text preprocessing - `--no-preprocess` to disable
- **Unicode Normalization**: Normalize Unicode characters - `--no-unicode-normalize` to disable
- **Ellipsis Conversion**: Convert ellipsis to dots - `--no-ellipsis-convert` to disable
- **Output Files**: Generates separate files for headers, footers, captions, main text, combined text, summary, and paragraph JSONL

## 📈 Performance Metrics

The pipeline provides detailed performance information:

```
🎉 LLM-Ready PDF Extraction Completed Successfully!
============================================================
📊 Processing Summary:
   ✅ Successfully processed: 1/1 PDFs
   📁 Output directory: /path/to/llm_output
   📊 LLM-ready tables extracted: 3 (CSV + Markdown + JSON)
   🖼️  LLM-ready images extracted: 24 (PNG/JPEG + Base64 + OCR)
   📝 LLM-ready text blocks extracted: 102 (Structured + Markdown)

⏱️  Performance Metrics:
   🕐 Total extraction time: 103.10s
   📊 Tables processing: 100.35s
   🖼️  Images processing: 1.83s
   📝 Text processing: 0.92s

📂 Output Structure:
   📁 /path/to/llm_output/
   ├── 📊 llm_tables/ (CSV + Markdown + JSON files)
   ├── 🖼️  figures/ (PNG images with embedded text)
   ├── 📝 llm_text/ (Structured + Markdown files)
   ├── 📋 table_provenance.* (CSV, Parquet, HTML)
   ├── 📄 figure_text.jsonl (Figure metadata)
   └── 📄 llm_manifest.json (LLM-ready metadata)
```

## 🔧 Advanced Usage

### Custom LLM Configuration
```python
from pipeline.llm_runner import run_llm_pipeline

config = {
    "llm_tables": {
        "strategy": "hi_res",
        "use_pymupdf4llm": True,
        "generate_markdown": True,
        "generate_json": True,
        "include_context": True
    },
    "llm_images": {
        "generate_base64": True,
        "extract_ocr_text": True,
        "include_context": True
    },
    "llm_text": {
        "use_pymupdf4llm": True,
        "extract_tables_as_markdown": True,
        "extract_images_with_captions": True
    }
}

result = run_llm_pipeline("sample_pdfs/document1.pdf", "llm_output", config)
```

### Batch Processing
```bash
# Process multiple PDFs with LLM optimization
python cli.py sample_pdfs/ --output batch_llm_output --log-level INFO

# Process with custom LLM settings
python cli.py sample_pdfs/ \
    --output batch_llm_output \
    --table-strategy hi_res \
    --min-rows 2 \
    --min-cols 2 \
    --min-image-size 100 \
    --image-dpi 300 \
    --no-base64 \
    --no-markdown \
    --no-ocr

# Process with OCR for scanned documents
python cli.py sample_pdfs/ \
    --output batch_llm_output \
    --use-ocr \
    --ocr-confidence 0.7

# Process with text preprocessing disabled
python cli.py sample_pdfs/ \
    --output batch_llm_output \
    --no-preprocess \
    --no-unicode-normalize \
    --no-ellipsis-convert
```

## 🤖 LLM Integration Examples

### RAG System Integration
```python
import json
from pathlib import Path

# Load LLM-ready manifest
with open("llm_output/document1/llm_manifest.json", "r") as f:
    manifest = json.load(f)

# Access LLM-ready content
tables = manifest.get("tables", [])
images = manifest.get("images", [])
text_blocks = manifest.get("text_blocks", [])

# Use in RAG system
for table in tables:
    markdown_table = table.get("markdown_content")
    csv_path = table.get("csv_path")
    json_path = table.get("json_path")
    # Feed to RAG system

# Load table provenance
import pandas as pd
provenance_df = pd.read_csv("llm_output/document1/document1_table_provenance.csv")

# Load figure text data
import json
with open("llm_output/document1/document1_figure_text.jsonl", "r") as f:
    for line in f:
        figure_data = json.loads(line)
        # Use figure data in RAG system
```

### Base64 Image Usage
```python
# Load Base64 encoded images
for image in images:
    base64_data = image.get("base64_data")
    ocr_text = image.get("ocr_text")
    
    # Use in LLM context
    context = f"Image: {base64_data}\nOCR Text: {ocr_text}"
```

## 🐛 Troubleshooting

### Common Issues

**pymupdf4llm Not Available:**
```
WARNING - pymupdf4llm not available. Install with: pip install pymupdf4llm
```

**Solution**: Install pymupdf4llm or use fallback mode:
```bash
pip install pymupdf4llm
# or
python cli.py sample_pdfs/document1.pdf --no-pymupdf4llm
```

**Path Not Found Error:**
```
❌ Path not found: sample_pdfs/document1.pdf
💡 Tip: Use relative paths like 'sample_pdfs/document1.pdf'
💡 Current directory: /path/to/llm_ready_pipeline
```

**Solution**: Use correct relative paths or absolute paths.

**Import Errors:**
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

## 📚 API Reference

### Main Functions
- `run_llm_pipeline()`: Main LLM pipeline function
- `extract_llm_ready_tables()`: LLM table extraction
- `extract_figures_with_embedded_text()`: Extract figures with embedded text
- `extract_llm_ready_text()`: LLM text extraction
- `export_table_provenance()`: Export table provenance data
- `export_figure_text_jsonl()`: Export figure metadata as JSONL

### LLM-Specific Features
- **Base64 Encoding**: Images encoded for LLM consumption
- **Markdown Format**: Text and tables in Markdown
- **Context Extraction**: Surrounding context for better understanding
- **OCR Integration**: Text extraction from images
- **Caption Extraction**: Automatic extraction of table and image captions
  - **Unified Pattern Matching**: Supports hyphen format (`Fig-2.1`, `Table-5.1`), space format (`Fig 2.1`), and dot format (`Fig.2.1`)
  - **Position-Based Mapping**: Uses distance and alignment to map captions to tables/images
  - **Confidence Scoring**: Ranks caption candidates based on proximity and horizontal alignment
  - **Storage**: Captions stored in JSON files, CSV files, Parquet files, HTML tables, and LLM manifest JSON
- **JSON Metadata**: Structured metadata for LLM processing including captions
- **Provenance Tracking**: Table and figure provenance with CSV, Parquet, and HTML formats (includes caption column/field)
- **Figure Extraction**: Extracts figures with embedded text using targeted OCR
- **Text Preprocessing**: Unicode normalization, ellipsis conversion, and enhanced preprocessing
- **Multiple Output Formats**: CSV, Markdown, JSON for tables; multiple text files for different content types

## 🎯 Use Cases

### RAG Systems
- **Document Retrieval**: Extract searchable content
- **Context Building**: Create rich context for LLMs
- **Multi-modal Input**: Combine text, tables, and images

### LLM Training
- **Data Preparation**: Prepare training data
- **Format Standardization**: Consistent input formats
- **Quality Control**: Validate extraction quality

### AI Applications
- **Document Analysis**: Analyze PDF content
- **Content Generation**: Generate summaries and insights
- **Question Answering**: Build Q&A systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample PDFs
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.