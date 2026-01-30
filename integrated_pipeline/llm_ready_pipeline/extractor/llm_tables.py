import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json
import re
from io import StringIO
from dataclasses import dataclass
import tempfile
import os

from unstructured.partition.pdf import partition_pdf
from . import TableResult
from utils.logging import get_logger
import pdfplumber

# Import text preprocessing function for table cell data
try:
    from .llm_text import preprocess_text
    TEXT_PREPROCESSING_AVAILABLE = True
except ImportError:
    TEXT_PREPROCESSING_AVAILABLE = False
    def preprocess_text(text: str, normalize_unicode: bool = True, convert_ellipsis: bool = True) -> str:
        """Fallback if preprocessing not available"""
        return text

# Note: Camelot removed - using only unstructured.partition_pdf for table extraction

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

logger = get_logger(__name__)

# ============================================================================
# Table Extraction Constants (tunable parameters)
# ============================================================================
# Note: Camelot constants removed - using only unstructured.partition_pdf

@dataclass
class LLMTableResult:
    """Represents LLM-ready extracted table from a PDF"""
    pdf_path: Path
    page: int
    index: int
    dataframe: pd.DataFrame
    markdown_table: str
    json_table: str
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None
    # New fields for provenance export format
    doc_id: Optional[str] = None
    table_id: Optional[str] = None
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    header_rows: Optional[int] = None
    source: Optional[str] = None  # 'vector' or 'image'
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    confidence: Optional[float] = None
    source_sha256: Optional[str] = None
    
    def __str__(self) -> str:
        return f"LLM Table (Page {self.page}, Table {self.index}): {self.dataframe.shape}"

# ============================================================================
# Camelot Helper Functions
# ============================================================================

def render_page_to_image(pdf_path: Path, page_number: int, dpi: int = 300) -> Optional[Path]:
    """
    Render a PDF page as a PNG image and return the path to the image.
    Prefers PyMuPDF if available, falls back to pdf2image.
    
    Args:
        pdf_path: Path to PDF file
        page_number: Page number (1-indexed)
        dpi: Resolution for rendering (default 300)
    
    Returns:
        Path to temporary PNG file, or None if rendering fails
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot render page to image")
        return None
    
    try:
        # Prefer PyMuPDF for rendering (faster, better quality)
        if PYMUPDF_AVAILABLE and fitz is not None:
            doc = fitz.open(str(pdf_path))
            if page_number < 1 or page_number > len(doc):
                doc.close()
                return None
            
            page = doc[page_number - 1]
            # Create transformation matrix for DPI scaling
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_path = Path(temp_file.name)
            pix.save(str(temp_path))
            doc.close()
            
            return temp_path
        # Fallback to pdf2image
        elif PDF2IMAGE_AVAILABLE and convert_from_path is not None:
            images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number)
            if not images:
                return None
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_path = Path(temp_file.name)
            images[0].save(str(temp_path), 'PNG')
            return temp_path
        else:
            logger.warning("Neither PyMuPDF nor pdf2image available for page rendering")
            return None
    except Exception as e:
        logger.debug(f"Failed to render page {page_number} to image: {e}")
        return None

def preprocess_table_image(pil_image: Image.Image) -> Optional[Image.Image]:
    """
    Preprocess a table image to enhance table structure detection.
    Converts to grayscale, applies adaptive threshold, and morphological operations
    to strengthen horizontal/vertical lines.
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Processed PIL Image, or None if processing fails
    """
    if not PIL_AVAILABLE or not CV2_AVAILABLE:
        return pil_image  # Return original if processing unavailable
    
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(pil_image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to strengthen lines
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Convert back to PIL Image
        processed_img = Image.fromarray(table_mask)
        return processed_img
    except Exception as e:
        logger.debug(f"Failed to preprocess table image: {e}")
        return pil_image  # Return original on error

def camelot_area_from_bbox(bbox: List[float], page_height: float) -> str:
    """
    Convert a bounding box from PDF coordinates to Camelot area string.
    Camelot uses bottom-left origin, so we need to invert Y coordinates.
    
    Args:
        bbox: Bounding box [x0, y0, x1, y1] in PDF coordinates (top-based)
        page_height: Height of the page in PDF units
    
    Returns:
        Camelot area string: "x0,bottom_y,x1,top_y"
    """
    x0, y0, x1, y1 = bbox
    # Convert top-based coordinates to bottom-based (Camelot format)
    bottom_y = page_height - y1
    top_y = page_height - y0
    return f"{x0},{bottom_y},{x1},{top_y}"

def compute_table_confidence(parsing_report: dict, df: pd.DataFrame, engine_priority: float = 1.0) -> float:
    """
    Compute a confidence score (0-100) for a table extraction result.
    
    Args:
        parsing_report: Camelot parsing report dictionary
        df: Extracted DataFrame
        engine_priority: Priority multiplier for the engine (default 1.0)
    
    Returns:
        Confidence score between 0 and 100
    """
    try:
        # Get accuracy from parsing report (typically 0-100)
        accuracy = float(parsing_report.get('accuracy', 0))
        
        # Compute non-empty cell ratio
        if df.size > 0:
            non_empty_count = df.count().sum()
            non_empty_ratio = non_empty_count / df.size
        else:
            non_empty_ratio = 0.0
        
        # Weighted formula: 60% accuracy, 40% non-empty ratio, 10% engine priority
        score = 0.6 * accuracy + 40.0 * non_empty_ratio + 10.0 * engine_priority
        
        # Clamp to 0-100
        score = max(0.0, min(100.0, score))
        return score
    except Exception as e:
        logger.debug(f"Failed to compute table confidence: {e}")
        return 0.0

def run_camelot_on_page(
    pdf_path: Path, 
    page_number: int, 
    table_area: Optional[str] = None, 
    dpi: int = 300, 
    try_lattice_first: bool = True
) -> List[Dict[str, Any]]:
    """
    Run Camelot table extraction on a specific page.
    Tries 'lattice' flavor first, falls back to 'stream' if requested.
    
    Args:
        pdf_path: Path to PDF file
        page_number: Page number (1-indexed, Camelot uses 1-indexed)
        table_area: Optional area string in Camelot format "x0,y0,x1,y1"
        dpi: DPI for extraction
        try_lattice_first: Whether to try lattice flavor first
    
    Returns:
        List of dictionaries with keys: 'df', 'parsing_report', 'confidence', 'camelot_table'
    """
    # Camelot is deprecated - this function is kept for compatibility but not used
    try:
        import camelot
        CAMELOT_AVAILABLE = True
    except ImportError:
        CAMELOT_AVAILABLE = False
        camelot = None
    
    if not CAMELOT_AVAILABLE or camelot is None:
        logger.debug("Camelot not available")
        return []
    
    results = []
    
    try:
        # Prepare Camelot parameters
        camelot_kwargs = {
            'pages': str(page_number),
            'flavor': 'lattice' if try_lattice_first else 'stream',
            'dpi': dpi
        }
        
        if table_area:
            camelot_kwargs['table_areas'] = [table_area]
        
        # Try lattice first if requested
        if try_lattice_first:
            try:
                tables = camelot.read_pdf(str(pdf_path), **camelot_kwargs)
                for table in tables:
                    df = table.df
                    parsing_report = table.parsing_report
                    confidence = compute_table_confidence(parsing_report, df, engine_priority=1.0)
                    results.append({
                        'df': df,
                        'parsing_report': parsing_report,
                        'confidence': confidence,
                        'camelot_table': table,
                        'flavor': 'lattice'
                    })
                if results:
                    logger.debug(f"Camelot lattice found {len(results)} tables on page {page_number}")
                    return results
            except Exception as e:
                logger.debug(f"Camelot lattice failed on page {page_number}: {e}")
        
        # Fallback to stream flavor
        camelot_kwargs['flavor'] = 'stream'
        try:
            tables = camelot.read_pdf(str(pdf_path), **camelot_kwargs)
            for table in tables:
                df = table.df
                parsing_report = table.parsing_report
                confidence = compute_table_confidence(parsing_report, df, engine_priority=0.9)  # Slightly lower priority for stream
                results.append({
                    'df': df,
                    'parsing_report': parsing_report,
                    'confidence': confidence,
                    'camelot_table': table,
                    'flavor': 'stream'
                })
            if results:
                logger.debug(f"Camelot stream found {len(results)} tables on page {page_number}")
        except Exception as e:
            logger.debug(f"Camelot stream failed on page {page_number}: {e}")
    
    except Exception as e:
        logger.debug(f"Camelot extraction failed on page {page_number}: {e}")
    
    return results

def extract_cell_text_with_tesseract(cell_image: Image.Image, psm: int = 6) -> str:
    """
    Extract text from a cell image using Tesseract OCR.
    
    Args:
        cell_image: PIL Image of the cell
        psm: Tesseract PSM mode (default 6 for uniform block)
    
    Returns:
        Extracted text string
    """
    if not TESSERACT_AVAILABLE or not PIL_AVAILABLE or pytesseract is None:
        return ""
    
    try:
        config = f"--oem 3 --psm {psm}"
        text = pytesseract.image_to_string(cell_image, config=config)
        # Minimal cleaning: replace newlines with spaces and strip
        text = text.replace('\n', ' ').replace('\r', ' ').strip()
        return text
    except Exception as e:
        logger.debug(f"Tesseract OCR failed for cell: {e}")
        return ""

def pdf_bbox_to_pixel_bbox(
    pdf_bbox: Tuple[float, float, float, float],
    page_width_pdf: float,
    page_height_pdf: float,
    image_width: int,
    image_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert a bounding box from PDF coordinates to pixel coordinates.
    
    Args:
        pdf_bbox: Bounding box (x0, y0, x1, y1) in PDF coordinates
        page_width_pdf: Page width in PDF units
        page_height_pdf: Page height in PDF units
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Pixel bounding box (x0, y0, x1, y1) as integers
    """
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = pdf_bbox
    
    # Calculate scaling factors
    scale_x = image_width / page_width_pdf
    scale_y = image_height / page_height_pdf
    
    # Convert coordinates (PDF uses top-left origin, same as images)
    x0_px = int(x0_pdf * scale_x)
    y0_px = int(y0_pdf * scale_y)
    x1_px = int(x1_pdf * scale_x)
    y1_px = int(y1_pdf * scale_y)
    
    # Ensure coordinates are within image bounds
    x0_px = max(0, min(x0_px, image_width))
    y0_px = max(0, min(y0_px, image_height))
    x1_px = max(0, min(x1_px, image_width))
    y1_px = max(0, min(y1_px, image_height))
    
    return (x0_px, y0_px, x1_px, y1_px)

def _get_unified_caption_patterns():
    """
    Get unified caption patterns that support both hyphen and space/dot formats.
    Supports: Fig-2.1, Fig-2.2, Table-5.1, Table-5.2, Fig 2.1, Fig.2.1, etc.
    """
    return [
        # Hyphen format: Fig-2.1, Table-5.1, Figure-2.1
        r'^(Fig(ure)?|Table|Figure)\s*-\s*[\d\.]+.*',
        # Space/dot format: Fig 2.1, Fig.2.1, Table 5.1
        r'^(Fig(ure)?|Table|Figure)\s*\.?\s*[\d\.]+.*',
        # Generic sentence starting with capital (fallback)
        r'^[A-Z][^.]*\.$',
    ]

def _is_table_of_contents(df: pd.DataFrame, page_text: Optional[str] = None, page_number: int = 1) -> bool:
    """
    Detect if a table is likely a table of contents.
    
    Args:
        df: DataFrame to check
        page_text: Optional text from the page (for context)
        page_number: Page number (TOC is usually on early pages)
    
    Returns:
        True if the table appears to be a table of contents
    """
    if df is None or df.empty:
        return False
    
    # TOC is typically on pages 1-5
    if page_number > 5:
        return False
    
    # Check if page text contains TOC indicators
    if page_text:
        toc_indicators = [
            'table of contents',
            'contents',
            'table des matières',
            'inhaltsverzeichnis',
            'indice',
            'índice'
        ]
        page_text_lower = page_text.lower()
        if any(indicator in page_text_lower for indicator in toc_indicators):
            return True
    
    # Check table characteristics typical of TOC
    # TOC usually has:
    # 1. Page numbers in the last column or rightmost columns
    # 2. Text in left columns (chapter/section names)
    # 3. Dots/leaders connecting text to page numbers
    # 4. Mostly numeric values in the last column
    
    num_cols = len(df.columns)
    if num_cols < 2:
        return False
    
    # Check if last column contains mostly page numbers
    last_col = df.iloc[:, -1]
    if last_col.dtype == 'object':
        # Check if last column values look like page numbers
        numeric_count = 0
        total_count = 0
        for val in last_col:
            if pd.notna(val):
                val_str = str(val).strip()
                total_count += 1
                # Check if it's a number or contains dots (like "1.1.1")
                if re.match(r'^\d+(\.\d+)*$', val_str) or re.match(r'^\d+$', val_str):
                    numeric_count += 1
        
        if total_count > 0 and numeric_count / total_count > 0.7:
            # Last column is mostly numbers - could be TOC
            
            # Check if other columns contain text (not numbers)
            text_cols = df.iloc[:, :-1]
            text_ratio = 0
            text_total = 0
            for col in text_cols.columns:
                for val in text_cols[col]:
                    if pd.notna(val):
                        val_str = str(val).strip()
                        text_total += 1
                        # Check if it's text (not just numbers)
                        if not re.match(r'^\d+(\.\d+)*$', val_str) and len(val_str) > 2:
                            text_ratio += 1
            
            if text_total > 0 and text_ratio / text_total > 0.6:
                # Has text in left columns and numbers in right column - likely TOC
                return True
    
    # Check for dot leaders (common in TOC)
    # TOC often has "Chapter 1 ................ 5" format
    for col in df.columns:
        for val in df[col]:
            if pd.notna(val):
                val_str = str(val)
                # Check for multiple dots/periods (leaders)
                if val_str.count('.') > 3 or val_str.count('·') > 2:
                    # Check if it's in a pattern typical of TOC
                    if re.search(r'[A-Za-z].*\.{3,}.*\d', val_str) or re.search(r'[A-Za-z].*·{2,}.*\d', val_str):
                        return True
    
    # Check if table has typical TOC structure:
    # - First column: Chapter/Section numbers or names
    # - Last column: Page numbers
    # - Middle columns: Optional (dots, sub-sections, etc.)
    if num_cols >= 2:
        first_col = df.iloc[:, 0]
        last_col = df.iloc[:, -1]
        
        # Check if first column has chapter-like patterns
        first_col_text = ' '.join([str(v) for v in first_col if pd.notna(v)])
        if re.search(r'(chapter|section|part|appendix)\s*\d+', first_col_text, re.IGNORECASE):
            # Check if last column has page numbers
            last_col_numeric = sum(1 for v in last_col if pd.notna(v) and re.match(r'^\d+$', str(v).strip()))
            if last_col_numeric > len(last_col) * 0.7:
                return True
    
    return False

def _extract_table_caption_from_pdf(pdf_path: Path, page_num: int, table_bbox: List[float], doc=None) -> Optional[str]:
    """
    Extract table caption from PDF using PyMuPDF with enhanced pattern matching.
    Tables typically have captions above them, so we search above first, then below.
    Captions are preprocessed to handle special characters and Unicode.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        table_bbox: Bounding box of table [x0, y0, x1, y1] in PDF coordinates
        doc: Optional PyMuPDF document object (if provided, won't open/close the PDF)
    
    Returns:
        Preprocessed caption text if found, None otherwise
    """
    if not PYMUPDF_AVAILABLE or fitz is None:
        return None
    
    should_close = False
    try:
        if doc is None:
            doc = fitz.open(str(pdf_path))
            should_close = True
        
        if page_num < 1 or page_num > len(doc):
            if should_close:
                doc.close()
            return None
        
        page = doc[page_num - 1]
        region = fitz.Rect(table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3])
        
        # Import caption extraction function from llm_images
        from .llm_images import _find_caption_in_text_layer
        
        # Search above first (typical for tables), then below
        caption = _find_caption_in_text_layer(page, region, search_below=False, search_above=True, max_distance=150)
        if not caption:
            caption = _find_caption_in_text_layer(page, region, search_below=True, search_above=False, max_distance=150)
        
        if should_close:
            doc.close()
        return caption
        
    except Exception as e:
        logger.debug(f"Failed to extract table caption from PDF: {e}")
        if 'doc' in locals() and should_close:
            try:
                doc.close()
            except:
                pass
        return None

def _extract_table_caption_from_pdfplumber(pdf_path: Path, page_num: int, table_obj, doc=None) -> Optional[str]:
    """
    Extract table caption from pdfplumber table object.
    Converts pdfplumber table bbox to PyMuPDF format and extracts caption.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        table_obj: pdfplumber table object
        doc: Optional PyMuPDF document object (if provided, won't open/close the PDF)
    
    Returns:
        Caption text if found, None otherwise
    """
    if not PYMUPDF_AVAILABLE or fitz is None:
        return None
    
    try:
        # Get table bounding box from pdfplumber
        if hasattr(table_obj, 'bbox'):
            bbox = table_obj.bbox
            # Convert pdfplumber bbox (x0, top, x1, bottom) to fitz format [x0, y0, x1, y1]
            table_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            return _extract_table_caption_from_pdf(pdf_path, page_num, table_bbox, doc=doc)
    except Exception as e:
        logger.debug(f"Failed to extract table caption from pdfplumber: {e}")
    
    return None

def _populate_table_provenance_fields(result: LLMTableResult, pdf_path: Path) -> LLMTableResult:
    """Populate the new provenance fields for a table result"""
    import hashlib
    
    # Generate doc_id from PDF filename
    doc_id = pdf_path.stem
    
    # Generate source_sha256
    try:
        with open(pdf_path, 'rb') as f:
            source_sha256 = hashlib.sha256(f.read()).hexdigest()
    except:
        source_sha256 = None
    
    # Generate table_id
    table_id = f"{doc_id}_p{result.page}_t{result.index}"
    
    # Set engine and version based on extraction method
    # Only using unstructured.partition_pdf now
    if result.metadata and result.metadata.get('extraction_method') == 'partition_pdf':
        engine = "partition_pdf"
        try:
            import unstructured
            engine_version = unstructured.__version__
        except:
            engine_version = "unknown"
    else:
        # Default to partition_pdf
        engine = "partition_pdf"
        try:
            import unstructured
            engine_version = unstructured.__version__
        except:
            engine_version = "unknown"
    
    # Determine source type (vector vs image)
    # Check if extraction used render+OCR (indicated in metadata)
    if result.metadata and result.metadata.get('used_ocr_fallback', False):
        source = "image"
    else:
        source = "vector"  # Default to vector
    
    # Extract header rows count
    header_rows = 1  # Default, could be enhanced to detect multiple header rows
    
    # Extract caption - prefer explicit caption field, then context
    caption = result.caption if result.caption else (result.context if result.context else None)
    
    # Extract bbox if available in metadata
    bbox = None
    if result.metadata and 'bbox' in result.metadata:
        bbox = result.metadata['bbox']
    elif result.structured_data and 'bbox' in result.structured_data:
        bbox = result.structured_data['bbox']
    
    # Set confidence if available
    confidence = result.metadata.get('confidence') if result.metadata else None
    
    # Update the result with new fields
    result.doc_id = doc_id
    result.table_id = table_id
    result.bbox = bbox
    result.caption = caption
    result.header_rows = header_rows
    result.source = source
    result.engine = engine
    result.engine_version = engine_version
    result.confidence = confidence
    result.source_sha256 = source_sha256
    
    return result

def extract_llm_ready_tables(
    pdf_path: str | Path,
    output_dir: str | Path,
    strategy: str = 'hi_res',
    min_rows: int = 1,
    min_cols: int = 2,
    extract_page_headers: bool = True,
    extract_sections: bool = True,
    skip_empty_rows: bool = True,
    max_table_nesting_level: int = 5,
    use_pymupdf4llm: bool = True,
    include_context: bool = True,
    generate_markdown: bool = True,
    generate_json: bool = True,
    start_page: int | None = None,
    end_page: int | None = None,
    camelot_dpi: int | None = None,
    camelot_confidence_threshold: float | None = None,
    preprocess_text_enabled: bool = True,
    normalize_unicode: bool = True,
    convert_ellipsis: bool = True
) -> List[LLMTableResult]:
    """
    Extract LLM-ready tables from a PDF using unstructured.partition_pdf.
    Maintains the same LLMTableResult output structure.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted tables
        strategy: Extraction strategy ('hi_res', 'fast', 'ocr_only')
        min_rows: Minimum number of rows for a valid table
        min_cols: Minimum number of columns for a valid table
        extract_page_headers: Whether to extract page headers
        extract_sections: Whether to extract section information
        skip_empty_rows: Whether to skip empty rows
        max_table_nesting_level: Maximum table nesting level
        use_pymupdf4llm: (Deprecated - kept for compatibility, not used)
        include_context: Whether to include surrounding context
        generate_markdown: Whether to generate markdown format
        generate_json: Whether to generate JSON format
        start_page: Start page number (1-indexed, None for all pages)
        end_page: End page number (1-indexed, None for all pages)
        camelot_dpi: (Deprecated - kept for compatibility, not used)
        camelot_confidence_threshold: (Deprecated - kept for compatibility, not used)
        preprocess_text_enabled: Whether to apply text preprocessing to table cell values
        normalize_unicode: Whether to normalize Unicode characters in table cells
        convert_ellipsis: Whether to convert ellipsis to dots in table cells
        
    Returns:
        List of LLMTableResult objects
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting LLM-ready tables from {pdf_path.name} using unstructured.partition_pdf with {strategy} strategy")

    results: List[LLMTableResult] = []
    original_table_count = 0  # Track original count for logging

    # ========================================================================
    # Use unstructured.partition_pdf for table extraction
    # ========================================================================
    try:
        # Build partition_pdf parameters
        partition_kwargs = {
            'filename': str(pdf_path),
            'infer_table_structure': True,
            'strategy': strategy,
            'extract_page_headers': extract_page_headers,
            'max_table_nesting_level': max_table_nesting_level,
            'include_page_breaks': True,
        }
        
        # Add page range if specified
        if start_page is not None or end_page is not None:
            # Note: partition_pdf uses 0-indexed pages, but we use 1-indexed
            # We'll filter after extraction if page range is specified
            pass
        
        logger.info(f"Running unstructured.partition_pdf extraction for {pdf_path.name}")
        elements = partition_pdf(**partition_kwargs)
        
        # Filter for table elements
        table_elements = [el for el in elements if el.category == 'Table']
        original_table_count = len(table_elements)
        logger.info(f"Found {original_table_count} table elements in {pdf_path.name}")
        
        # Process each table element
        filtered_count = 0
        for idx, table_el in enumerate(table_elements, start=1):
            try:
                # Get page number from metadata
                page_number = getattr(table_el.metadata, 'page_number', None) or 1
                
                # Filter by page range if specified
                if start_page is not None and page_number < start_page:
                    filtered_count += 1
                    continue
                if end_page is not None and page_number > end_page:
                    filtered_count += 1
                    continue
                
                # Extract table HTML and convert to DataFrame
                html = table_el.metadata.text_as_html
                if not html:
                    logger.debug(f"Table {idx} on page {page_number} has no HTML content, skipping")
                    continue
                
                df = pd.read_html(StringIO(html))[0]
                
                # Clean DataFrame with text preprocessing
                df = _clean_dataframe_for_llm(
                    df, 
                    skip_empty_rows=skip_empty_rows,
                    preprocess_text_enabled=preprocess_text_enabled,
                    normalize_unicode=normalize_unicode,
                    convert_ellipsis=convert_ellipsis
                )
                
                # Validate table size
                if df is None or df.empty or len(df) < min_rows or len(df.columns) < min_cols:
                    logger.debug(f"Table {idx} on page {page_number} does not meet size requirements ({df.shape if df is not None and not df.empty else 'empty'})")
                    continue
                
                # Skip table of contents
                try:
                    # Get page text for TOC detection
                    page_text = None
                    if PYMUPDF_AVAILABLE and fitz is not None:
                        try:
                            doc = fitz.open(str(pdf_path))
                            if page_number <= len(doc):
                                page = doc[page_number - 1]
                                page_text = page.get_text()
                            doc.close()
                        except:
                            pass
                    
                    if _is_table_of_contents(df, page_text=page_text, page_number=page_number):
                        logger.debug(f"Skipping table {idx} on page {page_number} - detected as table of contents")
                        continue
                except Exception as e:
                    logger.debug(f"Error checking for table of contents: {e}")
                    # Continue with extraction if TOC detection fails
                
                # Generate multiple formats
                markdown_table = _dataframe_to_markdown(df) if generate_markdown else ""
                json_table = _dataframe_to_json(df) if generate_json else ""
                context = _extract_table_context(table_el) if include_context else None
                
                # Extract caption from the page
                table_caption = _extract_table_caption_by_page(pdf_path, page_number)
                
                # Extract metadata
                metadata = _extract_table_metadata(table_el)
                metadata['extraction_method'] = 'partition_pdf'
                metadata['table_index'] = idx
                
                # Extract bbox if available
                bbox = None
                if hasattr(table_el.metadata, 'coordinates') and table_el.metadata.coordinates:
                    coords = table_el.metadata.coordinates
                    if hasattr(coords, 'x1') and hasattr(coords, 'y1') and hasattr(coords, 'x2') and hasattr(coords, 'y2'):
                        bbox = [coords.x1, coords.y1, coords.x2, coords.y2]
                        metadata['bbox'] = bbox
                
                # Create structured data
                structured_data = _extract_structured_table_data(df)
                if bbox:
                    structured_data['bbox'] = bbox
                
                # Create LLMTableResult
                result = LLMTableResult(
                    pdf_path=pdf_path,
                    page=page_number,
                    index=idx,
                    dataframe=df,
                    markdown_table=markdown_table,
                    json_table=json_table,
                    context=context,
                    caption=table_caption,
                    metadata=metadata,
                    structured_data=structured_data
                )
                results.append(result)
                logger.info(f"Extracted table {idx} on page {page_number}: {df.shape} (rows × cols) with caption: {table_caption[:50] if table_caption else 'None'}")
                
            except Exception as e:
                logger.warning(f"Failed to parse table element {idx}: {e}")
                continue
                
        if results:
            logger.info(f"Successfully extracted {len(results)} LLM-ready tables from {pdf_path.name} using unstructured.partition_pdf")
        else:
            logger.info(f"No valid tables found in {pdf_path.name} using unstructured.partition_pdf")
                    
    except Exception as e:
        logger.error(f"unstructured.partition_pdf extraction failed for {pdf_path.name}: {e}", exc_info=True)

    if not results:
        logger.info(f"No LLM-ready tables found in {pdf_path.name}")
    
    # Populate provenance fields for all results
    for i, result in enumerate(results):
        results[i] = _populate_table_provenance_fields(result, pdf_path)
    
    # Log filtering results if page range was specified
    if start_page is not None or end_page is not None:
        logger.info(f"Filtered tables: {original_table_count} -> {len(results)} (pages {start_page or 1}-{end_page or 'all'})")
    
    return results

def _extract_table_caption_by_page(pdf_path: Path, page_num: int) -> Optional[str]:
    """
    Extract table captions from a page by searching all text blocks.
    Used when table bbox is not available (e.g., from pymupdf4llm).
    Captions are preprocessed to handle special characters and Unicode.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
    
    Returns:
        Preprocessed table caption if found, None otherwise
    """
    if not PYMUPDF_AVAILABLE or fitz is None:
        return None
    
    try:
        # Import preprocessing function
        from extractor.llm_text import preprocess_text
        
        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        
        page = doc[page_num - 1]
        blocks = page.get_text("blocks")
        caption_patterns = _get_unified_caption_patterns()
        
        # Look for table captions (not figure captions)
        # Patterns are: Fig(ure)?|Table|Figure, so group(1) captures the full match
        for x0, y0, x1, y1, text, _, _ in blocks:
            clean_text = " ".join(text.split())
            for pattern in caption_patterns:
                match = re.match(pattern, clean_text, re.IGNORECASE)
                if match:
                    # Check if it's a table caption (starts with "Table")
                    matched_text = match.group(0)
                    if matched_text and matched_text.strip().upper().startswith('TABLE'):
                        doc.close()
                        # Apply text preprocessing to handle special characters and Unicode
                        preprocessed_caption = preprocess_text(clean_text, normalize_unicode=True, convert_ellipsis=True)
                        return preprocessed_caption
        
        doc.close()
        return None
        
    except Exception as e:
        logger.debug(f"Failed to extract table caption by page: {e}")
        return None

def _extract_tables_with_pymupdf4llm(
    pdf_path: Path,
    output_dir: Path,
    include_context: bool = True,
    generate_markdown: bool = True,
    generate_json: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMTableResult]:
    """Extract tables using pymupdf4llm for LLM-ready content"""
    results = []
    
    try:
        # Extract markdown content with tables
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        
        # Parse markdown to extract tables
        tables_data = _parse_markdown_tables(markdown_text, pdf_path)
        
        for table_data in tables_data:
            if table_data['dataframe'] is not None and not table_data['dataframe'].empty:
                page_num = table_data['page']
                
                # Filter by page range if specified
                if start_page is not None and page_num < start_page:
                    continue
                if end_page is not None and page_num > end_page:
                    continue
                
                df = table_data['dataframe']
                
                # Use pymupdf4llm metadata (Camelot already ran first, so this is a fallback)
                metadata = table_data.get('metadata', {})
                
                # Try to extract caption from the page (since we don't have bbox)
                # Caption extraction works even with page ranges - it searches on the specific page
                table_caption = _extract_table_caption_by_page(pdf_path, page_num)
                
                result = LLMTableResult(
                    pdf_path=pdf_path,
                    page=page_num,
                    index=table_data['index'],
                    dataframe=df,
                    markdown_table=_dataframe_to_markdown(df) if generate_markdown else table_data.get('markdown', ''),
                    json_table=_dataframe_to_json(df) if generate_json else "",
                    context=table_data.get('context', ''),
                    caption=table_caption,
                    metadata=metadata,
                    structured_data=_extract_structured_table_data(df)
                )
                results.append(result)
        
    except Exception as e:
        logger.warning(f"pymupdf4llm table extraction failed: {e}")
    
    return results

def _extract_tables_with_pdfplumber(
    pdf_path: Path,
    min_rows: int = 1,
    min_cols: int = 2,
    skip_empty_rows: bool = True,
    generate_markdown: bool = True,
    generate_json: bool = True,
    include_context: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMTableResult]:
    """
    Extract tables using pdfplumber with merged cell detection (FAST fallback).
    
    This is optimized for born-digital PDFs and provides merged cell detection.
    """
    results: List[LLMTableResult] = []
    
    # Open PyMuPDF document once to reuse for caption extraction
    fitz_doc = None
    if PYMUPDF_AVAILABLE and fitz is not None:
        try:
            fitz_doc = fitz.open(str(pdf_path))
        except Exception:
            pass
    
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            sp = max(1, start_page) if start_page else 1
            ep = min(total, end_page) if end_page else total
            
            tbl_idx = 0
            for p_idx in range(sp, ep + 1):
                try:
                    page = pdf.pages[p_idx - 1]
                    table_objects = page.find_tables()
                except Exception:
                    table_objects = []
                
                for tbl_idx_local, table_obj in enumerate(table_objects):
                    # ========================================================================
                    # PDFPLUMBER EXTRACTION (Pure fallback - Camelot already ran first)
                    # ========================================================================
                    # Extract table with merged cells using pdfplumber
                    result_tuple = _extract_table_with_merged_cells(
                        page, tbl_idx_local, min_rows, min_cols, skip_empty_rows
                    )
                    if result_tuple is None:
                        continue
                    
                    df, merged_cells = result_tuple
                    
                    # Skip table of contents
                    try:
                        # Get page text for TOC detection
                        page_text = None
                        if fitz_doc is not None and p_idx <= len(fitz_doc):
                            try:
                                page_fitz = fitz_doc[p_idx - 1]
                                page_text = page_fitz.get_text()
                            except:
                                pass
                        
                        if _is_table_of_contents(df, page_text=page_text, page_number=p_idx):
                            logger.debug(f"Skipping pdfplumber table p{p_idx}_i{tbl_idx_local} - detected as table of contents")
                            continue
                    except Exception as e:
                        logger.debug(f"Error checking for table of contents: {e}")
                        # Continue with extraction if TOC detection fails
                    
                    tbl_idx += 1
                    
                    # Generate multiple formats
                    markdown_table = _dataframe_to_markdown(df) if generate_markdown else ""
                    json_table = _dataframe_to_json(df) if generate_json else ""
                    context = _extract_pdfplumber_table_context(page, table_obj) if include_context else None
                    
                    # Extract table caption with enhanced pattern matching (reuse fitz_doc)
                    table_caption = _extract_table_caption_from_pdfplumber(pdf_path, p_idx, table_obj, doc=fitz_doc)
                    
                    # Create metadata with merged cells info
                    metadata = {
                        "extraction_method": "pdfplumber",
                        "table_index": tbl_idx,
                        "merged_cells": merged_cells if merged_cells else []
                    }
                    
                    # Create structured data with merged cells
                    structured_data = _extract_structured_table_data(df)
                    if merged_cells:
                        structured_data['merged_cells'] = merged_cells
                    
                    result = LLMTableResult(
                        pdf_path=pdf_path,
                        page=p_idx,
                        index=tbl_idx,
                        dataframe=df,
                        markdown_table=markdown_table,
                        json_table=json_table,
                        context=context,
                        caption=table_caption,
                        metadata=metadata,
                        structured_data=structured_data
                    )
                    results.append(result)
                    logger.debug(f"pdfplumber extracted table p{p_idx}_i{tbl_idx} with {len(merged_cells) if merged_cells else 0} merged cells and caption: {table_caption[:50] if table_caption else 'None'}")
            
            if results:
                logger.info(f"pdfplumber found {len(results)} tables in range {sp}-{ep} for {pdf_path.name}")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed for {pdf_path.name}: {e}")
    finally:
        # Close PyMuPDF document if we opened it
        if fitz_doc is not None:
            try:
                fitz_doc.close()
            except:
                pass
    
    return results

def _detect_merged_cells(pdfplumber_table) -> List[Dict[str, Any]]:
    """
    Detect merged cells from pdfplumber table object.
    
    Args:
        pdfplumber_table: Table object from pdfplumber.find_tables()
        
    Returns:
        List of merged cell info: [{"row": int, "col": int, "rowspan": int, "colspan": int}, ...]
    """
    merged_cells = []
    try:
        if not hasattr(pdfplumber_table, 'cells') or not pdfplumber_table.cells:
            return merged_cells
        
        cells = pdfplumber_table.cells
        rows = len(cells)
        if rows == 0:
            return merged_cells
        
        cols = len(cells[0]) if cells[0] else 0
        if cols == 0:
            return merged_cells
        
        # Track which cells have been processed
        processed = set()
        
        for row_idx in range(rows):
            for col_idx in range(cols):
                if (row_idx, col_idx) in processed:
                    continue
                
                cell = cells[row_idx][col_idx] if row_idx < len(cells) and col_idx < len(cells[row_idx]) else None
                if cell is None:
                    continue
                
                # Get cell coordinates
                if hasattr(cell, 'bbox'):
                    x0, y0, x1, y1 = cell.bbox
                elif hasattr(cell, 'x0') and hasattr(cell, 'y0') and hasattr(cell, 'x1') and hasattr(cell, 'y1'):
                    x0, y0, x1, y1 = cell.x0, cell.y0, cell.x1, cell.y1
                else:
                    continue
                
                # Check rowspan
                rowspan = 1
                for check_row in range(row_idx + 1, rows):
                    if check_row >= len(cells) or col_idx >= len(cells[check_row]):
                        break
                    check_cell = cells[check_row][col_idx]
                    if check_cell is None:
                        rowspan += 1
                    elif hasattr(check_cell, 'bbox'):
                        check_x0, check_y0, _, _ = check_cell.bbox
                        if abs(check_x0 - x0) < 2:
                            rowspan += 1
                        else:
                            break
                    elif hasattr(check_cell, 'x0'):
                        if abs(check_cell.x0 - x0) < 2:
                            rowspan += 1
                        else:
                            break
                    else:
                        break
                
                # Check colspan
                colspan = 1
                for check_col in range(col_idx + 1, cols):
                    if row_idx >= len(cells) or check_col >= len(cells[row_idx]):
                        break
                    check_cell = cells[row_idx][check_col]
                    if check_cell is None:
                        colspan += 1
                    elif hasattr(check_cell, 'bbox'):
                        _, check_y0, check_x1, _ = check_cell.bbox
                        if abs(check_x1 - x1) < 2:
                            colspan += 1
                        else:
                            break
                    elif hasattr(check_cell, 'x1'):
                        if abs(check_cell.x1 - x1) < 2:
                            colspan += 1
                        else:
                            break
                    else:
                        break
                
                # If cell spans multiple rows or columns, it's merged
                if rowspan > 1 or colspan > 1:
                    merged_cells.append({
                        "row": row_idx,
                        "col": col_idx,
                        "rowspan": rowspan,
                        "colspan": colspan
                    })
                    # Mark spanned cells as processed
                    for r in range(row_idx, row_idx + rowspan):
                        for c in range(col_idx, col_idx + colspan):
                            processed.add((r, c))
    except Exception as e:
        logger.debug(f"Error detecting merged cells: {e}")
    
    return merged_cells

def _extract_table_with_merged_cells(page, table_idx: int, min_rows: int, min_cols: int, skip_empty_rows: bool) -> Optional[tuple]:
    """
    Extract table from pdfplumber page with merged cell detection.
    
    Returns:
        Tuple of (DataFrame, merged_cells_list) or None
    """
    try:
        table_objects = page.find_tables()
        if table_idx >= len(table_objects):
            return None
        
        table_obj = table_objects[table_idx]
        # Extract table data
        table_data = page.extract_tables()[table_idx] if table_idx < len(page.extract_tables()) else None
        if not table_data:
            return None
        
        df = pd.DataFrame(table_data)
        if len(df) < min_rows or len(df.columns) < min_cols:
            return None
        
        df = _clean_dataframe_for_llm(
            df, 
            skip_empty_rows=skip_empty_rows,
            preprocess_text_enabled=True,  # Default to enabled for pdfplumber
            normalize_unicode=True,
            convert_ellipsis=True
        )
        if df is None or df.empty:
            return None
        
        # Detect merged cells
        merged_cells = _detect_merged_cells(table_obj)
        
        return (df, merged_cells)
    except Exception as e:
        logger.debug(f"Error extracting table with merged cells: {e}")
        return None

def _parse_markdown_tables(markdown_text: str, pdf_path: Path) -> List[Dict[str, Any]]:
    """Parse markdown content to extract table information"""
    tables_data = []
    
    # Split by page markers
    pages = markdown_text.split('---')
    
    for page_idx, page_content in enumerate(pages, start=1):
        if not page_content.strip():
            continue
        
        # Find table blocks in markdown
        lines = page_content.split('\n')
        table_lines = []
        in_table = False
        table_idx = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect table start
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
            elif in_table and line.startswith('|'):
                table_lines.append(line)
            elif in_table and not line.startswith('|'):
                # End of table, process it
                if table_lines:
                    table_data = _process_markdown_table(table_lines, pdf_path, page_idx, table_idx)
                    if table_data:
                        tables_data.append(table_data)
                        table_idx += 1
                in_table = False
                table_lines = []
        
        # Process last table if we were in one
        if in_table and table_lines:
            table_data = _process_markdown_table(table_lines, pdf_path, page_idx, table_idx)
            if table_data:
                tables_data.append(table_data)
    
    return tables_data

def _process_markdown_table(table_lines: List[str], pdf_path: Path, page: int, index: int) -> Optional[Dict[str, Any]]:
    """Process a markdown table into structured data"""
    if len(table_lines) < 2:
        return None
    
    try:
        # Convert markdown table to DataFrame
        table_text = '\n'.join(table_lines)
        df = pd.read_csv(StringIO(table_text), sep='|', skipinitialspace=True)
        
        # Clean up the DataFrame
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        if df.empty:
            return None
        
        # Generate markdown representation
        markdown_table = _dataframe_to_markdown(df)
        
        # Extract context (text before and after table)
        context = _extract_markdown_table_context(table_lines)
        
        return {
            'dataframe': df,
            'markdown': markdown_table,
            'page': page,
            'index': index,
            'context': context,
            'metadata': {
                'extraction_method': 'pymupdf4llm',
                'rows': len(df),
                'cols': len(df.columns)
            }
        }
        
    except Exception as e:
        logger.debug(f"Failed to process markdown table: {e}")
        return None

def _extract_markdown_table_context(table_lines: List[str]) -> str:
    """Extract context around a markdown table"""
    # This is a simplified context extraction
    # In a real implementation, you'd look at surrounding text in the markdown
    return "Table extracted from PDF content"

def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table format"""
    if df.empty:
        return ""
    
    # Create header row
    header = "| " + " | ".join(str(col) for col in df.columns) + " |"
    
    # Create separator row
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    
    # Create data rows
    data_rows = []
    for _, row in df.iterrows():
        data_row = "| " + " | ".join(str(val) if pd.notna(val) else "" for val in row) + " |"
        data_rows.append(data_row)
    
    return "\n".join([header, separator] + data_rows)

def _dataframe_to_json(df: pd.DataFrame) -> str:
    """Convert DataFrame to JSON format"""
    try:
        # Convert to records format for better JSON structure
        records = df.to_dict('records')
        return json.dumps(records, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Failed to convert DataFrame to JSON: {e}")
        return ""

def _extract_table_context(table) -> str:
    """Extract context around a table from partition_pdf"""
    try:
        # Get surrounding text elements
        context_parts = []
        
        # Look for text elements before and after the table
        if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html'):
            # Extract any surrounding text from the element
            context_parts.append("Table context extracted from document structure")
        
        return " ".join(context_parts) if context_parts else ""
        
    except Exception as e:
        logger.debug(f"Failed to extract table context: {e}")
        return ""

def _extract_pdfplumber_table_context(page, table_data) -> str:
    """Extract context around a table from pdfplumber"""
    try:
        # Get text around the table area
        # This is a simplified implementation
        return "Table extracted from PDF page content"
        
    except Exception as e:
        logger.debug(f"Failed to extract pdfplumber table context: {e}")
        return ""

def _extract_table_metadata(table) -> Dict[str, Any]:
    """Extract metadata from a table element"""
    metadata = {
        'extraction_method': 'partition_pdf',
        'element_type': 'table'
    }
    
    try:
        if hasattr(table, 'metadata'):
            if hasattr(table.metadata, 'page_number'):
                metadata['page_number'] = table.metadata.page_number
            if hasattr(table.metadata, 'text_as_html'):
                metadata['has_html'] = bool(table.metadata.text_as_html)
        
        return metadata
        
    except Exception as e:
        logger.debug(f"Failed to extract table metadata: {e}")
        return metadata

def _extract_structured_table_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract structured data from a DataFrame for LLM consumption"""
    try:
        structured_data = {
            'dimensions': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'column_info': {
                'names': df.columns.tolist(),
                'types': df.dtypes.astype(str).to_dict()
            },
            'data_summary': {
                'non_null_counts': df.count().to_dict(),
                'null_counts': df.isnull().sum().to_dict()
            }
        }
        
        # Add sample data (first few rows)
        if not df.empty:
            structured_data['sample_data'] = df.head(3).to_dict('records')
        
        return structured_data
        
    except Exception as e:
        logger.debug(f"Failed to extract structured table data: {e}")
        return {}

def _clean_dataframe_for_llm(
    df: pd.DataFrame, 
    skip_empty_rows: bool = True,
    preprocess_text_enabled: bool = True,
    normalize_unicode: bool = True,
    convert_ellipsis: bool = True
) -> Optional[pd.DataFrame]:
    """
    Clean and normalize a DataFrame for LLM consumption with enhanced processing.
    Applies text preprocessing to all string cell values.
    
    Args:
        df: DataFrame to clean
        skip_empty_rows: Whether to drop rows that are entirely empty
        preprocess_text_enabled: Whether to apply text preprocessing to cell values
        normalize_unicode: Whether to normalize Unicode characters
        convert_ellipsis: Whether to convert ellipsis to dots
        
    Returns:
        Cleaned DataFrame or None if empty
    """
    if df is None or df.empty:
        return None
    
    # Handle multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() if isinstance(col, tuple) else str(col).strip() 
                      for col in df.columns]
    
    # Replace NaN and empty values with None for cleaner output
    df = df.replace({pd.NA: None, "": None, "nan": None, "NaN": None, 
                     "N/A": None, "#N/A": None, "NULL": None})
    
    # Clean column names for better LLM consumption
    df.columns = [_clean_column_name_for_llm(col, i) for i, col in enumerate(df.columns)]
    
    # Apply text preprocessing to column names
    if preprocess_text_enabled and TEXT_PREPROCESSING_AVAILABLE:
        df.columns = [preprocess_text(
            str(col), 
            normalize_unicode=normalize_unicode, 
            convert_ellipsis=convert_ellipsis
        ) for col in df.columns]
    
    # Drop duplicate columns if they exist
    if len(df.columns) != len(set(df.columns)):
        df.columns = pd.Series(df.columns).astype(str)
        df.columns = [f"{x}_{i}" if df.columns.tolist().count(x) > 1 and i > 0 
                     else x for i, x in enumerate(df.columns)]
    
    # Apply text preprocessing to all string cell values
    if preprocess_text_enabled and TEXT_PREPROCESSING_AVAILABLE:
        for col in df.columns:
            # Process only string/object columns
            if df[col].dtype == 'object':
                df[col] = df[col].apply(
                    lambda x: preprocess_text(
                        str(x), 
                        normalize_unicode=normalize_unicode, 
                        convert_ellipsis=convert_ellipsis
                    ) if pd.notna(x) and str(x).strip() else x
                )
    
    # Drop rows that are entirely empty if configured
    if skip_empty_rows:
        df = df.dropna(how='all').reset_index(drop=True)
    
    # Enhanced post-processing for LLM consumption
    df = _post_process_table_for_llm(df)
    
    return df if not df.empty else None

def _clean_column_name_for_llm(col_name: Any, index: int) -> str:
    """Clean a column name for better LLM consumption"""
    if col_name is None:
        return f"Column_{index}"
    
    col_str = str(col_name).strip()
    
    # Remove excessive whitespace
    col_str = re.sub(r'\s+', ' ', col_str)
    
    # Remove special characters that might confuse LLMs
    col_str = col_str.replace('\r', ' ').replace('\n', ' ')
    
    # Remove common header indicators
    col_str = re.sub(r'^#+\s*', '', col_str)
    
    # If column is empty or just whitespace after cleaning
    if not col_str or col_str.isspace():
        return f"Column_{index}"
    
    return col_str

def _post_process_table_for_llm(df: pd.DataFrame) -> pd.DataFrame:
    """Apply post-processing to improve table quality for LLM consumption"""
    df = df.copy()
    
    # Try to detect and fix header rows
    first_row_is_header = False
    if len(df) > 1:
        first_row = df.iloc[0]
        rest_rows = df.iloc[1:]
        
        # Enhanced heuristic for header detection
        first_row_strings = first_row.astype(str).str.match(r'[A-Za-z ]+$').sum()
        rest_rows_numeric = rest_rows.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().sum().sum()
        
        if first_row_strings > len(df.columns) * 0.5 and rest_rows_numeric > rest_rows.size * 0.3:
            first_row_is_header = True
    
    # If we detected a header row, use it to rename columns
    if first_row_is_header:
        new_columns = [str(val) if val is not None else f"Column_{i}" 
                      for i, val in enumerate(df.iloc[0])]
        df.columns = new_columns
        df = df.iloc[1:].reset_index(drop=True)
    
    # Try to convert numeric columns to appropriate types
    for col in df.columns:
        try:
            if df[col].astype(str).str.match(r'^-?\d+(\.\d+)?$').all():
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    pass
        except Exception:
            continue
    
    return df

def export_table_provenance(results: List[LLMTableResult], output_dir: Path, pdf_path: Path):
    """Export table provenance data as CSV/Parquet + HTML"""
    import json
    from pathlib import Path
    
    try:
        # Create provenance data
        provenance_data = []
        for result in results:
            provenance_record = {
                "doc_id": result.doc_id,
                "page_no": result.page,
                "table_id": result.table_id,
                "bbox": result.bbox,
                "caption": result.caption,
                "header_rows": result.header_rows,
                "source": result.source,
                "engine": result.engine,
                "engine_version": result.engine_version,
                "confidence": result.confidence,
                "source_sha256": result.source_sha256
            }
            provenance_data.append(provenance_record)
        
        if not provenance_data:
            logger.info("No table provenance data to export")
            return
        
        # Create provenance DataFrame
        provenance_df = pd.DataFrame(provenance_data)
        
        # Export as CSV
        csv_path = output_dir / f"{pdf_path.stem}_table_provenance.csv"
        provenance_df.to_csv(csv_path, index=False)
        logger.debug(f"Exported table provenance CSV to {csv_path}")
        
        # Export as Parquet
        parquet_path = output_dir / f"{pdf_path.stem}_table_provenance.parquet"
        provenance_df.to_parquet(parquet_path, index=False)
        logger.debug(f"Exported table provenance Parquet to {parquet_path}")
        
        # Export as HTML
        html_path = output_dir / f"{pdf_path.stem}_table_provenance.html"
        _export_provenance_html(provenance_df, html_path, pdf_path)
        logger.debug(f"Exported table provenance HTML to {html_path}")
        
        logger.info(f"Exported table provenance for {len(results)} tables")
        
    except Exception as e:
        logger.error(f"Failed to export table provenance: {e}")

def _export_provenance_html(provenance_df: pd.DataFrame, html_path: Path, pdf_path: Path):
    """Export provenance data as HTML table"""
    try:
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Table Provenance - {pdf_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .bbox {{ font-family: monospace; }}
        .source {{ text-transform: uppercase; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Table Provenance Report</h1>
    <p><strong>Document:</strong> {pdf_path.name}</p>
    <p><strong>Total Tables:</strong> {len(provenance_df)}</p>
    <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <table>
        <thead>
            <tr>
                <th>Doc ID</th>
                <th>Page</th>
                <th>Table ID</th>
                <th>BBox</th>
                <th>Caption</th>
                <th>Header Rows</th>
                <th>Source</th>
                <th>Engine</th>
                <th>Version</th>
                <th>Confidence</th>
                <th>SHA256</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for _, row in provenance_df.iterrows():
            bbox_str = str(row['bbox']) if pd.notna(row['bbox']) else 'N/A'
            caption_str = str(row['caption']) if pd.notna(row['caption']) else 'N/A'
            confidence_str = f"{row['confidence']:.3f}" if pd.notna(row['confidence']) else 'N/A'
            sha256_str = str(row['source_sha256'])[:16] + '...' if pd.notna(row['source_sha256']) else 'N/A'
            
            html_content += f"""
            <tr>
                <td>{row['doc_id']}</td>
                <td>{row['page_no']}</td>
                <td>{row['table_id']}</td>
                <td class="bbox">{bbox_str}</td>
                <td>{caption_str}</td>
                <td>{row['header_rows']}</td>
                <td class="source">{row['source']}</td>
                <td>{row['engine']}</td>
                <td>{row['engine_version']}</td>
                <td>{confidence_str}</td>
                <td>{sha256_str}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    except Exception as e:
        logger.error(f"Failed to export provenance HTML: {e}")
