import os
import re
import hashlib
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass
from utils.logging import get_logger
from utils.gpu_utils import get_gpu_manager, is_gpu_available
import math
import json
import logging

# Hybrid Unstructured + PyMuPDF imports
from unstructured.partition.pdf import partition_pdf

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning("pymupdf4llm not available. Install with: pip install pymupdf4llm")

logger = get_logger(__name__)

# ============================================================================
# HYBRID TUNABLES — expose for easy tuning
# ============================================================================
HEADING_SIZE_FACTOR = 1.25  # Factor for detecting headings (max_size >= median_size * factor)
TITLE_TOP_PERC = 0.12  # Top percentage of page for title detection
TOP_MARGIN_RATIO = 0.08  # Top margin ratio for header detection
BOTTOM_MARGIN_RATIO = 0.08  # Bottom margin ratio for footer detection
REPEATED_THRESHOLD = 0.4  # Threshold for repeated header/footer detection (40% of pages)
LINE_GAP_THRESHOLD = 6  # Minimum gap in points to consider separate lines
MIN_SPANS_FOR_CONFIDENCE = 1  # Minimum spans needed for confidence calculation
BBOX_PADDING_TOLERANCE = 2  # Padding tolerance for bbox containment checks (pixels)

# ============================================================================
# REGION TYPE DEFINITIONS AND VALIDATION RULES
# ============================================================================
# All possible region_type values used in the extraction pipeline:
#
# 1. 'main_text' - Main body text/paragraphs (no length limit)
#    - Description: Regular paragraph text, body content
#    - Max length: 50,000 chars
#    - Max lines: 1,000
#
# 2. 'title' - Document or section titles
#    - Description: Short, concise titles (usually 1-3 lines)
#    - Max length: 500 chars
#    - Max lines: 5
#
# 3. 'heading' - Section headings
#    - Description: Short headings for sections (usually 1-2 lines)
#    - Max length: 300 chars
#    - Max lines: 3
#
# 4. 'header' - Page headers (repeated across pages)
#    - Description: Short header text, typically 1 line
#    - Max length: 200 chars
#    - Max lines: 2
#
# 5. 'footer' - Page footers (repeated across pages)
#    - Description: Short footer text, typically 1 line
#    - Max length: 200 chars
#    - Max lines: 2
#
# 6. 'caption' - Figure/table captions
#    - Description: Short descriptions for figures/tables (usually 1-2 lines)
#    - Max length: 300 chars
#    - Max lines: 3
#
# 7. 'list_item' - List items (bulleted or numbered)
#    - Description: Items in lists, can be longer but structured
#    - Max length: 1,000 chars
#    - Max lines: 10
#
# 8. 'image_caption' - Image captions
#    - Description: Short descriptions for images (usually 1-2 lines)
#    - Max length: 300 chars
#    - Max lines: 3
#
# 9. 'table' - Table content
#    - Description: Structured table data
#    - Max length: 5,000 chars
#    - Max lines: 100
#
# 10. 'ocr_text' - OCR-extracted text
#     - Description: Text extracted via OCR (can be long)
#     - Max length: 10,000 chars
#     - Max lines: 200
#
# Note: If text exceeds these limits, it will be reclassified as 'main_text'
# ============================================================================

@dataclass
class LLMTextResult:
    """Represents LLM-ready extracted text from a PDF"""
    pdf_path: Path
    page: int
    text: str
    text_type: str  # 'main_text', 'header', 'footer', 'caption', 'table', 'image_caption', etc.
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    markdown_content: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    # New fields for paragraph_text.jsonl format
    doc_id: Optional[str] = None
    region_type: Optional[str] = None
    bbox: Optional[List[float]] = None
    ocr_conf: Optional[float] = None
    source_sha256: Optional[str] = None
    region_id: Optional[str] = None
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    
    def __str__(self) -> str:
        return f"LLM Text (Page {self.page}, {self.text_type}): {len(self.text)} chars"

# ============================================================================
# Hybrid Unstructured + PyMuPDF Helper Functions
# ============================================================================

def run_unstructured_partition(pdf_path: Path, strategy: str = "hi_res") -> List[Any]:
    """
    Run unstructured.partition_pdf to extract layout elements.
    
    Args:
        pdf_path: Path to PDF file
        strategy: Extraction strategy ('hi_res', 'fast', 'ocr_only')
        
    Returns:
        List of unstructured elements (do not mutate)
    """
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=strategy,
            infer_table_structure=False,
            include_page_breaks=True
        )
        logger.debug(f"Unstructured partition extracted {len(elements)} elements from {pdf_path.name}")
        return elements
    except Exception as e:
        logger.warning(f"Unstructured partition failed for {pdf_path.name}: {e}")
        return []

def extract_pymupdf_spans(pdf_path: Path) -> Dict[int, List[Dict]]:
    """
    Extract PyMuPDF spans with font and formatting information per page.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary mapping page_number (1-indexed) to list of span dicts
    """
    spans_by_page = {}
    
    try:
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            text_dict = page.get_text("dict")
            page_spans = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            
                            # Extract bbox from span
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            
                            # Extract font information
                            font = span.get("font", "unknown")
                            size = span.get("size", None)
                            flags = span.get("flags", 0)
                            bold = bool(flags & 16)  # Bit 4 indicates bold
                            italic = bool(flags & 1)  # Bit 0 indicates italic
                            
                            span_dict = {
                                "text": text,
                                "bbox": list(bbox),
                                "font": font,
                                "size": size,
                                "bold": bold,
                                "italic": italic,
                                "page_number": page_num,
                                "origin": "pdf_text"
                            }
                            page_spans.append(span_dict)
            
            spans_by_page[page_num] = page_spans
            logger.debug(f"Extracted {len(page_spans)} spans from page {page_num}")
        
        doc.close()
    except Exception as e:
        logger.warning(f"Failed to extract PyMuPDF spans from {pdf_path.name}: {e}")
    
    return spans_by_page

def _bbox_contains(outer_bbox: Dict[str, float], inner_bbox: List[float], padding: float = BBOX_PADDING_TOLERANCE) -> bool:
    """Check if inner_bbox is contained within outer_bbox with padding tolerance."""
    return (inner_bbox[0] >= outer_bbox["x0"] - padding and
            inner_bbox[1] >= outer_bbox["y0"] - padding and
            inner_bbox[2] <= outer_bbox["x1"] + padding and
            inner_bbox[3] <= outer_bbox["y1"] + padding)

def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for header/footer matching (strip digits, punctuation, lowercase)."""
    # Remove page numbers and dates
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase and strip
    return text.lower().strip()

def reconcile_elements_with_spans(
    elements: List[Any],
    spans_by_page: Dict[int, List[Dict]],
    pdf_path: Path,
    thresholds: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Reconcile Unstructured elements with PyMuPDF spans to improve text classification.
    
    Args:
        elements: List of unstructured elements
        spans_by_page: Dictionary of spans per page
        pdf_path: Path to PDF file
        thresholds: Optional threshold overrides
        
    Returns:
        List of standardized block dictionaries
    """
    if thresholds is None:
        thresholds = {
            "heading_size_factor": HEADING_SIZE_FACTOR,
            "title_top_perc": TITLE_TOP_PERC,
            "top_margin_ratio": TOP_MARGIN_RATIO,
            "bottom_margin_ratio": BOTTOM_MARGIN_RATIO
        }
    
    blocks = []
    
    # Get page dimensions
    try:
        doc = fitz.open(pdf_path)
        page_dims = {}
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            page_dims[page_num] = (rect.width, rect.height)
        doc.close()
    except:
        page_dims = {}
    
    for el in elements:
        try:
            # Extract element properties
            if hasattr(el, 'to_dict'):
                el_dict = el.to_dict()
            elif hasattr(el, '__dict__'):
                el_dict = el.__dict__
            else:
                el_dict = {}
            
            # Get text and metadata
            text = getattr(el, 'text', '') or el_dict.get('text', '')
            if not text or not text.strip():
                continue
            
            # Get page number
            page_number = getattr(el.metadata, 'page_number', None) if hasattr(el, 'metadata') else None
            if page_number is None:
                page_number = el_dict.get('metadata', {}).get('page_number', 1)
            
            # Get bbox
            bbox_dict = {}
            if hasattr(el, 'metadata') and hasattr(el.metadata, 'coordinates'):
                coords = el.metadata.coordinates
                if hasattr(coords, 'x1') and hasattr(coords, 'y1'):
                    bbox_dict = {
                        "x0": getattr(coords, 'x1', 0),
                        "y0": getattr(coords, 'y1', 0),
                        "x1": getattr(coords, 'x2', getattr(coords, 'x1', 0)),
                        "y1": getattr(coords, 'y2', getattr(coords, 'y1', 0))
                    }
            elif 'coordinates' in el_dict.get('metadata', {}):
                coords = el_dict['metadata']['coordinates']
                if isinstance(coords, dict):
                    bbox_dict = {
                        "x0": coords.get('x1', coords.get('x0', 0)),
                        "y0": coords.get('y1', coords.get('y0', 0)),
                        "x1": coords.get('x2', coords.get('x1', coords.get('x0', 0))),
                        "y1": coords.get('y2', coords.get('y1', coords.get('y0', 0)))
                    }
                elif hasattr(coords, 'x1'):
                    bbox_dict = {
                        "x0": getattr(coords, 'x1', 0),
                        "y0": getattr(coords, 'y1', 0),
                        "x1": getattr(coords, 'x2', getattr(coords, 'x1', 0)),
                        "y1": getattr(coords, 'y2', getattr(coords, 'y1', 0))
                    }
            
            if not bbox_dict:
                # Fallback: use element bbox if available
                bbox_dict = el_dict.get('bbox', {})
            
            # Get original category from unstructured
            original_category = getattr(el, 'category', None) or el_dict.get('category', 'paragraph')
            
            # Map unstructured categories to our standard types
            # Unstructured uses: NarrativeText, Title, Image, Table, etc.
            unstructured_category_mapping = {
                'NarrativeText': 'paragraph',
                'narrativetext': 'paragraph',
                'Image': 'paragraph',  # Image elements - treat as regular text unless it's a caption
                'image': 'paragraph',
                'Table': 'paragraph',  # Tables are handled separately, text elements are paragraphs
                'table': 'paragraph',
                'Title': 'paragraph',  # Will be re-evaluated by title detection logic below
                'title': 'paragraph',
                'Heading': 'paragraph',  # Will be re-evaluated by heading detection logic below
                'heading': 'paragraph',
                'ListItem': 'paragraph',  # Will be re-evaluated by list detection logic below
                'listitem': 'paragraph',
                'Header': 'paragraph',  # Will be re-evaluated by header detection logic below
                'header': 'paragraph',
                'Footer': 'paragraph',  # Will be re-evaluated by footer detection logic below
                'footer': 'paragraph',
                'Caption': 'paragraph',  # Will be re-evaluated by caption detection logic below
                'caption': 'paragraph'
            }
            
            # Normalize original_category using mapping
            if original_category in unstructured_category_mapping:
                original_category = unstructured_category_mapping[original_category]
            
            # Collect spans within this element's bbox
            page_spans = spans_by_page.get(page_number, [])
            contained_spans = []
            
            if bbox_dict and page_spans:
                for span in page_spans:
                    span_bbox = span.get("bbox", [])
                    if len(span_bbox) == 4 and _bbox_contains(bbox_dict, span_bbox):
                        contained_spans.append(span)
            
            # Compute font statistics
            font_sizes = [s.get("size") for s in contained_spans if s.get("size") is not None]
            bold_count = sum(1 for s in contained_spans if s.get("bold", False))
            span_count = len(contained_spans)
            
            median_font_size = None
            max_font_size = None
            bold_fraction = 0.0
            
            if font_sizes:
                font_sizes.sort()
                median_font_size = font_sizes[len(font_sizes) // 2]
                max_font_size = max(font_sizes)
            
            if span_count > 0:
                bold_fraction = bold_count / span_count
            
            # Determine block type using heuristics
            block_type = original_category  # Default to unstructured label
            
            # Get page dimensions for position-based heuristics
            page_width, page_height = page_dims.get(page_number, (612, 792))  # Default letter size
            
            # Heading detection
            if median_font_size and max_font_size:
                if max_font_size >= median_font_size * thresholds["heading_size_factor"]:
                    if original_category in ['paragraph', 'text']:
                        block_type = 'heading'
            
            # Title detection (first page, large font, top of page)
            # Additional checks to prevent entire page text from being classified as title:
            # 1. Text should be relatively short (titles are concise, not entire pages)
            # 2. Bbox should not cover most of the page (titles are compact)
            # 3. Should not contain multiple paragraphs or excessive line breaks
            if page_number == 1 and max_font_size and bbox_dict:
                y0_ratio = bbox_dict.get("y0", 0) / page_height if page_height > 0 else 1.0
                y1_ratio = bbox_dict.get("y1", 0) / page_height if page_height > 0 else 1.0
                bbox_height_ratio = (y1_ratio - y0_ratio) if page_height > 0 else 1.0
                
                # Check if font size suggests title
                font_check = max_font_size > (median_font_size or 12) * 1.5
                
                # Check if position is at top of page
                position_check = y0_ratio < thresholds["title_top_perc"]
                
                # Additional strict checks for title:
                # - Text length should be reasonable (titles are usually < 500 chars, not entire pages)
                text_length = len(text.strip())
                length_check = text_length < 500  # Titles are concise
                
                # - Bbox should not cover too much of the page height (titles are compact)
                #   Titles typically take up < 15% of page height
                bbox_height_check = bbox_height_ratio < 0.15
                
                # - Should not have excessive line breaks (titles are usually 1-3 lines)
                line_count = text.count('\n') + 1
                line_check = line_count <= 5  # Allow up to 5 lines for multi-line titles
                
                # - Should not contain multiple periods (titles are usually single sentences)
                period_count = text.count('.')
                period_check = period_count <= 3  # Allow a few periods for abbreviations
                
                # All checks must pass for title classification
                if (font_check and position_check and length_check and 
                    bbox_height_check and line_check and period_check and
                    original_category in ['paragraph', 'text', 'heading']):
                    block_type = 'title'
            
            # Header/footer candidate detection
            if bbox_dict and page_height > 0:
                y0_ratio = bbox_dict.get("y0", 0) / page_height
                y1_ratio = bbox_dict.get("y1", 0) / page_height
                
                if y1_ratio < thresholds["top_margin_ratio"]:
                    if original_category in ['paragraph', 'text']:
                        block_type = 'header_candidate'
                elif y0_ratio > (1 - thresholds["bottom_margin_ratio"]):
                    if original_category in ['paragraph', 'text']:
                        block_type = 'footer_candidate'
            
            # Caption detection (short text, often near images/tables)
            if len(text.strip()) < 200 and original_category in ['paragraph', 'text']:
                # Check if text matches caption patterns
                if re.match(r'^(Fig(ure)?|Table|Figure)\s*\.?\s*[\d\.]+', text, re.IGNORECASE):
                    block_type = 'caption'
            
            # List item detection (starts with bullet/number)
            if re.match(r'^[\u2022\u2023\u25E6\u2043\u2219\-\*•]\s+', text) or re.match(r'^\d+[\.\)]\s+', text):
                if original_category in ['paragraph', 'text']:
                    block_type = 'list_item'
            
            # Determine provenance
            if span_count == 0:
                provenance = 'ocr'
            elif original_category == block_type:
                provenance = 'unstructured'
            else:
                provenance = 'hybrid'
            
            # Create standardized block
            block = {
                "type": block_type,
                "text": text.strip(),
                "page_number": page_number,
                "bbox": bbox_dict,
                "font_median": median_font_size,
                "font_max": max_font_size,
                "bold_fraction": bold_fraction,
                "spans": contained_spans[:10],  # Limit spans for storage
                "provenance": provenance,
                "original_category": original_category
            }
            blocks.append(block)
            
        except Exception as e:
            logger.debug(f"Failed to reconcile element: {e}")
            continue
    
    # Post-process blocks to remove duplicates and fix misclassifications
    # Remove blocks that are supersets of other blocks (e.g., entire page text containing actual title)
    filtered_blocks = []
    for i, block1 in enumerate(blocks):
        is_duplicate_or_superset = False
        
        # Check if this block is a superset of another block
        for j, block2 in enumerate(blocks):
            if i == j:
                continue
            
            # Same page
            if block1["page_number"] != block2["page_number"]:
                continue
            
            text1 = block1["text"].strip().lower()
            text2 = block2["text"].strip().lower()
            
            # If block1 contains block2's text and is much longer, it's likely a superset
            if text2 in text1 and len(text1) > len(text2) * 1.5:
                # Check bbox overlap - if block1's bbox contains block2's bbox, it's a superset
                bbox1 = block1.get("bbox", {})
                bbox2 = block2.get("bbox", {})
                
                if bbox1 and bbox2:
                    # Check if bbox1 contains bbox2
                    if (bbox1.get("x0", 0) <= bbox2.get("x0", 0) and
                        bbox1.get("y0", 0) <= bbox2.get("y0", 0) and
                        bbox1.get("x1", 0) >= bbox2.get("x1", 0) and
                        bbox1.get("y1", 0) >= bbox2.get("y1", 0)):
                        # block1 contains block2 - prefer the smaller, more specific block
                        # If block1 is classified as title but is too large, it's likely wrong
                        if block1["type"] == "title" and len(text1) > 500:
                            is_duplicate_or_superset = True
                            logger.debug(f"Removing superset block on page {block1['page_number']}: "
                                       f"'{text1[:50]}...' (contains '{text2[:50]}...')")
                            break
                        # If block2 is a title and block1 is not, prefer block2
                        elif block2["type"] == "title" and block1["type"] != "title":
                            is_duplicate_or_superset = True
                            logger.debug(f"Removing superset block on page {block1['page_number']}: "
                                       f"'{text1[:50]}...' (superset of title '{text2[:50]}...')")
                            break
            
            # Check for near-duplicate text (very similar content)
            if text1 and text2 and len(text1) > 50 and len(text2) > 50:
                # If texts are very similar (>90% overlap) and one is much longer
                shorter = min(len(text1), len(text2))
                longer = max(len(text1), len(text2))
                if shorter / longer > 0.9 and abs(len(text1) - len(text2)) > 100:
                    # Prefer the shorter one (more specific)
                    if len(text1) > len(text2):
                        is_duplicate_or_superset = True
                        logger.debug(f"Removing near-duplicate block on page {block1['page_number']}: "
                                   f"longer version of similar content")
                        break
        
        if not is_duplicate_or_superset:
            filtered_blocks.append(block1)
    
    logger.debug(f"Filtered {len(blocks)} blocks to {len(filtered_blocks)} after removing duplicates/supersets")
    return filtered_blocks

def detect_and_finalize_headers_footers(blocks: List[Dict], repeated_threshold: float = REPEATED_THRESHOLD) -> List[Dict]:
    """
    Detect repeated headers/footers across pages and finalize their labels.
    
    Args:
        blocks: List of block dictionaries
        repeated_threshold: Threshold for considering text as repeated (0.0-1.0)
        
    Returns:
        Updated list of blocks with finalized header/footer labels
    """
    # Group blocks by page
    blocks_by_page = {}
    for block in blocks:
        page = block.get("page_number", 1)
        if page not in blocks_by_page:
            blocks_by_page[page] = []
        blocks_by_page[page].append(block)
    
    # Collect header and footer candidates
    header_candidates = []
    footer_candidates = []
    
    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "header_candidate":
            header_candidates.append(block)
        elif block_type == "footer_candidate":
            footer_candidates.append(block)
    
    # Normalize and count occurrences
    header_texts = {}
    footer_texts = {}
    
    for block in header_candidates:
        normalized = _normalize_text_for_matching(block.get("text", ""))
        if normalized and len(normalized) > 3:  # Ignore very short texts
            header_texts[normalized] = header_texts.get(normalized, 0) + 1
    
    for block in footer_candidates:
        normalized = _normalize_text_for_matching(block.get("text", ""))
        if normalized and len(normalized) > 3:
            footer_texts[normalized] = footer_texts.get(normalized, 0) + 1
    
    total_pages = len(blocks_by_page)
    if total_pages == 0:
        return blocks
    
    # Finalize headers
    for block in header_candidates:
        normalized = _normalize_text_for_matching(block.get("text", ""))
        count = header_texts.get(normalized, 0)
        if count >= total_pages * repeated_threshold:
            block["type"] = "header"
            logger.debug(f"Promoted header_candidate to header on page {block.get('page_number')}: {normalized[:50]}")
        else:
            block["type"] = "paragraph"  # Downgrade to paragraph
    
    # Finalize footers
    for block in footer_candidates:
        normalized = _normalize_text_for_matching(block.get("text", ""))
        count = footer_texts.get(normalized, 0)
        if count >= total_pages * repeated_threshold:
            block["type"] = "footer"
            logger.debug(f"Promoted footer_candidate to footer on page {block.get('page_number')}: {normalized[:50]}")
        else:
            block["type"] = "paragraph"  # Downgrade to paragraph
    
    return blocks

def build_final_json(pdf_path: Path, blocks: List[Dict], page_dims: Dict[int, Tuple[float, float]]) -> Dict:
    """
    Build final JSON structure from reconciled blocks.
    
    Args:
        pdf_path: Path to PDF file
        blocks: List of reconciled block dictionaries
        page_dims: Dictionary mapping page_number to (width, height)
        
    Returns:
        Dictionary with structure: {"path": str, "pages": [{"page_number": int, "width": float, "height": float, "blocks": [...]}]}
    """
    # Group blocks by page
    blocks_by_page = {}
    for block in blocks:
        page = block.get("page_number", 1)
        if page not in blocks_by_page:
            blocks_by_page[page] = []
        blocks_by_page[page].append(block)
    
    # Build pages list
    pages = []
    for page_num in sorted(blocks_by_page.keys()):
        page_blocks = blocks_by_page[page_num]
        width, height = page_dims.get(page_num, (612, 792))
        
        page_dict = {
            "page_number": page_num,
            "width": width,
            "height": height,
            "blocks": page_blocks
        }
        pages.append(page_dict)
    
    result = {
        "path": str(pdf_path),
        "pages": pages
    }
    
    return result

def hybrid_text_extract(pdf_path: Path, strategy: str = "hi_res", thresholds: Optional[Dict[str, float]] = None) -> Dict:
    """
    Orchestrate hybrid Unstructured + PyMuPDF text extraction.
    
    Args:
        pdf_path: Path to PDF file
        strategy: Extraction strategy for unstructured
        thresholds: Optional threshold overrides
        
    Returns:
        Dictionary with final JSON structure
    """
    logger.info(f"Running hybrid text extraction for {pdf_path.name}")
    
    # Step 1: Run unstructured partition
    elements = run_unstructured_partition(pdf_path, strategy)
    logger.debug(f"Unstructured extracted {len(elements)} elements")
    
    # Step 2: Extract PyMuPDF spans
    spans_by_page = extract_pymupdf_spans(pdf_path)
    total_spans = sum(len(spans) for spans in spans_by_page.values())
    logger.debug(f"PyMuPDF extracted {total_spans} spans across {len(spans_by_page)} pages")
    
    # Step 3: Reconcile elements with spans
    blocks = reconcile_elements_with_spans(elements, spans_by_page, pdf_path, thresholds)
    
    # Step 4: Detect and finalize headers/footers
    blocks = detect_and_finalize_headers_footers(blocks, REPEATED_THRESHOLD)
    
    # Get page dimensions
    try:
        doc = fitz.open(pdf_path)
        page_dims = {}
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            page_dims[page_num] = (rect.width, rect.height)
        doc.close()
    except:
        page_dims = {}
    
    # Step 5: Build final JSON
    json_out = build_final_json(pdf_path, blocks, page_dims)
    
    logger.info(f"Hybrid extraction completed: {len(blocks)} blocks across {len(json_out.get('pages', []))} pages")
    
    return json_out

def _populate_paragraph_fields(result: LLMTextResult, pdf_path: Path, region_id: str = None) -> LLMTextResult:
    """Populate the new paragraph_text.jsonl fields for a text result"""
    import hashlib
    import fitz
    
    # Generate doc_id from PDF filename
    doc_id = pdf_path.stem
    
    # Generate source_sha256
    try:
        with open(pdf_path, 'rb') as f:
            source_sha256 = hashlib.sha256(f.read()).hexdigest()
    except:
        source_sha256 = None
    
    # Generate region_id if not provided
    if region_id is None:
        region_id = f"{doc_id}_p{result.page}_{result.text_type}_{hash(result.text) % 10000}"
    
    # Set region_type based on text_type, but validate it first
    region_type = result.text_type
    
    # CRITICAL: Validate that text length and characteristics match the region_type
    # Prevent misclassification of text blocks based on their expected characteristics
    text_length = len(result.text.strip()) if result.text else 0
    text_lines = result.text.count('\n') + 1 if result.text else 0
    
    # Define validation rules for each region type
    # Format: (max_length, max_lines, description)
    region_type_validation_rules = {
        'title': (500, 5, 'Titles are short, concise, usually 1-3 lines'),
        'heading': (300, 3, 'Headings are short, usually 1-2 lines'),
        'header': (200, 2, 'Headers are short, typically 1 line'),
        'footer': (200, 2, 'Footers are short, typically 1 line'),
        'caption': (300, 3, 'Captions are short descriptions, usually 1-2 lines'),
        'list_item': (1000, 10, 'List items can be longer but typically structured'),
        'image_caption': (300, 3, 'Image captions are short descriptions'),
        'table': (5000, 100, 'Tables can contain structured data'),
        'ocr_text': (10000, 200, 'OCR text can be long'),
        'main_text': (50000, 1000, 'Main text can be any length')  # No limit for main_text
    }
    
    # Normalize region_type for comparison
    region_type_lower = region_type.lower() if region_type else ''
    
    # Validate region type - prevent misclassification
    if region_type_lower in region_type_validation_rules:
        max_length, max_lines, description = region_type_validation_rules[region_type_lower]
        
        # Check if text violates the expected characteristics
        if text_length > max_length or text_lines > max_lines:
            # Text doesn't match expected characteristics - reclassify as main_text
            logger.debug(f"Reclassifying text block (length: {text_length}, lines: {text_lines}) "
                        f"from '{region_type}' to 'main_text' on page {result.page}. "
                        f"Reason: {description} (max length: {max_length}, max lines: {max_lines})")
            
            original_region_type = region_type
            region_type = "main_text"
            result.text_type = "main_text"
            
            # Update metadata to track the original misclassification
            if result.metadata is None:
                result.metadata = {}
            result.metadata['original_region_type'] = original_region_type
            result.metadata['reclassification_reason'] = f'text_too_long_for_{region_type_lower}'
            result.metadata['original_text_length'] = text_length
            result.metadata['original_text_lines'] = text_lines
            result.metadata['max_expected_length'] = max_length
            result.metadata['max_expected_lines'] = max_lines
    
    # Post-processing: Detect valid titles that were missed
    # If text is currently "main_text" but looks like a title, promote it
    if region_type_lower == 'main_text' and result.page == 1:
        # Check if this could be a valid title
        title_max_length, title_max_lines, _ = region_type_validation_rules.get('title', (500, 5, ''))
        
        # Title detection criteria:
        # 1. Short text (within title limits)
        # 2. At top of page (check bbox if available)
        # 3. Not too many periods (titles are usually single sentences)
        # 4. Not too many line breaks
        is_short = text_length <= title_max_length and text_lines <= title_max_lines
        period_count = result.text.count('.') if result.text else 0
        has_few_periods = period_count <= 3
        has_few_lines = text_lines <= 5
        
        # Check bbox position if available
        is_at_top = False
        if result.bbox and len(result.bbox) >= 4:
            # Assume standard page height of 792 points (11 inches at 72 DPI)
            # Titles should be in top 15% of page
            page_height = 792  # Default
            y0 = result.bbox[1] if len(result.bbox) > 1 else 0
            is_at_top = y0 < (page_height * 0.15)
        elif result.bbox is None:
            # If no bbox, we can't determine position, so skip position check
            is_at_top = True  # Don't exclude based on position if we don't have bbox
        
        # Additional check: title-like text patterns
        # Titles often start with capital letters, are concise, don't have many commas
        text_starts_capital = result.text and result.text.strip() and result.text.strip()[0].isupper()
        comma_count = result.text.count(',') if result.text else 0
        has_few_commas = comma_count <= 5  # Titles can have some commas but not many
        
        # Promote to title if it meets criteria
        if (is_short and has_few_periods and has_few_lines and 
            is_at_top and text_starts_capital and has_few_commas and
            text_length > 10):  # Must have some content
            logger.debug(f"Promoting text block to 'title' on page {result.page}: "
                        f"length={text_length}, lines={text_lines}, at_top={is_at_top}")
            region_type = "title"
            result.text_type = "title"
            if result.metadata is None:
                result.metadata = {}
            result.metadata['promoted_to_title'] = True
            result.metadata['title_detection_reason'] = 'post_processing_title_detection'
    
    # Map unstructured categories to our standard types
    # Unstructured uses: NarrativeText, Title, Image, Table, etc.
    category_mapping = {
        'NarrativeText': 'main_text',
        'narrativetext': 'main_text',
        'Image': 'image_caption',  # Image elements are typically captions or image-related text
        'image': 'image_caption',
        'Table': 'table',
        'table': 'table',
        'Title': 'title',  # Keep Title if it passes validation above
        'title': 'title',
        'Heading': 'heading',
        'heading': 'heading',
        'ListItem': 'list_item',
        'listitem': 'list_item',
        'Header': 'header',
        'header': 'header',
        'Footer': 'footer',
        'footer': 'footer',
        'Caption': 'caption',
        'caption': 'caption'
    }
    
    # Apply category mapping if region_type matches unstructured categories
    if region_type in category_mapping:
        mapped_type = category_mapping[region_type]
        # Only remap if it's not already a valid type
        if mapped_type != region_type:
            # Re-validate after mapping using the same validation rules
            mapped_type_lower = mapped_type.lower()
            if mapped_type_lower in region_type_validation_rules:
                max_length, max_lines, description = region_type_validation_rules[mapped_type_lower]
                if text_length > max_length or text_lines > max_lines:
                    mapped_type = 'main_text'
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata['original_region_type'] = region_type
                    result.metadata['reclassification_reason'] = f'text_too_long_for_{mapped_type_lower}_after_mapping'
                    result.metadata['original_text_length'] = text_length
                    result.metadata['original_text_lines'] = text_lines
                    result.metadata['max_expected_length'] = max_length
                    result.metadata['max_expected_lines'] = max_lines
            
            region_type = mapped_type
            result.text_type = mapped_type
    
    # Set engine and version
    engine = "pymupdf4llm" if result.metadata and result.metadata.get('extraction_method') == 'pymupdf4llm' else "fitz"
    # Check if it came from hybrid classification
    if result.metadata and result.metadata.get('extraction_method') == 'hybrid':
        engine = "hybrid"
    engine_version = "1.0.0"  # Default version
    
    # Set OCR confidence
    ocr_conf = result.confidence if result.confidence is not None else None
    
    # Extract bbox from structured_data if available
    bbox = None
    if result.structured_data and 'rect' in result.structured_data:
        bbox = result.structured_data['rect']
    elif result.metadata and 'position' in result.metadata:
        bbox = result.metadata['position']
    
    # Update the result with new fields
    result.doc_id = doc_id
    result.region_type = region_type
    result.bbox = bbox
    result.source_sha256 = source_sha256
    result.region_id = region_id
    result.engine = engine
    result.engine_version = engine_version
    result.ocr_conf = ocr_conf
    
    return result

def preprocess_text(text: str, normalize_unicode: bool = True, convert_ellipsis: bool = True) -> str:
    """
    Preprocess extracted text to clean up special characters and formatting issues
    while preserving the meaning of Unicode and special characters.
    
    Args:
        text: Raw extracted text
        normalize_unicode: Whether to normalize Unicode characters
        convert_ellipsis: Whether to convert ellipsis to dots
        
    Returns:
        Cleaned text with special characters handled appropriately
    """
    if not text:
        return text
    
    # First, handle Unicode escape sequences (e.g., \u00b0 -> °, \u2026 -> …, \u2212 -> −, \u03c1 -> ρ)
    if normalize_unicode:
        try:
            # Handle \uXXXX patterns (4-digit hex) - only if exactly 4 digits
            text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
            # Handle \xXX patterns (2-digit hex) - only if exactly 2 digits
            text = re.sub(r'\\x([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), text)
            # Handle \UXXXXXXXX patterns (8-digit hex for extended Unicode) - only if exactly 8 digits
            text = re.sub(r'\\U([0-9a-fA-F]{8})', lambda m: chr(int(m.group(1), 16)), text)
            # Handle \n, \t, \r patterns (but preserve their meaning)
            text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            # Handle other common escape sequences
            text = text.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
            
            # Handle malformed Unicode escape sequences (like \u2, \u20, etc.)
            # Remove incomplete \u sequences that aren't exactly 4 digits
            text = re.sub(r'\\u[0-9a-fA-F]{1,3}(?![0-9a-fA-F])', '', text)
            # Remove incomplete \x sequences that aren't exactly 2 digits
            text = re.sub(r'\\x[0-9a-fA-F]{1}(?![0-9a-fA-F])', '', text)
            
        except (ValueError, OverflowError):
            # If Unicode decoding fails, keep original text
            pass
    
    # Handle specific problematic characters that commonly appear in PDFs
    if convert_ellipsis:
        # Replace multiple ellipsis characters with single ellipsis
        text = re.sub(r'…+', '…', text)
        
        # Replace multiple dots with ellipsis (common PDF artifact)
        text = re.sub(r'\.{3,}', '…', text)
        
        # Handle various ellipsis-like characters and normalize them
        text = text.replace('…', '...')  # Replace ellipsis with three dots
        text = text.replace('⋯', '...')  # Midline horizontal ellipsis
        text = text.replace('⋮', '...')  # Vertical ellipsis
        text = text.replace('⋰', '...')  # Up right diagonal ellipsis
        text = text.replace('⋱', '...')  # Down right diagonal ellipsis
    
    # Replace multiple dashes with em dash
    text = re.sub(r'-{2,}', '—', text)
    
    # Replace multiple underscores with single underscore
    text = re.sub(r'_{2,}', '_', text)
    
    # Handle specific Unicode characters that appear as artifacts
    if normalize_unicode:
        # Mathematical symbols and operators
        text = text.replace('\u2212', '-')  # Minus sign (U+2212) → hyphen-minus
        text = text.replace('\u2215', '/')  # Division slash → forward slash
        text = text.replace('\u00d7', 'x')  # Multiplication sign → x
        text = text.replace('\u00f7', '/')  # Division sign → forward slash
        text = text.replace('\u2260', '!=')  # Not equal to → !=
        text = text.replace('\u2264', '<=')  # Less-than or equal to → <=
        text = text.replace('\u2265', '>=')  # Greater-than or equal to → >=
        text = text.replace('\u2248', '≈')  # Almost equal to (keep as is)
        text = text.replace('\u00b1', '±')  # Plus-minus sign (keep as is)
        text = text.replace('\u221a', '√')  # Square root (keep as is)
        text = text.replace('\u221e', '∞')  # Infinity (keep as is)
        text = text.replace('\u03c0', 'π')  # Greek letter pi (keep as is)
        text = text.replace('\u03b1', 'α')  # Greek letter alpha (keep as is)
        text = text.replace('\u03b2', 'β')  # Greek letter beta (keep as is)
        text = text.replace('\u03b3', 'γ')  # Greek letter gamma (keep as is)
        text = text.replace('\u03b4', 'δ')  # Greek letter delta (keep as is)
        text = text.replace('\u03b5', 'ε')  # Greek letter epsilon (keep as is)
        text = text.replace('\u03b6', 'ζ')  # Greek letter zeta (keep as is)
        text = text.replace('\u03b7', 'η')  # Greek letter eta (keep as is)
        text = text.replace('\u03b8', 'θ')  # Greek letter theta (keep as is)
        text = text.replace('\u03b9', 'ι')  # Greek letter iota (keep as is)
        text = text.replace('\u03ba', 'κ')  # Greek letter kappa (keep as is)
        text = text.replace('\u03bb', 'λ')  # Greek letter lambda (keep as is)
        text = text.replace('\u03bc', 'μ')  # Greek letter mu (keep as is)
        text = text.replace('\u03bd', 'ν')  # Greek letter nu (keep as is)
        text = text.replace('\u03be', 'ξ')  # Greek letter xi (keep as is)
        text = text.replace('\u03bf', 'ο')  # Greek letter omicron (keep as is)
        text = text.replace('\u03c1', 'ρ')  # Greek letter rho (keep as is)
        text = text.replace('\u03c3', 'σ')  # Greek letter sigma (keep as is)
        text = text.replace('\u03c4', 'τ')  # Greek letter tau (keep as is)
        text = text.replace('\u03c5', 'υ')  # Greek letter upsilon (keep as is)
        text = text.replace('\u03c6', 'φ')  # Greek letter phi (keep as is)
        text = text.replace('\u03c7', 'χ')  # Greek letter chi (keep as is)
        text = text.replace('\u03c8', 'ψ')  # Greek letter psi (keep as is)
        text = text.replace('\u03c9', 'ω')  # Greek letter omega (keep as is)
        # Greek uppercase letters
        text = text.replace('\u0391', 'Α')  # Greek capital alpha
        text = text.replace('\u0392', 'Β')  # Greek capital beta
        text = text.replace('\u0393', 'Γ')  # Greek capital gamma
        text = text.replace('\u0394', 'Δ')  # Greek capital delta
        text = text.replace('\u0395', 'Ε')  # Greek capital epsilon
        text = text.replace('\u0396', 'Ζ')  # Greek capital zeta
        text = text.replace('\u0397', 'Η')  # Greek capital eta
        text = text.replace('\u0398', 'Θ')  # Greek capital theta
        text = text.replace('\u0399', 'Ι')  # Greek capital iota
        text = text.replace('\u039a', 'Κ')  # Greek capital kappa
        text = text.replace('\u039b', 'Λ')  # Greek capital lambda
        text = text.replace('\u039c', 'Μ')  # Greek capital mu
        text = text.replace('\u039d', 'Ν')  # Greek capital nu
        text = text.replace('\u039e', 'Ξ')  # Greek capital xi
        text = text.replace('\u039f', 'Ο')  # Greek capital omicron
        text = text.replace('\u03a0', 'Π')  # Greek capital pi
        text = text.replace('\u03a1', 'Ρ')  # Greek capital rho
        text = text.replace('\u03a3', 'Σ')  # Greek capital sigma
        text = text.replace('\u03a4', 'Τ')  # Greek capital tau
        text = text.replace('\u03a5', 'Υ')  # Greek capital upsilon
        text = text.replace('\u03a6', 'Φ')  # Greek capital phi
        text = text.replace('\u03a7', 'Χ')  # Greek capital chi
        text = text.replace('\u03a8', 'Ψ')  # Greek capital psi
        text = text.replace('\u03a9', 'Ω')  # Greek capital omega
        
        # Replace various dash-like characters with standard dash
        text = text.replace('–', '-')  # En dash (U+2013)
        text = text.replace('—', '-')  # Em dash (U+2014)
        text = text.replace('−', '-')  # Minus sign (U+2212) - already handled above but keep for clarity
        text = text.replace('‐', '-')  # Hyphen (U+2010)
        text = text.replace('‑', '-')  # Non-breaking hyphen (U+2011)
        text = text.replace('‒', '-')  # Figure dash (U+2012)
        text = text.replace('―', '-')  # Horizontal bar (U+2015)
        text = text.replace('⁃', '-')  # Hyphen bullet (U+2043)
        text = text.replace('⁻', '-')  # Superscript minus (U+207B)
        text = text.replace('₋', '-')  # Subscript minus (U+208B)
        
        # Replace various quote characters with standard quotes
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes (U+201C, U+201D)
        text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes (U+2018, U+2019)
        text = text.replace('«', '"').replace('»', '"')  # Guillemets (U+00AB, U+00BB)
        text = text.replace('„', '"').replace('‚', "'")  # Double/single low-9 quotation marks
        text = text.replace('‹', "'").replace('›', "'")  # Single guillemets
        
        # Replace various space characters with standard space
        text = text.replace('\u00a0', ' ')  # Non-breaking space (U+00A0)
        text = text.replace('\u2000', ' ')  # En quad (U+2000)
        text = text.replace('\u2001', ' ')  # Em quad (U+2001)
        text = text.replace('\u2002', ' ')  # En space (U+2002)
        text = text.replace('\u2003', ' ')  # Em space (U+2003)
        text = text.replace('\u2004', ' ')  # Three-per-em space (U+2004)
        text = text.replace('\u2005', ' ')  # Four-per-em space (U+2005)
        text = text.replace('\u2006', ' ')  # Six-per-em space (U+2006)
        text = text.replace('\u2007', ' ')  # Figure space (U+2007)
        text = text.replace('\u2008', ' ')  # Punctuation space (U+2008)
        text = text.replace('\u2009', ' ')  # Thin space (U+2009)
        text = text.replace('\u200a', ' ')  # Hair space (U+200A)
        text = text.replace('\u200b', '')   # Zero-width space (U+200B) - remove
        text = text.replace('\u200c', '')   # Zero-width non-joiner (U+200C) - remove
        text = text.replace('\u200d', '')   # Zero-width joiner (U+200D) - remove
        text = text.replace('\u202f', ' ')  # Narrow no-break space (U+202F)
        text = text.replace('\u205f', ' ')  # Medium mathematical space (U+205F)
        text = text.replace('\u3000', ' ')  # Ideographic space (U+3000)
        
        # Currency and special symbols
        text = text.replace('\u20ac', '€')  # Euro sign (keep as is)
        text = text.replace('\u00a3', '£')  # Pound sign (keep as is)
        text = text.replace('\u00a5', '¥')  # Yen sign (keep as is)
        text = text.replace('\u00a2', '¢')  # Cent sign (keep as is)
        text = text.replace('\u00a4', '¤')  # Currency sign (keep as is)
        
        # Fractions (convert common ones to readable format)
        text = text.replace('\u00bc', '1/4')  # ¼
        text = text.replace('\u00bd', '1/2')  # ½
        text = text.replace('\u00be', '3/4')  # ¾
        text = text.replace('\u2153', '1/3')  # ⅓
        text = text.replace('\u2154', '2/3')  # ⅔
        text = text.replace('\u2155', '1/5')  # ⅕
        text = text.replace('\u2156', '2/5')  # ⅖
        text = text.replace('\u2157', '3/5')  # ⅗
        text = text.replace('\u2158', '4/5')  # ⅘
        text = text.replace('\u2159', '1/6')  # ⅙
        text = text.replace('\u215a', '5/6')  # ⅚
        text = text.replace('\u215b', '1/8')  # ⅛
        text = text.replace('\u215c', '3/8')  # ⅜
        text = text.replace('\u215d', '5/8')  # ⅝
        text = text.replace('\u215e', '7/8')  # ⅞
        
        # Superscripts and subscripts (convert to regular numbers)
        text = text.replace('\u00b9', '1')  # ¹
        text = text.replace('\u00b2', '2')  # ²
        text = text.replace('\u00b3', '3')  # ³
        text = text.replace('\u2070', '0')  # ⁰
        text = text.replace('\u2074', '4')  # ⁴
        text = text.replace('\u2075', '5')  # ⁵
        text = text.replace('\u2076', '6')  # ⁶
        text = text.replace('\u2077', '7')  # ⁷
        text = text.replace('\u2078', '8')  # ⁸
        text = text.replace('\u2079', '9')  # ⁹
        text = text.replace('\u2080', '0')  # ₀
        text = text.replace('\u2081', '1')  # ₁
        text = text.replace('\u2082', '2')  # ₂
        text = text.replace('\u2083', '3')  # ₃
        text = text.replace('\u2084', '4')  # ₄
        text = text.replace('\u2085', '5')  # ₅
        text = text.replace('\u2086', '6')  # ₆
        text = text.replace('\u2087', '7')  # ₇
        text = text.replace('\u2088', '8')  # ₈
        text = text.replace('\u2089', '9')  # ₉
        
        # Degree and other symbols
        text = text.replace('\u00b0', '°')  # Degree sign (keep as is)
        text = text.replace('\u2032', "'")  # Prime (U+2032) → apostrophe
        text = text.replace('\u2033', '"')  # Double prime (U+2033) → quote
        text = text.replace('\u2035', "'")  # Reversed prime → apostrophe
        text = text.replace('\u2036', '"')  # Reversed double prime → quote
        
        # Bullets and list markers
        text = text.replace('\u2022', '•')  # Bullet (keep as is)
        text = text.replace('\u2023', '‣')  # Triangular bullet (keep as is)
        text = text.replace('\u25e6', '◦')  # White bullet (keep as is)
        text = text.replace('\u2043', '-')  # Hyphen bullet → dash
        text = text.replace('\u2219', '•')  # Bullet operator → bullet
        
        # Arrows (keep as is, but normalize common ones)
        text = text.replace('\u2190', '←')  # Leftwards arrow (keep)
        text = text.replace('\u2192', '→')  # Rightwards arrow (keep)
        text = text.replace('\u2191', '↑')  # Upwards arrow (keep)
        text = text.replace('\u2193', '↓')  # Downwards arrow (keep)
        text = text.replace('\u2194', '↔')  # Left right arrow (keep)
        text = text.replace('\u21d0', '⇐')  # Leftwards double arrow (keep)
        text = text.replace('\u21d2', '⇒')  # Rightwards double arrow (keep)
        text = text.replace('\u21d4', '⇔')  # Left right double arrow (keep)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    
    # Remove empty lines
    cleaned_lines = [line for line in cleaned_lines if line]
    
    # Join lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove any remaining excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Clean up common PDF extraction artifacts
    cleaned_text = re.sub(r'\f', '', cleaned_text)  # Remove form feed characters
    cleaned_text = re.sub(r'\r', '', cleaned_text)  # Remove carriage returns
    cleaned_text = re.sub(r'\v', '', cleaned_text)  # Remove vertical tabs
    
    # Remove any remaining control characters except newlines
    cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_text)
    
    return cleaned_text.strip()

def extract_llm_ready_text(
    pdf_path: str | Path,
    output_dir: str | Path,
    use_pymupdf4llm: bool = True,
    extract_headers: bool = True,
    extract_footers: bool = True,
    extract_captions: bool = True,
    extract_tables_as_markdown: bool = True,
    extract_images_with_captions: bool = True,
    preserve_formatting: bool = True,
    use_ocr: bool = False,
    min_confidence: float = 0.5,
    write_images: bool = True,
    embed_images: bool = False,
    image_dir: Optional[str] = None,
    preprocess_text_enabled: bool = True,
    normalize_unicode: bool = True,
    convert_ellipsis: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMTextResult]:
    """
    Extract LLM-ready text from a PDF file using pymupdf4llm and fallback strategies.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text
        use_pymupdf4llm: Whether to use pymupdf4llm for primary extraction
        extract_headers: Whether to extract headers separately
        extract_footers: Whether to extract footers separately
        extract_captions: Whether to extract captions separately
        extract_tables_as_markdown: Whether to convert tables to markdown
        extract_images_with_captions: Whether to extract image captions
        preserve_formatting: Whether to preserve text formatting
        use_ocr: Whether to use OCR for scanned documents
        min_confidence: Minimum confidence for OCR results
        write_images: Whether to save images to disk
        embed_images: Whether to embed images as base64 in markdown
        image_dir: Directory to save images (if write_images=True)
        
    Returns:
        List of LLMTextResult objects
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting LLM-ready text from {pdf_path.name}")
    
    # Create output directory for this PDF's text
    pdf_text_dir = Path(output_dir) / pdf_path.stem / "llm_text"
    pdf_text_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    try:
        # Strategy 1: pymupdf4llm - Best for LLM-ready content
        if use_pymupdf4llm and PYMUPDF4LLM_AVAILABLE:
            pymupdf4llm_results = _extract_with_pymupdf4llm(
                pdf_path,
                pdf_text_dir,
                write_images=write_images,
                embed_images=embed_images,
                image_dir=image_dir,
                extract_tables_as_markdown=extract_tables_as_markdown,
                start_page=start_page,
                end_page=end_page
            )
            results.extend(pymupdf4llm_results)
        
        # Strategy 2: Enhanced PyMuPDF extraction with better structure
        if not results or len([r for r in results if r.text.strip()]) < 10:
            logger.info(f"Trying enhanced PyMuPDF extraction for {pdf_path.name}")
            fitz_results = _extract_with_enhanced_fitz(
                pdf_path,
                extract_headers=extract_headers,
                extract_footers=extract_footers,
                extract_captions=extract_captions,
                preserve_formatting=preserve_formatting,
                extract_tables_as_markdown=extract_tables_as_markdown,
                preprocess_text_enabled=preprocess_text_enabled,
                normalize_unicode=normalize_unicode,
                convert_ellipsis=convert_ellipsis,
                start_page=start_page,
                end_page=end_page
            )
            results.extend(fitz_results)
        
        # Strategy 3: pdfplumber fallback for complex layouts
        if not results or len([r for r in results if r.text.strip()]) < 10:
            logger.info(f"Trying pdfplumber fallback for {pdf_path.name}")
            pdfplumber_results = _extract_with_enhanced_pdfplumber(
                pdf_path,
                extract_headers=extract_headers,
                extract_footers=extract_footers,
                extract_captions=extract_captions,
                extract_tables_as_markdown=extract_tables_as_markdown,
                preprocess_text_enabled=preprocess_text_enabled,
                normalize_unicode=normalize_unicode,
                convert_ellipsis=convert_ellipsis,
                start_page=start_page,
                end_page=end_page
            )
            results.extend(pdfplumber_results)
        
        # Strategy 4: OCR fallback for scanned documents
        if use_ocr and (not results or len([r for r in results if r.text.strip()]) < 50):
            logger.info(f"Trying OCR extraction for {pdf_path.name}")
            ocr_results = _extract_with_enhanced_ocr(
                pdf_path,
                min_confidence=min_confidence,
                preprocess_text_enabled=preprocess_text_enabled,
                normalize_unicode=normalize_unicode,
                convert_ellipsis=convert_ellipsis,
                start_page=start_page,
                end_page=end_page
            )
            results.extend(ocr_results)
        
        # Filter results by page range if specified
        if start_page is not None or end_page is not None:
            original_count = len(results)
            filtered_results = []
            for result in results:
                if start_page is not None and result.page < start_page:
                    continue
                if end_page is not None and result.page > end_page:
                    continue
                filtered_results.append(result)
            results = filtered_results
            if original_count != len(results):
                logger.info(f"Filtered text content: {original_count} -> {len(results)} (pages {start_page or 1}-{end_page or 'all'})")
        
        # ========================================================================
        # Hybrid Unstructured + PyMuPDF post-processing for improved classification
        # ========================================================================
        try:
            if results:
                logger.info(f"Running hybrid text classification for {pdf_path.name}")
                hybrid_json = hybrid_text_extract(pdf_path, strategy="hi_res")
                
                # Map hybrid blocks to existing results
                hybrid_blocks_by_page = {}
                for page_data in hybrid_json.get("pages", []):
                    page_num = page_data.get("page_number", 1)
                    hybrid_blocks_by_page[page_num] = page_data.get("blocks", [])
                
                # Helper function to check if bboxes overlap
                def bboxes_overlap(bbox1: Optional[List[float]], bbox2: Dict[str, float], tolerance: float = 10.0) -> bool:
                    """Check if two bboxes overlap with tolerance."""
                    if not bbox1 or len(bbox1) < 4 or not bbox2:
                        return False
                    return not (bbox1[2] < bbox2.get("x0", 0) - tolerance or
                               bbox1[0] > bbox2.get("x1", 0) + tolerance or
                               bbox1[3] < bbox2.get("y0", 0) - tolerance or
                               bbox1[1] > bbox2.get("y1", 0) + tolerance)
                
                # Helper function for fuzzy text matching
                def texts_similar(text1: str, text2: str, threshold: float = 0.7) -> bool:
                    """Check if two texts are similar (simple character overlap)."""
                    if not text1 or not text2:
                        return False
                    text1_clean = text1.strip().lower()
                    text2_clean = text2.strip().lower()
                    if len(text1_clean) < 10 or len(text2_clean) < 10:
                        return text1_clean == text2_clean
                    # Simple overlap check
                    shorter = min(len(text1_clean), len(text2_clean))
                    longer = max(len(text1_clean), len(text2_clean))
                    if shorter == 0:
                        return False
                    # Count common characters
                    common = sum(1 for c in text1_clean if c in text2_clean)
                    return (common / longer) >= threshold
                
                # Update existing results with hybrid labels
                for result in results:
                    page_num = result.page
                    hybrid_blocks = hybrid_blocks_by_page.get(page_num, [])
                    
                    # Find matching hybrid block
                    matched_block = None
                    for block in hybrid_blocks:
                        block_bbox = block.get("bbox", {})
                        block_text = block.get("text", "")
                        
                        # Try bbox overlap first
                        if bboxes_overlap(result.bbox, block_bbox):
                            matched_block = block
                            break
                        # Fallback to text similarity
                        elif texts_similar(result.text, block_text):
                            matched_block = block
                            break
                    
                    if matched_block:
                        hybrid_type = matched_block.get("type", "")
                        original_type = result.text_type
                        
                        # Map hybrid types to result text_types
                        type_mapping = {
                            "title": "title",
                            "heading": "heading",
                            "paragraph": "main_text",
                            "header": "header",
                            "footer": "footer",
                            "caption": "caption",
                            "list_item": "list_item"
                        }
                        
                        mapped_type = type_mapping.get(hybrid_type, hybrid_type)
                        
                        # Additional validation: Don't update to title if the text is too long
                        # This prevents entire page text from being classified as title
                        if mapped_type == "title":
                            text_length = len(result.text.strip())
                            if text_length > 500:
                                # Text is too long to be a title, keep original type
                                logger.debug(f"Skipping title classification for long text on page {page_num}: "
                                           f"{text_length} chars (max 500 for titles)")
                                continue
                        
                        # Update if type differs
                        if mapped_type != original_type and hybrid_type not in ["header_candidate", "footer_candidate"]:
                            if result.metadata is None:
                                result.metadata = {}
                            result.metadata['original_label'] = original_type
                            result.metadata['hybrid_label'] = mapped_type
                            result.metadata['hybrid_label_source'] = 'unstructured+pymupdf'
                            result.metadata['provenance'] = matched_block.get("provenance", "hybrid")
                            
                            # Update text_type
                            result.text_type = mapped_type
                            
                            logger.debug(f"Relabeled page {page_num}: {original_type} -> {mapped_type} (provenance: {matched_block.get('provenance', 'hybrid')})")
                            
                            # Optionally update text if hybrid text is cleaner
                            hybrid_text = matched_block.get("text", "").strip()
                            if hybrid_text and len(hybrid_text) > len(result.text.strip()) * 0.9:
                                # Hybrid text is similar length or longer, consider replacing
                                if texts_similar(result.text, hybrid_text, threshold=0.8):
                                    # Apply preprocessing to hybrid text
                                    if preprocess_text_enabled:
                                        hybrid_text = preprocess_text(
                                            hybrid_text,
                                            normalize_unicode=normalize_unicode,
                                            convert_ellipsis=convert_ellipsis
                                        )
                                    result.text = hybrid_text
                                    result.metadata['hybrid_text_replaced'] = True
                
                # Add missing blocks from hybrid extraction
                existing_pages = set(r.page for r in results)
                for page_num, hybrid_blocks in hybrid_blocks_by_page.items():
                    if page_num not in existing_pages or len([r for r in results if r.page == page_num and r.text.strip()]) < 3:
                        # Page has few or no results, add hybrid blocks
                        for block in hybrid_blocks:
                            block_type = block.get("type", "main_text")
                            block_text = block.get("text", "").strip()
                            
                            if not block_text or block_type in ["header_candidate", "footer_candidate"]:
                                continue
                            
                            # Apply preprocessing to hybrid block text
                            if preprocess_text_enabled:
                                block_text = preprocess_text(
                                    block_text,
                                    normalize_unicode=normalize_unicode,
                                    convert_ellipsis=convert_ellipsis
                                )
                            
                            # Map type
                            type_mapping = {
                                "title": "title",
                                "heading": "heading",
                                "paragraph": "main_text",
                                "header": "header",
                                "footer": "footer",
                                "caption": "caption",
                                "list_item": "list_item"
                            }
                            mapped_type = type_mapping.get(block_type, "main_text")
                            
                            # Extract bbox
                            bbox = None
                            block_bbox = block.get("bbox", {})
                            if block_bbox:
                                bbox = [block_bbox.get("x0", 0), block_bbox.get("y0", 0),
                                       block_bbox.get("x1", 0), block_bbox.get("y1", 0)]
                            
                            new_result = LLMTextResult(
                                pdf_path=pdf_path,
                                page=page_num,
                                text=block_text,
                                text_type=mapped_type,
                                metadata={
                                    "extraction_method": "hybrid",
                                    "hybrid_label_source": "unstructured+pymupdf",
                                    "provenance": block.get("provenance", "hybrid"),
                                    "font_median": block.get("font_median"),
                                    "font_max": block.get("font_max"),
                                    "bold_fraction": block.get("bold_fraction", 0.0)
                                },
                                bbox=bbox
                            )
                            results.append(new_result)
                            logger.debug(f"Added hybrid block from page {page_num}: {mapped_type}")
                
                # Write debug JSON if debug logging is enabled
                if logger.isEnabledFor(logging.DEBUG):
                    debug_file = pdf_text_dir / f"{pdf_path.stem}_hybrid_debug.json"
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            json.dump(hybrid_json, f, indent=2, ensure_ascii=False)
                        logger.debug(f"Wrote hybrid debug JSON to {debug_file}")
                    except Exception as e:
                        logger.debug(f"Failed to write debug JSON: {e}")
                
                logger.info(f"Hybrid classification completed: updated {len([r for r in results if r.metadata and r.metadata.get('hybrid_label')])} results")
            
            elif not results:
                # No results from existing strategies, use hybrid extraction directly
                logger.info(f"No results from existing strategies, using hybrid extraction for {pdf_path.name}")
                hybrid_json = hybrid_text_extract(pdf_path, strategy="hi_res")
                
                # Convert hybrid blocks to LLMTextResult objects
                for page_data in hybrid_json.get("pages", []):
                    page_num = page_data.get("page_number", 1)
                    # Filter by page range if specified
                    if start_page is not None and page_num < start_page:
                        continue
                    if end_page is not None and page_num > end_page:
                        continue
                    for block in page_data.get("blocks", []):
                        block_type = block.get("type", "main_text")
                        block_text = block.get("text", "").strip()
                        
                        if not block_text or block_type in ["header_candidate", "footer_candidate"]:
                            continue
                        
                        # Apply preprocessing to hybrid block text
                        if preprocess_text_enabled:
                            block_text = preprocess_text(
                                block_text,
                                normalize_unicode=normalize_unicode,
                                convert_ellipsis=convert_ellipsis
                            )
                        
                        # Map type
                        type_mapping = {
                            "title": "title",
                            "heading": "heading",
                            "paragraph": "main_text",
                            "header": "header",
                            "footer": "footer",
                            "caption": "caption",
                            "list_item": "list_item"
                        }
                        mapped_type = type_mapping.get(block_type, "main_text")
                        
                        # CRITICAL: Validate title - prevent long text from being labeled as title
                        text_length = len(block_text.strip())
                        if mapped_type == "title" and text_length > 500:
                            # Text is too long to be a title - reclassify as main_text
                            logger.debug(f"Reclassifying hybrid block from 'title' to 'main_text' on page {page_num}: "
                                       f"text length {text_length} chars (max 500 for titles)")
                            mapped_type = "main_text"
                        
                        # Extract bbox
                        bbox = None
                        block_bbox = block.get("bbox", {})
                        if block_bbox:
                            bbox = [block_bbox.get("x0", 0), block_bbox.get("y0", 0),
                                   block_bbox.get("x1", 0), block_bbox.get("y1", 0)]
                        
                        result = LLMTextResult(
                            pdf_path=pdf_path,
                            page=page_num,
                            text=block_text,
                            text_type=mapped_type,
                            metadata={
                                "extraction_method": "hybrid",
                                "hybrid_label_source": "unstructured+pymupdf",
                                "provenance": block.get("provenance", "hybrid"),
                                "font_median": block.get("font_median"),
                                "font_max": block.get("font_max"),
                                "bold_fraction": block.get("bold_fraction", 0.0),
                                "original_block_type": block_type if mapped_type != block_type else None
                            },
                            bbox=bbox
                        )
                        results.append(result)
                
                logger.info(f"Created {len(results)} results from hybrid extraction")
        
        except Exception as e:
            logger.warning(f"Hybrid text classification failed for {pdf_path.name}: {e}", exc_info=True)
        
        # Apply text preprocessing to all results if enabled (final pass to ensure all text is preprocessed)
        if preprocess_text_enabled:
            for i, result in enumerate(results):
                if result.text:
                    results[i].text = preprocess_text(
                        result.text,
                        normalize_unicode=normalize_unicode,
                        convert_ellipsis=convert_ellipsis
                    )
            logger.debug(f"Applied text preprocessing to {len(results)} text blocks")
        
        # Populate new paragraph fields for all results
        for i, result in enumerate(results):
            results[i] = _populate_paragraph_fields(result, pdf_path)
        
        # Save extracted text to files
        _save_llm_text_results(results, pdf_text_dir, pdf_path)
        
    except Exception as e:
        logger.error(f"Failed to extract LLM-ready text from {pdf_path.name}: {e}")
        return []
    
    # Filter out empty results
    results = [r for r in results if r.text.strip()]
    
    logger.info(f"Successfully extracted LLM-ready text from {len(results)} blocks in {pdf_path.name}")
    return results

def _extract_with_pymupdf4llm(
    pdf_path: Path,
    output_dir: Path,
    write_images: bool = True,
    embed_images: bool = False,
    image_dir: Optional[str] = None,
    extract_tables_as_markdown: bool = True,
    preprocess_text_enabled: bool = True,
    normalize_unicode: bool = True,
    convert_ellipsis: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMTextResult]:
    """Extract text using pymupdf4llm for LLM-ready content"""
    results = []
    
    try:
        # Configure image extraction
        image_output_dir = None
        if write_images and image_dir:
            image_output_dir = Path(image_dir) / pdf_path.stem / "images"
            image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract markdown content using pymupdf4llm
        markdown_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            write_images=write_images,
            image_path=image_output_dir,
            embed_images=embed_images
        )
        
        # Parse the markdown to extract structured content
        structured_content = _parse_markdown_content(markdown_text, pdf_path)
        
        # Convert to LLMTextResult objects
        for page_num, content in structured_content.items():
            # Filter by page range if specified
            if start_page is not None and page_num < start_page:
                continue
            if end_page is not None and page_num > end_page:
                continue
            for content_type, text_data in content.items():
                text_content = text_data.get('text', '').strip()
                if text_content:
                    # Apply preprocessing immediately
                    if preprocess_text_enabled:
                        text_content = preprocess_text(
                            text_content,
                            normalize_unicode=normalize_unicode,
                            convert_ellipsis=convert_ellipsis
                        )
                    
                    result = LLMTextResult(
                        pdf_path=pdf_path,
                        page=page_num,
                        text=text_content,
                        text_type=content_type,
                        markdown_content=text_data.get('markdown', ''),
                        metadata=text_data.get('metadata', {}),
                        structured_data=text_data.get('structured_data', {})
                    )
                    results.append(result)
        
        # Save the full markdown content
        markdown_file = output_dir / f"{pdf_path.stem}_full_markdown.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        
        logger.info(f"pymupdf4llm extracted {len(results)} content blocks from {pdf_path.name}")
        
    except Exception as e:
        logger.warning(f"pymupdf4llm extraction failed for {pdf_path.name}: {e}")
    
    return results

def _extract_with_enhanced_fitz(
    pdf_path: Path,
    extract_headers: bool = True,
    extract_footers: bool = True,
    extract_captions: bool = True,
    preserve_formatting: bool = True,
    extract_tables_as_markdown: bool = True,
    preprocess_text_enabled: bool = True,
    normalize_unicode: bool = True,
    convert_ellipsis: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMTextResult]:
    """Enhanced PyMuPDF extraction with better structure for LLM consumption"""
    results = []
    
    try:
        doc = fitz.open(pdf_path)
        
        total_pages = len(doc)
        sp = max(1, start_page) if start_page else 1
        ep = min(total_pages, end_page) if end_page else total_pages
        
        for page_idx in range(sp - 1, ep):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            # Extract main text with enhanced structure
            if preserve_formatting:
                text_dict = page.get_text("dict")
                main_text, structured_data = _extract_enhanced_text_from_dict(text_dict)
            else:
                main_text = page.get_text()
                structured_data = {}
            
            if main_text.strip():
                # Apply preprocessing immediately
                if preprocess_text_enabled:
                    main_text = preprocess_text(
                        main_text,
                        normalize_unicode=normalize_unicode,
                        convert_ellipsis=convert_ellipsis
                    )
                
                result = LLMTextResult(
                    pdf_path=pdf_path,
                    page=page_num,
                    text=main_text,
                    text_type="main_text",
                    metadata={"extraction_method": "enhanced_fitz"},
                    structured_data=structured_data
                )
                results.append(result)
            
            # Extract headers and footers with better detection
            if extract_headers or extract_footers:
                header_footer_text = _extract_enhanced_headers_footers(page, extract_headers, extract_footers)
                for text_type, text_data in header_footer_text.items():
                    text_content = text_data.get('text', '').strip()
                    if text_content:
                        # Apply preprocessing immediately
                        if preprocess_text_enabled:
                            text_content = preprocess_text(
                                text_content,
                                normalize_unicode=normalize_unicode,
                                convert_ellipsis=convert_ellipsis
                            )
                        
                        result = LLMTextResult(
                            pdf_path=pdf_path,
                            page=page_num,
                            text=text_content,
                            text_type=text_type,
                            metadata=text_data.get('metadata', {}),
                            structured_data=text_data.get('structured_data', {})
                        )
                        results.append(result)
            
            # Extract captions with better context
            if extract_captions:
                captions = _extract_enhanced_captions(page)
                for caption_data in captions:
                    caption_text = caption_data.get('text', '').strip()
                    if caption_text:
                        # Apply preprocessing immediately
                        if preprocess_text_enabled:
                            caption_text = preprocess_text(
                                caption_text,
                                normalize_unicode=normalize_unicode,
                                convert_ellipsis=convert_ellipsis
                            )
                        
                        result = LLMTextResult(
                            pdf_path=pdf_path,
                            page=page_num,
                            text=caption_text,
                            text_type="caption",
                            metadata=caption_data.get('metadata', {}),
                            structured_data=caption_data.get('structured_data', {})
                        )
                        results.append(result)
            
            # Extract tables as markdown if requested
            if extract_tables_as_markdown:
                tables = _extract_tables_as_markdown(page, page_num)
                for table_data in tables:
                    table_text = table_data.get('text', '').strip()
                    if table_text:
                        # Apply preprocessing immediately
                        if preprocess_text_enabled:
                            table_text = preprocess_text(
                                table_text,
                                normalize_unicode=normalize_unicode,
                                convert_ellipsis=convert_ellipsis
                            )
                        
                        result = LLMTextResult(
                            pdf_path=pdf_path,
                            page=page_num,
                            text=table_text,
                            text_type="table",
                            markdown_content=table_data.get('markdown', ''),
                            metadata=table_data.get('metadata', {}),
                            structured_data=table_data.get('structured_data', {})
                        )
                        results.append(result)
        
        doc.close()
        
    except Exception as e:
        logger.warning(f"Enhanced PyMuPDF extraction failed for {pdf_path.name}: {e}")
    
    return results

def _extract_with_enhanced_pdfplumber(
    pdf_path: Path,
    extract_headers: bool = True,
    extract_footers: bool = True,
    extract_captions: bool = True,
    extract_tables_as_markdown: bool = True,
    preprocess_text_enabled: bool = True,
    normalize_unicode: bool = True,
    convert_ellipsis: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMTextResult]:
    """Enhanced pdfplumber extraction with better structure"""
    results = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            sp = max(1, start_page) if start_page else 1
            ep = min(len(pdf.pages), end_page) if end_page else len(pdf.pages)
            for page_idx in range(sp, ep + 1):
                page = pdf.pages[page_idx - 1]
                # Extract main text with better structure
                text = page.extract_text()
                if text and text.strip():
                    # Apply preprocessing immediately
                    if preprocess_text_enabled:
                        text = preprocess_text(
                            text,
                            normalize_unicode=normalize_unicode,
                            convert_ellipsis=convert_ellipsis
                        )
                    
                    result = LLMTextResult(
                        pdf_path=pdf_path,
                        page=page_idx,
                        text=text,
                        text_type="main_text",
                        metadata={"extraction_method": "enhanced_pdfplumber"}
                    )
                    results.append(result)
                
                # Extract text from tables with markdown conversion
                if extract_tables_as_markdown:
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_text, markdown_table = _table_to_enhanced_text_and_markdown(table)
                            if table_text.strip():
                                # Apply preprocessing immediately
                                if preprocess_text_enabled:
                                    table_text = preprocess_text(
                                        table_text,
                                        normalize_unicode=normalize_unicode,
                                        convert_ellipsis=convert_ellipsis
                                    )
                                
                                result = LLMTextResult(
                                    pdf_path=pdf_path,
                                    page=page_idx,
                                    text=table_text,
                                    text_type="table",
                                    markdown_content=markdown_table,
                                    metadata={"table_index": table_idx, "extraction_method": "pdfplumber"},
                                    structured_data={"table_data": table}
                                )
                                results.append(result)
    
    except Exception as e:
        logger.warning(f"Enhanced pdfplumber extraction failed for {pdf_path.name}: {e}")
    
    return results

def _extract_with_enhanced_ocr(
    pdf_path: Path,
    min_confidence: float = 0.5,
    preprocess_text_enabled: bool = True,
    normalize_unicode: bool = True,
    convert_ellipsis: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMTextResult]:
    """Enhanced OCR extraction with better structure and M1 GPU acceleration"""
    results = []
    
    # Initialize GPU acceleration for M1 Mac
    gpu_manager = get_gpu_manager()
    use_gpu = gpu_manager.gpu_available
    
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        # Convert PDF pages to images with higher DPI for better OCR
        dpi = 400 if use_gpu else 300  # Higher DPI for GPU processing
        images = convert_from_path(pdf_path, dpi=dpi)
        
        logger.info(f"Processing {len(images)} pages with {'M1 GPU' if use_gpu else 'CPU'} acceleration")
        
        # Filter pages by range if specified
        sp = max(1, start_page) if start_page else 1
        ep = min(len(images), end_page) if end_page else len(images)
        
        for page_idx in range(sp, ep + 1):
            image = images[page_idx - 1]
            # Extract text using OCR with confidence scores
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Filter by confidence
            confident_text = []
            confidences = []
            for i, conf in enumerate(data['conf']):
                if int(conf) >= min_confidence * 100:
                    text = data['text'][i].strip()
                    if text:
                        confident_text.append(text)
                        confidences.append(int(conf) / 100.0)
            
            if confident_text:
                text = ' '.join(confident_text)
                avg_confidence = sum(confidences) / len(confidences) if confidences else min_confidence
                
                # Apply preprocessing immediately
                if preprocess_text_enabled:
                    text = preprocess_text(
                        text,
                        normalize_unicode=normalize_unicode,
                        convert_ellipsis=convert_ellipsis
                    )
                
                result = LLMTextResult(
                    pdf_path=pdf_path,
                    page=page_idx,
                    text=text,
                    text_type="ocr_text",
                    confidence=avg_confidence,
                    metadata={
                        "extraction_method": "enhanced_ocr_gpu" if use_gpu else "enhanced_ocr",
                        "min_confidence": min_confidence,
                        "dpi_used": dpi,
                        "gpu_accelerated": use_gpu
                    }
                )
                results.append(result)
    
    except ImportError:
        logger.warning("OCR dependencies not available. Install pytesseract and pdf2image for OCR support.")
    except Exception as e:
        logger.warning(f"Enhanced OCR extraction failed for {pdf_path.name}: {e}")
    
    return results

def _parse_markdown_content(markdown_text: str, pdf_path: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Parse markdown content to extract structured information"""
    structured_content = {}
    
    # Split by page markers (pymupdf4llm adds page breaks)
    pages = markdown_text.split('---')
    
    for page_idx, page_content in enumerate(pages, start=1):
        if not page_content.strip():
            continue
            
        page_data = {}
        
        # Extract different content types
        lines = page_content.split('\n')
        current_section = "main_text"
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect headers
            if line.startswith('#'):
                if current_text:
                    page_data[current_section] = {
                        'text': '\n'.join(current_text),
                        'markdown': '\n'.join(current_text),
                        'metadata': {'section_type': current_section}
                    }
                current_section = "header"
                current_text = [line]
            # Detect tables
            elif '|' in line and line.count('|') >= 2:
                if current_text and current_section != "table":
                    if current_text:
                        page_data[current_section] = {
                            'text': '\n'.join(current_text),
                            'markdown': '\n'.join(current_text),
                            'metadata': {'section_type': current_section}
                        }
                current_section = "table"
                current_text = [line]
            # Detect image references
            elif line.startswith('![') or 'image' in line.lower():
                if current_text:
                    page_data[current_section] = {
                        'text': '\n'.join(current_text),
                        'markdown': '\n'.join(current_text),
                        'metadata': {'section_type': current_section}
                    }
                current_section = "image_caption"
                current_text = [line]
            else:
                current_text.append(line)
        
        # Add the last section
        if current_text:
            page_data[current_section] = {
                'text': '\n'.join(current_text),
                'markdown': '\n'.join(current_text),
                'metadata': {'section_type': current_section}
            }
        
        structured_content[page_idx] = page_data
    
    return structured_content

def _extract_enhanced_text_from_dict(text_dict: Dict) -> Tuple[str, Dict[str, Any]]:
    """Extract text from PyMuPDF text dictionary with enhanced structure"""
    text_parts = []
    structured_data = {
        'blocks': [],
        'paragraphs': [],
        'sentences': []
    }
    
    for block in text_dict.get("blocks", []):
        if "lines" in block:
            block_text = []
            for line in block["lines"]:
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                if line_text.strip():
                    block_text.append(line_text)
                    text_parts.append(line_text)
            
            if block_text:
                structured_data['blocks'].append({
                    'text': '\n'.join(block_text),
                    'line_count': len(block_text)
                })
    
    return "\n".join(text_parts), structured_data

def _extract_enhanced_headers_footers(page, extract_headers: bool, extract_footers: bool) -> Dict[str, Dict[str, Any]]:
    """Extract headers and footers with enhanced metadata"""
    results = {}
    
    try:
        # Get page dimensions
        rect = page.rect
        page_height = rect.height
        page_width = rect.width
        
        # Define header and footer regions (top and bottom 10% of page)
        header_rect = fitz.Rect(0, 0, page_width, page_height * 0.1)
        footer_rect = fitz.Rect(0, page_height * 0.9, page_width, page_height)
        
        if extract_headers:
            header_text = page.get_textbox(header_rect)
            if header_text.strip():
                results["header"] = {
                    'text': header_text.strip(),
                    'metadata': {'region': 'header', 'position': 'top'},
                    'structured_data': {'rect': [header_rect.x0, header_rect.y0, header_rect.x1, header_rect.y1]}
                }
        
        if extract_footers:
            footer_text = page.get_textbox(footer_rect)
            if footer_text.strip():
                results["footer"] = {
                    'text': footer_text.strip(),
                    'metadata': {'region': 'footer', 'position': 'bottom'},
                    'structured_data': {'rect': [footer_rect.x0, footer_rect.y0, footer_rect.x1, footer_rect.y1]}
                }
    
    except Exception as e:
        logger.debug(f"Failed to extract enhanced headers/footers: {e}")
    
    return results

def _extract_enhanced_captions(page) -> List[Dict[str, Any]]:
    """Extract captions with enhanced context"""
    captions = []
    
    try:
        # Look for common caption patterns with better context
        caption_patterns = [
            r'^(Fig(ure)?|Table|Figure)\s*\.?\s*[\d\.]+.*',
            r'^[A-Z][^.]*\.$',  # Sentences starting with capital letter
        ]
        
        blocks = page.get_text("blocks")
        for x0, y0, x1, y1, text, _, _ in blocks:
            if text and text.strip():
                clean_text = " ".join(text.split())
                for pattern in caption_patterns:
                    if re.match(pattern, clean_text, re.IGNORECASE):
                        captions.append({
                            'text': clean_text,
                            'metadata': {'pattern_matched': pattern, 'position': [x0, y0, x1, y1]},
                            'structured_data': {'rect': [x0, y0, x1, y1], 'confidence': 0.8}
                        })
                        break
    
    except Exception as e:
        logger.debug(f"Failed to extract enhanced captions: {e}")
    
    return captions

def _extract_tables_as_markdown(page, page_num: int) -> List[Dict[str, Any]]:
    """Extract tables and convert to markdown format"""
    tables = []
    
    try:
        # Use pdfplumber for table detection
        import pdfplumber
        with pdfplumber.open(page.parent) as pdf:
            pdf_page = pdf.pages[page_num - 1]
            page_tables = pdf_page.extract_tables()
            
            for table_idx, table in enumerate(page_tables):
                if table:
                    table_text, markdown_table = _table_to_enhanced_text_and_markdown(table)
                    if table_text.strip():
                        tables.append({
                            'text': table_text,
                            'markdown': markdown_table,
                            'metadata': {'table_index': table_idx, 'page': page_num},
                            'structured_data': {'table_data': table}
                        })
    
    except Exception as e:
        logger.debug(f"Failed to extract tables as markdown: {e}")
    
    return tables

def _table_to_enhanced_text_and_markdown(table: List[List[str]]) -> Tuple[str, str]:
    """Convert a table to both readable text and markdown format"""
    if not table:
        return "", ""
    
    # Create readable text
    text_lines = []
    for row in table:
        if row:
            text_lines.append("\t".join(str(cell) if cell else "" for cell in row))
    text_content = "\n".join(text_lines)
    
    # Create markdown table
    if not table or len(table) < 2:
        return text_content, ""
    
    markdown_lines = []
    
    # Header row
    header_row = "| " + " | ".join(str(cell) if cell else "" for cell in table[0]) + " |"
    markdown_lines.append(header_row)
    
    # Separator row
    separator = "| " + " | ".join("---" for _ in table[0]) + " |"
    markdown_lines.append(separator)
    
    # Data rows
    for row in table[1:]:
        if row:
            data_row = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
            markdown_lines.append(data_row)
    
    markdown_content = "\n".join(markdown_lines)
    
    return text_content, markdown_content

def _save_llm_text_results(results: List[LLMTextResult], output_dir: Path, pdf_path: Path):
    """Save LLM-ready text results to files"""
    import json
    
    try:
        # Save paragraph_text.jsonl format
        paragraph_file = output_dir / f"{pdf_path.stem}_paragraph_text.jsonl"
        with open(paragraph_file, 'w', encoding='utf-8') as f:
            for result in results:
                # Create the paragraph record with all required fields
                paragraph_record = {
                    "doc_id": result.doc_id,
                    "page_no": result.page,
                    "region_type": result.region_type,
                    "text": result.text,
                    "bbox": result.bbox,
                    "ocr_conf": result.ocr_conf,
                    "source_sha256": result.source_sha256,
                    "region_id": result.region_id,
                    "engine": result.engine,
                    "engine_version": result.engine_version
                }
                f.write(json.dumps(paragraph_record) + '\n')
        
        logger.debug(f"Saved paragraph_text.jsonl to {paragraph_file}")
        
        # Group results by type
        by_type = {}
        for result in results:
            text_type = result.text_type
            if text_type not in by_type:
                by_type[text_type] = []
            by_type[text_type].append(result)
        
        # Save each type to separate files (keeping existing functionality)
        for text_type, type_results in by_type.items():
            filename = f"{pdf_path.stem}_llm_{text_type}.txt"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for result in type_results:
                    f.write(f"=== Page {result.page} ===\n")
                    f.write(result.text)
                    if result.markdown_content:
                        f.write(f"\n\n--- Markdown ---\n{result.markdown_content}")
                    if result.metadata:
                        f.write(f"\n\n--- Metadata ---\n{result.metadata}")
                    f.write("\n\n")
            
            logger.debug(f"Saved LLM {text_type} text to {filename}")
        
        # Save combined LLM-ready text
        combined_file = output_dir / f"{pdf_path.stem}_llm_combined.txt"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for result in sorted(results, key=lambda x: (x.page, x.text_type)):
                f.write(f"=== Page {result.page} - {result.text_type} ===\n")
                f.write(result.text)
                if result.markdown_content:
                    f.write(f"\n\n--- Markdown ---\n{result.markdown_content}")
                f.write("\n\n")
        
        # Create LLM-ready summary
        summary_file = output_dir / f"{pdf_path.stem}_llm_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"LLM-Ready Text Extraction Summary for {pdf_path.name}\n")
            f.write("=" * 60 + "\n\n")
            
            total_chars = sum(len(r.text) for r in results)
            total_pages = len(set(r.page for r in results))
            
            f.write(f"Total pages processed: {total_pages}\n")
            f.write(f"Total content blocks extracted: {len(results)}\n")
            f.write(f"Total characters: {total_chars:,}\n\n")
            
            f.write("Content types found:\n")
            for text_type, type_results in by_type.items():
                f.write(f"  {text_type}: {len(type_results)} blocks\n")
            
            f.write(f"\nLLM-Ready files created:\n")
            f.write(f"  {pdf_path.stem}_paragraph_text.jsonl\n")
            for text_type in by_type.keys():
                f.write(f"  {pdf_path.stem}_llm_{text_type}.txt\n")
            f.write(f"  {pdf_path.stem}_llm_combined.txt\n")
            f.write(f"  {pdf_path.stem}_llm_summary.txt\n")
        
        logger.info(f"LLM-ready text extraction completed for {pdf_path.name}")
        
    except Exception as e:
        logger.error(f"Failed to save LLM text results: {e}")
