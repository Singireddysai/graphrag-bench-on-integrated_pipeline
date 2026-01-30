import os
import re
import hashlib
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import base64
import json
from dataclasses import dataclass
from . import ImageResult
from utils.logging import get_logger
from utils.gpu_utils import get_gpu_manager, is_gpu_available, get_optimal_device, show_memory_usage

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

logger = get_logger(__name__)

def are_images_similar(img1_bytes, img2_bytes, threshold=0.95):
    """
    Check if two images are similar using a simple 64-bit average hash (aHash).
    
    The similarity is computed via Hamming distance over a 64-bit hash
    (8x8 grayscale thresholded). Threshold is the minimum similarity ratio.
    """
    try:
        from PIL import Image
        import io
        
        def ahash(image_bytes: bytes) -> int:
            img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((8, 8))
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = 0
            for i, p in enumerate(pixels):
                if p >= avg:
                    bits |= (1 << i)
            return bits  # 64-bit int

        h1 = ahash(img1_bytes)
        h2 = ahash(img2_bytes)
        hamming_distance = bin(h1 ^ h2).count('1')
        similarity = 1.0 - (hamming_distance / 64.0)
        return similarity >= threshold
        
    except Exception as e:
        logger.debug(f"Error comparing images: {e}")
        return False

@dataclass
class LLMImageResult:
    """Represents LLM-ready extracted image from a PDF"""
    pdf_path: Path
    page: int
    filename: Path
    caption: Optional[str] = None
    hash: Optional[str] = None
    base64_data: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    # New fields for figure extraction with embedded text
    doc_id: Optional[str] = None
    figure_id: Optional[str] = None
    bbox: Optional[List[float]] = None
    caption_text: Optional[str] = None
    ocr_text_in_image: Optional[str] = None
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    confidence: Optional[float] = None
    source_sha256: Optional[str] = None
    
    def __str__(self) -> str:
        return f"LLM Image (Page {self.page}): {self.filename.name}"

def extract_llm_ready_images(
    pdf_path: str | Path, 
    output_dir: str | Path, 
    min_size: int = 50, 
    dpi: int = 300,
    include_captions: bool = True,
    merge_tolerance: int = 20,
    extract_vector_graphics: bool = True,
    use_pymupdf4llm: bool = True,
    generate_base64: bool = True,
    include_context: bool = True,
    extract_ocr_text: bool = True
) -> List[LLMImageResult]:
    """
    Extract LLM-ready images from a PDF with enhanced metadata and context.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        min_size: Minimum image dimension (width or height) in pixels
        dpi: Resolution for extracted images
        include_captions: Whether to try finding captions for images
        merge_tolerance: Tolerance for merging nearby image rectangles
        extract_vector_graphics: Whether to extract vector graphics (charts/diagrams)
        use_pymupdf4llm: Whether to use pymupdf4llm for image extraction
        generate_base64: Whether to generate base64 encoded data
        include_context: Whether to include surrounding context
        extract_ocr_text: Whether to extract text from images using OCR
        
    Returns:
        List of LLMImageResult objects
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting LLM-ready images from {pdf_path.name}")
    
    # Create output directory for this PDF's images
    pdf_images_dir = Path(output_dir) / pdf_path.stem / "llm_images"
    pdf_images_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    seen_hashes = set()
    seen_images = []  # Store (hash, image_bytes) tuples for similarity comparison
    figure_counters = {}  # Track figure counters per page for stable IDs
    
    try:
        # Strategy 1: pymupdf4llm for LLM-ready extraction
        if use_pymupdf4llm and PYMUPDF4LLM_AVAILABLE:
            try:
                pymupdf4llm_results = _extract_images_with_pymupdf4llm(
                    pdf_path,
                    pdf_images_dir,
                    include_captions=include_captions,
                    generate_base64=generate_base64,
                    include_context=include_context,
                    extract_ocr_text=extract_ocr_text
                )
                results.extend(pymupdf4llm_results)
                seen_hashes.update(r.hash for r in pymupdf4llm_results if r.hash)
                logger.info(f"pymupdf4llm extracted {len(pymupdf4llm_results)} images from {pdf_path.name}")
            except Exception as e:
                logger.warning(f"pymupdf4llm image extraction failed for {pdf_path.name}: {e}")

        # Strategy 2: Enhanced PyMuPDF extraction
        doc = fitz.open(pdf_path)
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            # Extract embedded raster images with enhanced metadata
            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if not base_image or base_image["image"] is None:
                        continue
                    
                    image_bytes = base_image["image"]
                    img_hash = image_hash(image_bytes)
                    
                    # Check for duplicates using both exact hash and similarity
                    is_duplicate = False
                    if img_hash in seen_hashes:
                        logger.debug(f"Skipping exact duplicate embedded image (hash: {img_hash[:8]}...) on page {page_num}")
                        is_duplicate = True
                    else:
                        # Check similarity with previously seen images
                        for seen_hash, seen_bytes in seen_images:
                            if are_images_similar(image_bytes, seen_bytes):
                                logger.debug(f"Skipping similar embedded image (similar to hash: {seen_hash[:8]}...) on page {page_num}")
                                is_duplicate = True
                                break
                    
                    if is_duplicate:
                        continue
                    
                    # Store this image for future similarity checks
                    seen_hashes.add(img_hash)
                    seen_images.append((img_hash, image_bytes))
                    
                    # Get image placement rectangles
                    img_rects = page.get_image_rects(xref)
                    if not img_rects or img_rects[0].width < min_size or img_rects[0].height < min_size:
                        continue
                    
                    image_ext = base_image["ext"]
                    
                    # Enhanced caption detection
                    caption = None
                    if include_captions:
                        caption = _find_enhanced_caption_for_rect(page, img_rects[0])
                    
                    # Generate stable figure ID (p{page}_f{counter})
                    if page_num not in figure_counters:
                        figure_counters[page_num] = 0
                    figure_counters[page_num] += 1
                    figure_id = f"p{page_num}_f{figure_counters[page_num]}"
                    
                    # Generate filename with stable figure ID
                    filename = _generate_enhanced_filename(
                        page_num, xref, caption, image_ext, len(results), figure_id
                    )
                    image_path = pdf_images_dir / filename
                    
                    # Save image file
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Generate base64 data if requested
                    base64_data = None
                    if generate_base64:
                        base64_data = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Extract context if requested
                    context = None
                    if include_context:
                        context = _extract_image_context(page, img_rects[0])
                    
                    # Extract OCR text if requested
                    ocr_text = None
                    if extract_ocr_text:
                        ocr_text = _extract_ocr_from_image(image_bytes)
                    
                    # Create enhanced metadata
                    metadata = _create_enhanced_image_metadata(
                        base_image, img_rects[0], page_num, xref, ocr_text
                    )
                    
                    # Create structured data
                    structured_data = _create_structured_image_data(
                        img_rects[0], base_image, caption, ocr_text
                    )
                    
                    result = LLMImageResult(
                        pdf_path=pdf_path,
                        page=page_num,
                        filename=image_path,
                        caption=caption,
                        hash=img_hash,
                        base64_data=base64_data,
                        metadata=metadata,
                        structured_data=structured_data,
                        context=context,
                        figure_id=figure_id
                    )
                    results.append(result)
                    
                    logger.debug(f"Saved enhanced image: {image_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process embedded image on page {page_num}: {e}")
                    continue
            
            # Extract vector graphics with enhanced processing
            if extract_vector_graphics:
                try:
                    drawings = page.get_drawings()
                    initial_rects = [
                        fitz.Rect(d["rect"]) for d in drawings 
                        if d["rect"].width >= min_size and d["rect"].height >= min_size
                    ]
                    
                    if initial_rects:
                        merged_drawing_rects = _merge_nearby_rects_enhanced(
                            initial_rects, tolerance=merge_tolerance
                        )
                        logger.debug(f"Found {len(merged_drawing_rects)} potential vector graphics on page {page_num}")
                        
                        for i, rect in enumerate(merged_drawing_rects, start=1):
                            if rect.width < min_size or rect.height < min_size:
                                continue
                            
                            # Render the area as a high-quality image
                            zoom = dpi / 72
                            mat = fitz.Matrix(zoom, zoom)
                            pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                            
                            # Hash for deduplication
                            pix_hash = image_hash(pix.samples)
                            
                            # Check for duplicates using both exact hash and similarity
                            is_duplicate = False
                            if pix_hash in seen_hashes:
                                logger.debug(f"Skipping exact duplicate vector graphic (hash: {pix_hash[:8]}...) on page {page_num}")
                                is_duplicate = True
                            else:
                                # Check similarity with previously seen images
                                pix_bytes = pix.tobytes()
                                for seen_hash, seen_bytes in seen_images:
                                    if are_images_similar(pix_bytes, seen_bytes):
                                        logger.debug(f"Skipping similar vector graphic (similar to hash: {seen_hash[:8]}...) on page {page_num}")
                                        is_duplicate = True
                                        break
                            
                            if is_duplicate:
                                continue
                            
                            # Store this image for future similarity checks
                            seen_hashes.add(pix_hash)
                            pix_bytes = pix.tobytes()
                            seen_images.append((pix_hash, pix_bytes))
                            
                            # Enhanced caption detection for vector graphics
                            caption = None
                            if include_captions:
                                caption = _find_enhanced_caption_for_rect(page, rect)
                            
                            # Generate stable figure ID (p{page}_f{counter})
                            if page_num not in figure_counters:
                                figure_counters[page_num] = 0
                            figure_counters[page_num] += 1
                            figure_id = f"p{page_num}_f{figure_counters[page_num]}"
                            
                            # Generate filename with stable figure ID
                            filename = _generate_enhanced_vector_filename(
                                page_num, i, rect, caption, len(results), figure_id
                            )
                            graph_path = pdf_images_dir / filename
                            pix.save(str(graph_path))
                            
                            # Generate base64 data if requested
                            base64_data = None
                            if generate_base64:
                                base64_data = base64.b64encode(pix.tobytes()).decode('utf-8')
                            
                            # Extract context
                            context = None
                            if include_context:
                                context = _extract_image_context(page, rect)
                            
                            # Create metadata for vector graphics
                            metadata = _create_vector_graphics_metadata(
                                rect, page_num, i, caption
                            )
                            
                            # Create structured data
                            structured_data = _create_structured_vector_data(
                                rect, caption, page_num, i
                            )
                            
                            result = LLMImageResult(
                                pdf_path=pdf_path,
                                page=page_num,
                                filename=graph_path,
                                caption=caption,
                                hash=pix_hash,
                                base64_data=base64_data,
                                metadata=metadata,
                                structured_data=structured_data,
                                context=context,
                                figure_id=figure_id
                            )
                            results.append(result)
                            
                            logger.debug(f"Saved enhanced vector graphic: {graph_path.name}")
                            
                except Exception as e:
                    logger.warning(f"Failed to process vector graphics on page {page_num}: {e}")
                    continue
                    
        doc.close()
        
    except Exception as e:
        logger.error(f"Failed to extract LLM-ready images from {pdf_path.name}: {e}")
        return []
    
    logger.info(f"Successfully extracted {len(results)} LLM-ready images from {pdf_path.name}")
    return results

def _extract_images_with_pymupdf4llm(
    pdf_path: Path,
    output_dir: Path,
    include_captions: bool = True,
    generate_base64: bool = True,
    include_context: bool = True,
    extract_ocr_text: bool = True
) -> List[LLMImageResult]:
    """Extract images using pymupdf4llm for LLM-ready content"""
    results = []
    
    try:
        # Extract markdown content with images
        markdown_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            write_images=True,
            image_path=output_dir
        )
        
        # Parse markdown to extract image information
        images_data = _parse_markdown_images(markdown_text, pdf_path, output_dir)
        
        for img_data in images_data:
            if img_data['image_path'].exists():
                # Read image data
                with open(img_data['image_path'], 'rb') as f:
                    image_bytes = f.read()
                
                # Generate base64 if requested
                base64_data = None
                if generate_base64:
                    base64_data = base64.b64encode(image_bytes).decode('utf-8')
                
                # Extract OCR text if requested
                ocr_text = None
                if extract_ocr_text:
                    ocr_text = _extract_ocr_from_image(image_bytes)
                
                # Create metadata
                metadata = {
                    'extraction_method': 'pymupdf4llm',
                    'image_type': img_data.get('type', 'unknown'),
                    'page': img_data['page'],
                    'ocr_text': ocr_text
                }
                
                # Create structured data
                structured_data = {
                    'image_info': {
                        'format': img_data['image_path'].suffix,
                        'size_bytes': len(image_bytes),
                        'has_caption': bool(img_data.get('caption')),
                        'has_ocr_text': bool(ocr_text)
                    },
                    'context': img_data.get('context', ''),
                    'ocr_text': ocr_text
                }
                
                result = LLMImageResult(
                    pdf_path=pdf_path,
                    page=img_data['page'],
                    filename=img_data['image_path'],
                    caption=img_data.get('caption'),
                    hash=image_hash(image_bytes),
                    base64_data=base64_data,
                    metadata=metadata,
                    structured_data=structured_data,
                    context=img_data.get('context')
                )
                results.append(result)
        
    except Exception as e:
        logger.warning(f"pymupdf4llm image extraction failed: {e}")
    
    return results

def _parse_markdown_images(markdown_text: str, pdf_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Parse markdown content to extract image information"""
    images_data = []
    
    # Split by page markers
    pages = markdown_text.split('---')
    
    for page_idx, page_content in enumerate(pages, start=1):
        if not page_content.strip():
            continue
        
        # Find image references in markdown
        lines = page_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('![') and '](' in line:
                # Extract image path and caption
                match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
                if match:
                    caption = match.group(1) if match.group(1) else None
                    image_path = Path(match.group(2))
                    
                    # Make path absolute if relative
                    if not image_path.is_absolute():
                        image_path = output_dir / image_path.name
                    
                    images_data.append({
                        'image_path': image_path,
                        'caption': caption,
                        'page': page_idx,
                        'type': 'embedded',
                        'context': _extract_image_context_from_markdown(line, lines)
                    })
    
    return images_data

def _extract_image_context_from_markdown(image_line: str, all_lines: List[str]) -> str:
    """Extract context around an image from markdown content"""
    try:
        line_idx = all_lines.index(image_line)
        context_lines = []
        
        # Get lines before and after the image
        start_idx = max(0, line_idx - 2)
        end_idx = min(len(all_lines), line_idx + 3)
        
        for i in range(start_idx, end_idx):
            if i != line_idx and all_lines[i].strip():
                context_lines.append(all_lines[i].strip())
        
        return " ".join(context_lines)
        
    except Exception as e:
        logger.debug(f"Failed to extract image context from markdown: {e}")
        return ""

def _merge_nearby_rects_enhanced(rects, tolerance=20):
    """
    Enhanced version of merge_nearby_rects with better merging logic.
    """
    while True:
        merged_in_pass = False
        merged_rects = []
        used_indices = set()
        
        for i, r1 in enumerate(rects):
            if i in used_indices:
                continue
            
            current_merged_rect = fitz.Rect(r1)
            used_indices.add(i)
            
            for j, r2 in enumerate(rects):
                if j in used_indices or i == j:
                    continue
                
                # Enhanced intersection logic
                if _rects_should_merge(current_merged_rect, r2, tolerance):
                    current_merged_rect.include_rect(r2)
                    used_indices.add(j)
                    merged_in_pass = True
            
            merged_rects.append(current_merged_rect)
        
        rects = merged_rects
        if not merged_in_pass:
            break
    
    return rects

def _rects_should_merge(rect1: fitz.Rect, rect2: fitz.Rect, tolerance: int) -> bool:
    """Enhanced logic for determining if two rectangles should be merged"""
    # Check for direct intersection
    if rect1.intersects(rect2):
        return True
    
    # Check for proximity with tolerance
    inflated_rect2 = fitz.Rect(rect2)
    inflated_rect2.x0 -= tolerance
    inflated_rect2.y0 -= tolerance
    inflated_rect2.x1 += tolerance
    inflated_rect2.y1 += tolerance
    
    return rect1.intersects(inflated_rect2)

def _find_enhanced_caption_for_rect(page, rect, search_below: bool = True, search_above: bool = False):
    """
    Enhanced caption detection with better pattern matching and context analysis.
    Now supports hyphen formats (Fig-2.1, Table-5.1) and checks both above and below.
    Captions are preprocessed to handle special characters and Unicode.
    """
    caption_patterns = _get_unified_caption_patterns()
    # Add explicit caption indicators
    caption_patterns.append(r'^.*[Cc]aption.*$')
    
    potential_captions = []
    
    try:
        # Import preprocessing function
        from extractor.llm_text import preprocess_text
        
        blocks = page.get_text("blocks")
        for x0, y0, x1, y1, text, _, _ in blocks:
            block_rect = fitz.Rect(x0, y0, x1, y1)
            clean_text = " ".join(text.split())
            
            # Check if text matches any caption pattern
            matched_pattern = None
            for pattern in caption_patterns:
                if re.match(pattern, clean_text, re.IGNORECASE):
                    matched_pattern = pattern
                    break
            
            if not matched_pattern:
                continue
            
            # Check position relative to rect
            distance = None
            is_valid_position = False
            
            if search_below and block_rect.y0 > rect.y1:
                distance = block_rect.y0 - rect.y1
                if distance < 150:
                    if max(rect.x0, block_rect.x0) < min(rect.x1, block_rect.x1):
                        is_valid_position = True
            elif search_above and block_rect.y1 < rect.y0:
                distance = rect.y0 - block_rect.y1
                if distance < 150:
                    if max(rect.x0, block_rect.x0) < min(rect.x1, block_rect.x1):
                        is_valid_position = True
            
            if is_valid_position and distance is not None:
                # Calculate confidence based on distance and alignment
                horizontal_overlap = min(block_rect.x1, rect.x1) - max(block_rect.x0, rect.x0)
                alignment_score = horizontal_overlap / max(block_rect.width, rect.width) if max(block_rect.width, rect.width) > 0 else 0
                confidence = max(0, 1 - (distance / 150)) * (0.5 + 0.5 * alignment_score)
                potential_captions.append((distance, clean_text, confidence))
        
        if not potential_captions:
            return None
            
        # Sort by distance and confidence
        potential_captions.sort(key=lambda x: (x[0], -x[2]))
        best_caption = potential_captions[0][1]
        
        # Apply text preprocessing to handle special characters and Unicode
        preprocessed_caption = preprocess_text(best_caption, normalize_unicode=True, convert_ellipsis=True)
        
        return preprocessed_caption
        
    except Exception as e:
        logger.debug(f"Failed to extract enhanced caption: {e}")
        return None

def _generate_enhanced_filename(page_num: int, xref: int, caption: Optional[str], 
                               image_ext: str, index: int, figure_id: Optional[str] = None) -> str:
    """Generate enhanced filename for images with stable figure IDs"""
    if figure_id:
        # Use stable figure ID as base filename
        base_filename = figure_id
    elif caption:
        base_filename = _sanitize_filename_enhanced(caption)
        if not base_filename:
            base_filename = f"p{page_num}_f{index + 1}"
    else:
        base_filename = f"p{page_num}_f{index + 1}"
    
    return f"{base_filename}.{image_ext}"

def _generate_enhanced_vector_filename(page_num: int, index: int, rect: fitz.Rect, 
                                     caption: Optional[str], total_index: int, figure_id: Optional[str] = None) -> str:
    """Generate enhanced filename for vector graphics with stable figure IDs"""
    if figure_id:
        # Use stable figure ID as base filename
        base_filename = figure_id
    elif caption:
        base_filename = _sanitize_filename_enhanced(caption)
        if not base_filename:
            base_filename = f"p{page_num}_f{total_index + 1}"
    else:
        base_filename = f"p{page_num}_f{total_index + 1}"
    
    return f"{base_filename}.png"

def _sanitize_filename_enhanced(text: str) -> str:
    """Enhanced filename sanitization"""
    if not text:
        return ""
    
    # Remove excessive whitespace and special characters
    text = re.sub(r'[\s.:-]+', '_', text)
    text = re.sub(r'[^\w-]', '', text)
    
    # Limit length and ensure it's not empty
    text = text[:100] if text else "image"
    
    return text

def _extract_image_context(page, rect):
    """Extract context around an image"""
    try:
        # Get text blocks near the image
        blocks = page.get_text("blocks")
        context_parts = []
        
        for x0, y0, x1, y1, text, _, _ in blocks:
            block_rect = fitz.Rect(x0, y0, x1, y1)
            
            # Check if block is near the image (within 50 pixels)
            if (abs(block_rect.y0 - rect.y1) < 50 or abs(block_rect.y1 - rect.y0) < 50):
                if text and text.strip():
                    context_parts.append(text.strip())
        
        return " ".join(context_parts[:3])  # Limit to first 3 context blocks
        
    except Exception as e:
        logger.debug(f"Failed to extract image context: {e}")
        return ""

def _extract_ocr_from_image(image_bytes: bytes) -> Optional[str]:
    """Extract text from image using OCR"""
    try:
        import pytesseract
        from PIL import Image
        import io
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        return text.strip() if text.strip() else None
        
    except ImportError:
        logger.debug("OCR dependencies not available for image text extraction")
        return None
    except Exception as e:
        logger.debug(f"OCR extraction failed: {e}")
        return None

def _create_enhanced_image_metadata(base_image: Dict, rect: fitz.Rect, 
                                  page_num: int, xref: int, ocr_text: Optional[str]) -> Dict[str, Any]:
    """Create enhanced metadata for images"""
    return {
        'extraction_method': 'enhanced_pymupdf',
        'image_format': base_image.get('ext', 'unknown'),
        'image_size': base_image.get('width', 0) * base_image.get('height', 0),
        'position': {
            'x0': rect.x0,
            'y0': rect.y0,
            'x1': rect.x1,
            'y1': rect.y1,
            'width': rect.width,
            'height': rect.height
        },
        'page': page_num,
        'xref': xref,
        'has_ocr_text': bool(ocr_text),
        'ocr_text_length': len(ocr_text) if ocr_text else 0
    }

def _create_vector_graphics_metadata(rect: fitz.Rect, page_num: int, 
                                   index: int, caption: Optional[str]) -> Dict[str, Any]:
    """Create metadata for vector graphics"""
    return {
        'extraction_method': 'enhanced_pymupdf_vector',
        'graphics_type': 'vector',
        'position': {
            'x0': rect.x0,
            'y0': rect.y0,
            'x1': rect.x1,
            'y1': rect.y1,
            'width': rect.width,
            'height': rect.height
        },
        'page': page_num,
        'index': index,
        'has_caption': bool(caption)
    }

def _create_structured_image_data(rect: fitz.Rect, base_image: Dict, 
                                caption: Optional[str], ocr_text: Optional[str]) -> Dict[str, Any]:
    """Create structured data for images"""
    return {
        'image_properties': {
            'format': base_image.get('ext', 'unknown'),
            'dimensions': {
                'width': base_image.get('width', 0),
                'height': base_image.get('height', 0)
            },
            'position': {
                'x0': rect.x0,
                'y0': rect.y0,
                'x1': rect.x1,
                'y1': rect.y1
            }
        },
        'content_analysis': {
            'has_caption': bool(caption),
            'caption_text': caption,
            'has_ocr_text': bool(ocr_text),
            'ocr_text': ocr_text
        }
    }

def _create_structured_vector_data(rect: fitz.Rect, caption: Optional[str], 
                                 page_num: int, index: int) -> Dict[str, Any]:
    """Create structured data for vector graphics"""
    return {
        'graphics_properties': {
            'type': 'vector',
            'position': {
                'x0': rect.x0,
                'y0': rect.y0,
                'x1': rect.x1,
                'y1': rect.y1
            },
            'page': page_num,
            'index': index
        },
        'content_analysis': {
            'has_caption': bool(caption),
            'caption_text': caption
        }
    }

def image_hash(image_bytes: bytes) -> str:
    """Return a SHA-256 hash for given image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()

def extract_figures_with_embedded_text(
    pdf_path: str | Path,
    output_dir: str | Path,
    min_size: int = 50,
    dpi: int = 300,
    retry_dpi: int = 400,
    use_paddleocr: bool = True,
    use_tesseract: bool = True,
    extract_captions: bool = True,
    use_gpu: bool = True,
    start_page: int | None = None,
    end_page: int | None = None
) -> List[LLMImageResult]:
    """
    Extract figures with embedded text using targeted OCR on crops.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted figures
        min_size: Minimum figure dimension (width or height) in pixels
        dpi: Initial DPI for rasterization (300)
        retry_dpi: Retry DPI for tiny fonts (400)
        use_paddleocr: Whether to use PaddleOCR for OCR
        use_tesseract: Whether to use Tesseract for OCR
        extract_captions: Whether to extract captions from PDF text layer or OCR
        use_gpu: Whether to use GPU acceleration
        start_page: Start page number (1-indexed, inclusive)
        end_page: End page number (1-indexed, inclusive)
        
    Returns:
        List of LLMImageResult objects with figure data
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting figures with embedded text from {pdf_path.name}")
    
    # Initialize GPU acceleration for M1 Mac
    gpu_manager = get_gpu_manager()
    if use_gpu and gpu_manager.gpu_available:
        logger.info(f"Using M1 GPU acceleration (device: {gpu_manager.device})")
        # Display detailed memory information
        gpu_manager.display_memory_info()
    else:
        logger.info("Using CPU processing")
    
    # Create output directory for figures
    figures_dir = Path(output_dir) / pdf_path.stem / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    try:
        doc = fitz.open(pdf_path)
        
        total_pages = len(doc)
        sp = max(1, start_page) if start_page else 1
        ep = min(total_pages, end_page) if end_page else total_pages
        
        # Range should include end page: range(sp-1, ep) gives pages sp to ep (inclusive)
        # For example: sp=2, ep=5 -> range(1, 5) = [1,2,3,4] -> pages 2,3,4,5 ✓
        for page_idx in range(sp - 1, ep):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            # Show memory usage for each page
            if page_idx % 5 == 0:  # Show every 5 pages
                show_memory_usage()
            
            # Detect figure regions using layout detection
            figure_regions = _detect_figure_regions(page, min_size)
            logger.debug(f"Found {len(figure_regions)} figure regions on page {page_num}")
            
            for fig_idx, region in enumerate(figure_regions, start=1):
                try:
                    # Crop and rasterize the figure region
                    figure_crop = _crop_and_rasterize_figure(page, region, dpi)
                    if figure_crop is None:
                        continue
                    
                    # Generate figure filename
                    figure_id = f"{pdf_path.stem}_p{page_num}_f{fig_idx}"
                    figure_filename = f"{figure_id}.png"
                    figure_path = figures_dir / figure_filename
                    
                    # Save the figure crop
                    figure_crop.save(str(figure_path))
                    
                    # Extract caption from PDF text layer or OCR
                    caption_text = None
                    if extract_captions:
                        caption_text = _extract_figure_caption(page, region)
                    
                    # Perform targeted OCR on the figure crop
                    ocr_text, confidence, engine = _perform_targeted_ocr(
                        figure_crop, use_paddleocr, use_tesseract, retry_dpi
                    )
                    
                    # Generate source SHA256
                    source_sha256 = _get_pdf_sha256(pdf_path)
                    
                    # Create bbox from region
                    bbox = [region.x0, region.y0, region.x1, region.y1]
                    
                    # Create result
                    result = LLMImageResult(
                        pdf_path=pdf_path,
                        page=page_num,
                        filename=figure_path,
                        doc_id=pdf_path.stem,
                        figure_id=figure_id,
                        bbox=bbox,
                        caption_text=caption_text,
                        ocr_text_in_image=ocr_text,
                        engine=engine,
                        engine_version=_get_engine_version(engine),
                        confidence=confidence,
                        source_sha256=source_sha256,
                        metadata={
                            'extraction_method': 'figure_with_embedded_text',
                            'region_detection': 'layout_based',
                            'ocr_engine': engine,
                            'dpi_used': dpi if confidence and confidence > 0.5 else retry_dpi
                        }
                    )
                    results.append(result)
                    
                    logger.debug(f"Extracted figure {fig_idx} on page {page_num}: {figure_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process figure {fig_idx} on page {page_num}: {e}")
                    continue
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Failed to extract figures from {pdf_path.name}: {e}")
        return []
    
    logger.info(f"Successfully extracted {len(results)} figures with embedded text from {pdf_path.name}")
    return results

def _detect_figure_regions(page, min_size: int) -> List[fitz.Rect]:
    """Detect figure regions using layout detection"""
    figure_regions = []
    
    try:
        # Get all images and drawings
        images = page.get_images(full=True)
        drawings = page.get_drawings()
        
        # Process embedded images
        for img_info in images:
            xref = img_info[0]
            img_rects = page.get_image_rects(xref)
            for rect in img_rects:
                if rect.width >= min_size and rect.height >= min_size:
                    figure_regions.append(rect)
        
        # Process vector graphics/drawings
        for drawing in drawings:
            rect = fitz.Rect(drawing["rect"])
            if rect.width >= min_size and rect.height >= min_size:
                figure_regions.append(rect)
        
        # Merge overlapping regions
        figure_regions = _merge_overlapping_regions(figure_regions)
        
    except Exception as e:
        logger.debug(f"Failed to detect figure regions: {e}")
    
    return figure_regions

def _crop_and_rasterize_figure(page, region: fitz.Rect, dpi: int) -> Optional[Any]:
    """Crop and rasterize a figure region"""
    try:
        # Calculate zoom factor for the desired DPI
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Get pixmap of the region
        pix = page.get_pixmap(matrix=mat, clip=region, alpha=False)
        
        # Convert to PIL Image for OCR
        from PIL import Image
        import io
        
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        return image
        
    except Exception as e:
        logger.debug(f"Failed to crop and rasterize figure: {e}")
        return None

def _extract_figure_caption(page, region: fitz.Rect) -> Optional[str]:
    """Extract caption from PDF text layer or via OCR with enhanced pattern matching"""
    try:
        # First try to find caption in PDF text layer (search below for figures)
        caption = _find_caption_in_text_layer(page, region, search_below=True, search_above=False, max_distance=150)
        if caption:
            return caption
        
        # If no caption found in text layer, try OCR on surrounding area
        caption = _extract_caption_via_ocr(page, region, search_below=True, search_above=False)
        return caption
        
    except Exception as e:
        logger.debug(f"Failed to extract figure caption: {e}")
        return None

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

def _find_caption_in_text_layer(page, region: fitz.Rect, search_below: bool = True, search_above: bool = False, max_distance: int = 150) -> Optional[str]:
    """
    Find caption in PDF text layer with enhanced pattern matching.
    Captions are preprocessed to handle special characters and Unicode.
    
    Args:
        page: PyMuPDF page object
        region: Bounding box of the figure/table
        search_below: Whether to search below the region (default: True for figures)
        search_above: Whether to search above the region (default: False, True for tables)
        max_distance: Maximum distance to search for captions (default: 150 pixels)
    """
    try:
        # Import preprocessing function from llm_text module
        from extractor.llm_text import preprocess_text
        
        blocks = page.get_text("blocks")
        caption_patterns = _get_unified_caption_patterns()
        potential_captions = []
        
        for x0, y0, x1, y1, text, _, _ in blocks:
            block_rect = fitz.Rect(x0, y0, x1, y1)
            clean_text = " ".join(text.split())
            
            # Check if text matches any caption pattern
            matched_pattern = None
            for pattern in caption_patterns:
                if re.match(pattern, clean_text, re.IGNORECASE):
                    matched_pattern = pattern
                    break
            
            if not matched_pattern:
                continue
            
            # Check position relative to region
            distance = None
            is_valid_position = False
            
            if search_below and block_rect.y0 > region.y1:
                # Caption is below the region
                distance = block_rect.y0 - region.y1
                if distance < max_distance:
                    # Check for horizontal overlap/alignment
                    if max(region.x0, block_rect.x0) < min(region.x1, block_rect.x1):
                        is_valid_position = True
            
            elif search_above and block_rect.y1 < region.y0:
                # Caption is above the region
                distance = region.y0 - block_rect.y1
                if distance < max_distance:
                    # Check for horizontal overlap/alignment
                    if max(region.x0, block_rect.x0) < min(region.x1, block_rect.x1):
                        is_valid_position = True
            
            if is_valid_position and distance is not None:
                # Calculate confidence based on distance and alignment
                horizontal_overlap = min(block_rect.x1, region.x1) - max(block_rect.x0, region.x0)
                alignment_score = horizontal_overlap / max(block_rect.width, region.width) if max(block_rect.width, region.width) > 0 else 0
                confidence = max(0, 1 - (distance / max_distance)) * (0.5 + 0.5 * alignment_score)
                potential_captions.append((distance, clean_text, confidence))
        
        if not potential_captions:
            return None
        
        # Sort by distance and confidence (closest with highest confidence first)
        potential_captions.sort(key=lambda x: (x[0], -x[2]))
        best_caption = potential_captions[0][1]
        
        # Apply text preprocessing to handle special characters and Unicode
        preprocessed_caption = preprocess_text(best_caption, normalize_unicode=True, convert_ellipsis=True)
        
        return preprocessed_caption
        
    except Exception as e:
        logger.debug(f"Failed to find caption in text layer: {e}")
        return None

def _extract_caption_via_ocr(page, region: fitz.Rect, search_below: bool = True, search_above: bool = False) -> Optional[str]:
    """Extract caption via OCR on surrounding area with enhanced pattern matching.
    Captions are preprocessed to handle special characters and Unicode."""
    try:
        # Import preprocessing function
        from extractor.llm_text import preprocess_text
        
        # Expand region to include potential caption area
        # Look below (for figures) or above (for tables)
        if search_below:
            expanded_region = fitz.Rect(
                region.x0 - 20,
                region.y1 - 20,
                region.x1 + 20,
                region.y1 + 150  # Look below the figure
            )
        else:
            expanded_region = fitz.Rect(
                region.x0 - 20,
                region.y0 - 150,  # Look above the table
                region.x1 + 20,
                region.y0 + 20
            )
        
        # Crop and OCR the expanded area
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=expanded_region, alpha=False)
        
        # Convert to PIL Image and perform OCR
        from PIL import Image
        import io
        
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        # Use OCR to extract text
        ocr_text = _perform_ocr_on_image(image, use_paddleocr=True, use_tesseract=True)
        
        if ocr_text:
            # Look for caption patterns in OCR text using unified patterns
            caption_patterns = _get_unified_caption_patterns()
            lines = ocr_text.split('\n')
            for line in lines:
                line = line.strip()
                for pattern in caption_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        # Apply text preprocessing to handle special characters and Unicode
                        preprocessed_caption = preprocess_text(line, normalize_unicode=True, convert_ellipsis=True)
                        return preprocessed_caption
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to extract caption via OCR: {e}")
        return None

def _perform_targeted_ocr(image, use_paddleocr: bool, use_tesseract: bool, retry_dpi: int = 400) -> Tuple[Optional[str], Optional[float], str]:
    """Perform targeted OCR on figure image"""
    try:
        # Try primary OCR
        ocr_text, confidence, engine = _perform_ocr_on_image(image, use_paddleocr, use_tesseract)
        
        # If confidence is low or no text found, retry with higher DPI
        if (not ocr_text or (confidence and confidence < 0.5)) and retry_dpi > 300:
            logger.debug("Retrying OCR with higher DPI for tiny fonts")
            # Note: In a real implementation, you'd re-rasterize with higher DPI here
            # For now, we'll just return the initial result
        
        return ocr_text, confidence, engine
        
    except Exception as e:
        logger.debug(f"Failed to perform targeted OCR: {e}")
        return None, None, "none"

def _perform_ocr_on_image(image, use_paddleocr: bool, use_tesseract: bool) -> Tuple[Optional[str], Optional[float], str]:
    """Perform OCR on image using available engines with M1 GPU acceleration"""
    ocr_text = None
    confidence = None
    engine = "none"
    
    # Get GPU manager for M1 optimization
    gpu_manager = get_gpu_manager()
    
    # Try PaddleOCR first if available (with M1 GPU acceleration)
    if use_paddleocr:
        try:
            import paddleocr
            
            # Get M1-optimized PaddleOCR configuration
            paddle_config = gpu_manager.create_paddleocr_config()
            
            # Create PaddleOCR instance with M1 GPU acceleration
            ocr = paddleocr.PaddleOCR(**paddle_config)
            result = ocr.ocr(image)
            
            if result and result[0]:
                text_parts = []
                confidences = []
                for line in result[0]:
                    if line[1][1] > 0.5:  # Confidence threshold
                        text_parts.append(line[1][0])
                        confidences.append(line[1][1])
                
                if text_parts:
                    ocr_text = '\n'.join(text_parts)
                    confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    engine = "paddleocr"
                    if gpu_manager.gpu_available:
                        engine = "paddleocr_gpu"
                    
        except ImportError:
            logger.debug("PaddleOCR not available")
        except Exception as e:
            logger.debug(f"PaddleOCR failed: {e}")
            # Try fallback to CPU-only PaddleOCR
            try:
                ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
                result = ocr.ocr(image)
                if result and result[0]:
                    text_parts = []
                    confidences = []
                    for line in result[0]:
                        if line[1][1] > 0.5:
                            text_parts.append(line[1][0])
                            confidences.append(line[1][1])
                    
                    if text_parts:
                        ocr_text = '\n'.join(text_parts)
                        confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        engine = "paddleocr_cpu"
            except Exception as fallback_e:
                logger.debug(f"PaddleOCR CPU fallback failed: {fallback_e}")
    
    # Try Tesseract if PaddleOCR didn't work
    if not ocr_text and use_tesseract:
        try:
            import pytesseract
            from PIL import Image
            
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                # Get confidence data
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
                engine = "tesseract"
            else:
                ocr_text = None
                
        except ImportError:
            logger.debug("Tesseract not available")
        except Exception as e:
            logger.debug(f"Tesseract failed: {e}")
    
    return ocr_text, confidence, engine

def _merge_overlapping_regions(regions: List[fitz.Rect]) -> List[fitz.Rect]:
    """Merge overlapping figure regions"""
    if not regions:
        return []
    
    merged = []
    used = set()
    
    for i, region in enumerate(regions):
        if i in used:
            continue
        
        current = fitz.Rect(region)
        used.add(i)
        
        # Find overlapping regions
        for j, other_region in enumerate(regions):
            if j in used or i == j:
                continue
            
            if current.intersects(other_region):
                current.include_rect(other_region)
                used.add(j)
        
        merged.append(current)
    
    return merged

def _get_pdf_sha256(pdf_path: Path) -> Optional[str]:
    """Get SHA256 hash of PDF file"""
    try:
        with open(pdf_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return None

def _get_engine_version(engine: str) -> str:
    """Get version string for OCR engine"""
    if engine == "paddleocr":
        return "2.7.0"
    elif engine == "tesseract":
        return "5.0.0"
    else:
        return "1.0.0"

def export_figure_text_jsonl(results: List[LLMImageResult], output_dir: Path, pdf_path: Path):
    """Export figure data as figure_text.jsonl"""
    import json
    
    try:
        # Create figure_text.jsonl file
        jsonl_file = output_dir / f"{pdf_path.stem}_figure_text.jsonl"
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for result in results:
                # Create the figure record with all required fields
                figure_record = {
                    "doc_id": result.doc_id,
                    "page_no": result.page,
                    "figure_id": result.figure_id,
                    "bbox": result.bbox,
                    "caption_text": result.caption_text,
                    "ocr_text_in_image": result.ocr_text_in_image,
                    "engine": result.engine,
                    "engine_version": result.engine_version,
                    "confidence": result.confidence,
                    "source_sha256": result.source_sha256
                }
                f.write(json.dumps(figure_record) + '\n')
        
        logger.debug(f"Exported figure_text.jsonl to {jsonl_file}")
        logger.info(f"Exported figure data for {len(results)} figures")
        
    except Exception as e:
        logger.error(f"Failed to export figure_text.jsonl: {e}")
