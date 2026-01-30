import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

from extractor.llm_tables import extract_llm_ready_tables, LLMTableResult
from extractor.llm_images import extract_llm_ready_images, extract_figures_with_embedded_text, export_figure_text_jsonl, LLMImageResult
from extractor.llm_text import extract_llm_ready_text, LLMTextResult
from utils.fs import discover_pdfs, ensure_dir
from utils.hashing import sha256_file
from utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class LLMExtractionResult:
    """Represents LLM-ready extraction results from a PDF"""
    pdf_path: Path
    tables: List[LLMTableResult]
    images: List[LLMImageResult]
    text: List[LLMTextResult]
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return f"LLM Extraction Result: {len(self.tables)} tables, {len(self.images)} images, {len(self.text)} text blocks"

def process_pdf_llm_ready(
    pdf_path: str | Path, 
    out_root: str | Path, 
    config: Dict[str, Any]
) -> Optional[LLMExtractionResult]:
    """
    Process a single PDF to extract LLM-ready tables, images, and text.
    
    Args:
        pdf_path: Path to the PDF file
        out_root: Root directory for outputs
        config: Configuration dictionary
        
    Returns:
        LLMExtractionResult or None if processing failed
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return None
        
    logger.info(f"Processing PDF for LLM-ready extraction: {pdf_path.name}")
    
    # Create output directory structure
    pdf_dir = Path(out_root) / pdf_path.stem
    ensure_dir(pdf_dir)
    
    try:
        # Get page range info from config
        page_range_info = config.get('page_range', {})
        page_range_enabled = page_range_info.get('enabled', False)
        start_page = page_range_info.get('start_page') if page_range_enabled else None
        end_page = page_range_info.get('end_page') if page_range_enabled else None
        
        # Get page count for metadata
        try:
            import fitz  # PyMuPDF
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
        except Exception:
            page_count = None
        
        # Get selected extractions (for interactive mode)
        selected_extractions = config.get('selected_extractions', ['tables', 'images', 'text'])
        
        # Extract LLM-ready tables
        tables = []
        tables_elapsed = 0.0
        if 'tables' in selected_extractions:
            t_start = time.perf_counter()
            logger.info(f"Starting table extraction for {pdf_path.name} using unstructured.partition_pdf")
            tables = extract_llm_ready_tables(
                pdf_path, 
                out_root,
                strategy=config['llm_tables']['strategy'],
                min_rows=config['llm_tables']['min_rows'],
                min_cols=config['llm_tables']['min_cols'],
                extract_page_headers=config['llm_tables']['extract_page_headers'],
                extract_sections=config['llm_tables']['extract_sections'],
                skip_empty_rows=config['llm_tables']['skip_empty_rows'],
                max_table_nesting_level=config['llm_tables']['max_table_nesting_level'],
                use_pymupdf4llm=config['llm_tables']['use_pymupdf4llm'],
                include_context=config['llm_tables']['include_context'],
                generate_markdown=config['llm_tables']['generate_markdown'],
                generate_json=config['llm_tables']['generate_json'],
                start_page=start_page,
                end_page=end_page,
                camelot_dpi=config['llm_tables'].get('camelot_dpi'),
                camelot_confidence_threshold=config['llm_tables'].get('camelot_confidence_threshold'),
                preprocess_text_enabled=config['llm_text'].get('preprocess_text', config['llm_text'].get('preprocess_text_enabled', True)),
                normalize_unicode=config['llm_text'].get('normalize_unicode', True),
                convert_ellipsis=config['llm_text'].get('convert_ellipsis', True)
            )
            t_end = time.perf_counter()
            tables_elapsed = t_end - t_start
            
            # Log extraction results
            if tables:
                logger.info(f"Extracted {len(tables)} LLM-ready tables from {pdf_path.name} using unstructured.partition_pdf")
            else:
                logger.info(f"No tables extracted from {pdf_path.name}")
        
        # Extract figures with embedded text
        images = []
        images_elapsed = 0.0
        if 'images' in selected_extractions:
            i_start = time.perf_counter()
            images = extract_figures_with_embedded_text(
                pdf_path, 
                output_dir=out_root,
                min_size=config['llm_images']['min_size'],
                dpi=config['llm_images']['dpi'],
                retry_dpi=400,
                use_paddleocr=True,
                use_tesseract=True,
                extract_captions=config['llm_images']['include_captions'],
                use_gpu=config['llm_images'].get('use_gpu', True),
                start_page=start_page,
                end_page=end_page
            )
            i_end = time.perf_counter()
            images_elapsed = i_end - i_start
            logger.info(f"Extracted {len(images)} figures with embedded text")
            
            # Export figure_text.jsonl
            if images:
                export_figure_text_jsonl(images, pdf_dir, pdf_path)
        
        # Extract LLM-ready text
        text_results = []
        text_elapsed = 0.0
        if 'text' in selected_extractions:
            text_start = time.perf_counter()
            text_results = extract_llm_ready_text(
                pdf_path,
                output_dir=out_root,
                use_pymupdf4llm=config['llm_text']['use_pymupdf4llm'],
                extract_headers=config['llm_text']['extract_headers'],
                extract_footers=config['llm_text']['extract_footers'],
                extract_captions=config['llm_text']['extract_captions'],
                extract_tables_as_markdown=config['llm_text']['extract_tables_as_markdown'],
                extract_images_with_captions=config['llm_text']['extract_images_with_captions'],
                preserve_formatting=config['llm_text']['preserve_formatting'],
                use_ocr=config['llm_text']['use_ocr'],
                min_confidence=config['llm_text']['min_confidence'],
                write_images=config['llm_text']['write_images'],
                embed_images=config['llm_text']['embed_images'],
                image_dir=config['llm_text'].get('image_dir'),
                preprocess_text_enabled=config['llm_text'].get('preprocess_text', True),
                normalize_unicode=config['llm_text'].get('normalize_unicode', True),
                convert_ellipsis=config['llm_text'].get('convert_ellipsis', True),
                start_page=start_page if page_range_enabled else None,
                end_page=end_page if page_range_enabled else None
            )
            text_end = time.perf_counter()
            text_elapsed = text_end - text_start
            logger.info(f"Extracted {len(text_results)} LLM-ready text blocks")
        
        # Save LLM-ready tables to multiple formats
        tables_dir = pdf_dir / "llm_tables"
        ensure_dir(tables_dir)
        
        for table in tables:
            # Save CSV
            csv_path = tables_dir / f"table_p{table.page}_i{table.index}.csv"
            table.dataframe.to_csv(csv_path, index=False)
            
            # Save Markdown
            if table.markdown_table:
                md_path = tables_dir / f"table_p{table.page}_i{table.index}.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(table.markdown_table)
            
            # Save JSON
            if table.json_table:
                json_path = tables_dir / f"table_p{table.page}_i{table.index}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(table.json_table)
            
            logger.debug(f"Saved LLM-ready table to {csv_path}")
        
        # Export table provenance data
        if tables:
            from extractor.llm_tables import export_table_provenance
            export_table_provenance(tables, pdf_dir, pdf_path)
        
        # Create LLM-ready result object
        result = LLMExtractionResult(
            pdf_path=pdf_path,
            tables=tables,
            images=images,
            text=text_results,
            metadata={
                'extraction_times': {
                    'tables': tables_elapsed,
                    'images': images_elapsed,
                    'text': text_elapsed,
                    'total_time': tables_elapsed + images_elapsed + text_elapsed
                },
                'extraction_methods': {
                    'tables': 'llm_enhanced',
                    'images': 'llm_enhanced',
                    'text': 'llm_enhanced'
                },
                'config': config,
                'page_range': {
                    'enabled': page_range_enabled,
                    'start_page': start_page if page_range_enabled else 1,
                    'end_page': end_page if page_range_enabled else page_count
                },
                'page_count': page_count
            }
        )
        
        # Generate LLM-ready manifest
        try:
            manifest_data = _create_llm_manifest_data(result, pdf_dir)
        except Exception as e:
            logger.error(f"Failed to create manifest data for {pdf_path.name}: {e}", exc_info=True)
            raise  # Re-raise to be caught by outer exception handler
        
        # Save manifest using atomic write to prevent corruption
        manifest_path = pdf_dir / "llm_manifest.json"
        temp_manifest_path = pdf_dir / "llm_manifest.json.tmp"
        
        try:
            # Write to temporary file first
            with open(temp_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Validate the written JSON before replacing
            with open(temp_manifest_path, "r", encoding="utf-8") as f:
                validated_data = json.load(f)  # Validate JSON is complete and valid
            
            # Verify the manifest has all required top-level keys
            required_keys = ['pdf', 'hash', 'extraction_type', 'tables', 'images', 'text_blocks']
            missing_keys = [key for key in required_keys if key not in validated_data]
            if missing_keys:
                raise ValueError(f"Manifest missing required keys: {missing_keys}")
            
            # Atomic replace: rename temp file to final file
            temp_manifest_path.replace(manifest_path)
            
            # Get file size for logging
            file_size = manifest_path.stat().st_size
            logger.info(f"Saved manifest: {manifest_path} ({file_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to save manifest {manifest_path}: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_manifest_path.exists():
                try:
                    temp_manifest_path.unlink()
                except:
                    pass
            raise  # Re-raise to be caught by outer exception handler
            
        logger.info(f"Completed LLM-ready processing {pdf_path.name}: {len(tables)} tables, {len(images)} images, {len(text_results)} text blocks")
        return result
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name} for LLM-ready extraction: {e}", exc_info=True)
        return None

def run_llm_pipeline(
    source: str | Path,
    out_root: str | Path = "multiple_docs_op",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run the LLM-ready extraction pipeline on a single PDF or directory of PDFs.
    
    Args:
        source: PDF file or directory containing PDFs
        out_root: Root directory for outputs
        config: Configuration dictionary (overrides defaults)
        
    Returns:
        Dictionary with summary of processing results
    """
    # Ensure output directory exists
    out_root = Path(out_root)
    ensure_dir(out_root)
    
    # Load default LLM config if none provided
    if config is None:
        config = _get_default_llm_config()
    
    # Find PDFs to process
    pdfs = list(discover_pdfs(source))
    logger.info(f"Found {len(pdfs)} PDFs to process for LLM-ready extraction")
    
    if not pdfs:
        logger.warning(f"No PDFs found at {source}")
        return {"success": False, "message": "No PDFs found", "processed": 0}
    
    # Process each PDF
    results = []
    successful = 0
    
    for pdf in pdfs:
        result = process_pdf_llm_ready(pdf, out_root, config)
        if result is not None:
            results.append(result)
            successful += 1
    
    # Generate LLM-ready summary
    summary = _create_llm_summary(results, successful, len(pdfs))
    
    # Save LLM-ready summary index
    index_path = out_root / "llm_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"LLM-ready pipeline completed: {successful}/{len(pdfs)} PDFs processed successfully")
    return summary

def _get_default_llm_config() -> Dict[str, Any]:
    """Get default configuration for LLM-ready extraction"""
    return {
        'llm_tables': {
            'strategy': 'hi_res',
            'min_rows': 1,
            'min_cols': 2,
            'extract_page_headers': True,
            'extract_sections': True,
            'skip_empty_rows': True,
            'max_table_nesting_level': 5,
            'use_pymupdf4llm': True,
            'include_context': True,
            'generate_markdown': True,
            'generate_json': True
        },
        'llm_images': {
            'min_size': 50,
            'dpi': 300,
            'include_captions': True,
            'merge_tolerance': 20,
            'extract_vector_graphics': True,
            'use_pymupdf4llm': True,
            'generate_base64': True,
            'include_context': True,
            'extract_ocr_text': True,
            'use_gpu': True  # Enable M1 GPU acceleration
        },
        'llm_text': {
            'use_pymupdf4llm': True,
            'extract_headers': True,
            'extract_footers': True,
            'extract_captions': True,
            'extract_tables_as_markdown': True,
            'extract_images_with_captions': True,
            'preserve_formatting': True,
            'use_ocr': False,
            'min_confidence': 0.5,
            'write_images': True,
            'embed_images': False
        }
    }

def _create_llm_manifest_data(result: LLMExtractionResult, pdf_dir: Path) -> Dict[str, Any]:
    """Create LLM-ready manifest data with complete file paths"""
    # Build directory paths
    tables_dir = pdf_dir / "llm_tables"
    images_dir = pdf_dir / "llm_images"
    text_dir = pdf_dir / "llm_text"
    
    # Build figure text jsonl path
    figure_text_jsonl = pdf_dir / f"{result.pdf_path.stem}_figure_text.jsonl" if result.images else None
    
    # Build text file paths
    paragraph_text_jsonl = text_dir / f"{result.pdf_path.stem}_paragraph_text.jsonl" if result.text else None
    combined_text = text_dir / f"{result.pdf_path.stem}_llm_combined.txt" if result.text else None
    summary_text = text_dir / f"{result.pdf_path.stem}_llm_summary.txt" if result.text else None
    
    # Get unique text types for individual text files
    text_types = set()
    if result.text:
        text_types = set(t.text_type for t in result.text)
    
    # Build manifest data
    manifest_data = {
        "pdf": str(result.pdf_path.absolute()),
        "hash": sha256_file(result.pdf_path),
        "extraction_type": "llm_ready",
        "tables_count": len(result.tables),
        "images_count": len(result.images),
        "text_blocks_count": len(result.text),
        "metadata": result.metadata,
        "tables": {
            "folder_path": str(tables_dir.absolute()),
        "tables": [
            {
                "page": t.page,
                "index": t.index,
                "rows": t.dataframe.shape[0],
                "cols": t.dataframe.shape[1],
                "caption": t.caption if t.caption else None,
                "has_markdown": bool(t.markdown_table),
                "has_json": bool(t.json_table),
                "has_context": bool(t.context),
                    "engine": t.engine if t.engine else None,
                    "engine_version": str(t.engine_version) if t.engine_version is not None else None,
                    "confidence": float(t.confidence) if t.confidence is not None else None,
                    "source": t.source if t.source else None,
                "extraction_method": t.metadata.get('extraction_method') if t.metadata else None,
                "used_ocr_fallback": t.metadata.get('used_ocr_fallback', False) if t.metadata else False,
                    # File paths - exact locations of extracted table files
                    "file_paths": {
                        "csv": str((tables_dir / f"table_p{t.page}_i{t.index}.csv").absolute()),
                        "markdown": str((tables_dir / f"table_p{t.page}_i{t.index}.md").absolute()) if t.markdown_table else None,
                        "json": str((tables_dir / f"table_p{t.page}_i{t.index}.json").absolute()) if t.json_table else None
                    }
            }
            for t in result.tables
            ]
        },
        "images": {
            "folder_path": str(images_dir.absolute()),
            "figure_text_jsonl": str(figure_text_jsonl.absolute()) if figure_text_jsonl else None,
        "images": [
            {
                "page": img.page,
                    "filename": img.filename.name if isinstance(img.filename, Path) else str(img.filename),
                "caption": img.caption,
                "hash": img.hash,
                "has_base64": bool(img.base64_data),
                "has_context": bool(img.context),
                    "has_ocr": bool(img.structured_data and img.structured_data.get('content_analysis', {}).get('ocr_text')),
                    # File paths - exact locations of extracted image files
                    "image_path": str(Path(img.filename).absolute()) if img.filename else None,
                    "file_paths": {
                        "image": str(Path(img.filename).absolute()) if img.filename else None
                    }
            }
            for img in result.images
            ]
        },
        "text_blocks": {
            "folder_path": str(text_dir.absolute()),
            "paragraph_text_jsonl": str(paragraph_text_jsonl.absolute()) if paragraph_text_jsonl else None,
            "combined_text": str(combined_text.absolute()) if combined_text else None,
            "summary_text": str(summary_text.absolute()) if summary_text else None,
            "text_type_files": {
                text_type: str((text_dir / f"{result.pdf_path.stem}_llm_{text_type}.txt").absolute())
                for text_type in text_types
            } if text_types else {},
        "text_blocks": [
            {
                "page": text.page,
                "text_type": text.text_type,
                "region_type": text.region_type,
                "text_length": len(text.text),
                "has_markdown": bool(text.markdown_content),
                "has_structured_data": bool(text.structured_data),
                    "confidence": float(text.confidence) if text.confidence is not None else None,
                    "ocr_conf": float(text.ocr_conf) if text.ocr_conf is not None else None,
                    "engine": text.engine if text.engine else None,
                    "engine_version": str(text.engine_version) if text.engine_version is not None else None,
                "bbox": text.bbox,
                "region_id": text.region_id,
                "doc_id": text.doc_id,
                "source_sha256": text.source_sha256,
                # Hybrid classification metadata
                "original_label": text.metadata.get('original_label') if text.metadata else None,
                "hybrid_label": text.metadata.get('hybrid_label') if text.metadata else None,
                "hybrid_label_source": text.metadata.get('hybrid_label_source') if text.metadata else None,
                "provenance": text.metadata.get('provenance') if text.metadata else None,
                "extraction_method": text.metadata.get('extraction_method') if text.metadata else None,
                "hybrid_text_replaced": text.metadata.get('hybrid_text_replaced', False) if text.metadata else False,
                "font_median": text.metadata.get('font_median') if text.metadata else None,
                "font_max": text.metadata.get('font_max') if text.metadata else None,
                "bold_fraction": text.metadata.get('bold_fraction') if text.metadata else None
            }
            for text in result.text
        ]
        }
    }
    
    return manifest_data

def update_existing_manifest(manifest_path: Path) -> bool:
    """
    Update an existing manifest file to the new structure with folder paths and file locations.
    
    Args:
        manifest_path: Path to the existing manifest file
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Read existing manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try to fix incomplete JSON (common issue with truncated files)
        try:
            manifest = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Manifest {manifest_path} has JSON errors, attempting to fix incomplete JSON...")
            # Try to close incomplete structures
            if content.endswith(','):
                content = content.rstrip().rstrip(',')
            
            # Find the last complete line and close structures
            lines = content.split('\n')
            # Remove incomplete last line if it exists
            if lines and not lines[-1].strip().endswith(('}', ']', ',')):
                # Check if last line looks incomplete
                last_line = lines[-1].strip()
                if last_line and not any(last_line.endswith(c) for c in ['}', ']', ',', 'null', 'true', 'false']):
                    lines = lines[:-1]
                    content = '\n'.join(lines)
            
            # Close arrays and objects
            open_braces = content.count('{') - content.count('}')
            open_brackets = content.count('[') - content.count(']')
            
            # Remove trailing comma if present
            content = content.rstrip().rstrip(',')
            
            # Close structures
            if open_brackets > 0:
                content += '\n' + '  ' * (len(lines) - lines.count('') - 1) + ']' * open_brackets
            if open_braces > 0:
                content += '\n' + '}' * open_braces
            
            try:
                manifest = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse manifest {manifest_path} even after repair attempt: {e}")
                # Try to reconstruct from what we have - at minimum preserve metadata
                logger.warning("Attempting to reconstruct manifest from available data...")
                # For now, we'll skip corrupted files and log them
                return False
        
        # Check if already in new format (tables is a dict with 'folder_path')
        if isinstance(manifest.get('tables'), dict) and 'folder_path' in manifest.get('tables', {}):
            logger.debug(f"Manifest {manifest_path} already in new format, skipping")
            return True
        
        # Get the PDF directory (parent of manifest file)
        pdf_dir = manifest_path.parent
        
        # Get PDF name from manifest or directory name
        pdf_path_str = manifest.get('pdf', '')
        if pdf_path_str:
            pdf_path = Path(pdf_path_str)
            pdf_stem = pdf_path.stem
        else:
            pdf_stem = pdf_dir.name
        
        # Build directory paths
        tables_dir = pdf_dir / "llm_tables"
        images_dir = pdf_dir / "llm_images"
        text_dir = pdf_dir / "llm_text"
        
        # Build figure text jsonl path
        figure_text_jsonl = pdf_dir / f"{pdf_stem}_figure_text.jsonl"
        
        # Build text file paths
        paragraph_text_jsonl = text_dir / f"{pdf_stem}_paragraph_text.jsonl"
        combined_text = text_dir / f"{pdf_stem}_llm_combined.txt"
        summary_text = text_dir / f"{pdf_stem}_llm_summary.txt"
        
        # Update tables section
        old_tables = manifest.get('tables', [])
        if isinstance(old_tables, list):
            updated_tables = []
            for table in old_tables:
                # Add file paths if not present
                page = table.get('page', 0)
                index = table.get('index', 0)
                
                csv_path = table.get('csv_path')
                if not csv_path:
                    csv_path = str((tables_dir / f"table_p{page}_i{index}.csv").absolute())
                
                markdown_path = table.get('markdown_path')
                if not markdown_path and table.get('has_markdown'):
                    markdown_path = str((tables_dir / f"table_p{page}_i{index}.md").absolute())
                
                json_path = table.get('json_path')
                if not json_path and table.get('has_json'):
                    json_path = str((tables_dir / f"table_p{page}_i{index}.json").absolute())
                
                # Update table entry
                updated_table = table.copy()
                # Remove old individual path fields if they exist (we use file_paths instead)
                updated_table.pop('csv_path', None)
                updated_table.pop('markdown_path', None)
                updated_table.pop('json_path', None)
                
                # Add file_paths object
                updated_table['file_paths'] = {
                    'csv': csv_path,
                    'markdown': markdown_path if markdown_path else None,
                    'json': json_path if json_path else None
                }
                updated_tables.append(updated_table)
            
            manifest['tables'] = {
                'folder_path': str(tables_dir.absolute()),
                'tables': updated_tables
            }
        
        # Update images section
        old_images = manifest.get('images', [])
        if isinstance(old_images, list):
            updated_images = []
            for img in old_images:
                # Get image path
                image_path = img.get('image_path')
                if not image_path:
                    filename = img.get('filename', '')
                    if filename:
                        if isinstance(filename, str) and not Path(filename).is_absolute():
                            image_path = str((images_dir / filename).absolute())
                        else:
                            image_path = str(Path(filename).absolute()) if filename else None
                    else:
                        image_path = None
                
                # Update image entry
                updated_img = img.copy()
                if image_path:
                    updated_img['image_path'] = image_path
                updated_img['file_paths'] = {
                    'image': image_path
                }
                updated_images.append(updated_img)
            
            manifest['images'] = {
                'folder_path': str(images_dir.absolute()),
                'figure_text_jsonl': str(figure_text_jsonl.absolute()) if figure_text_jsonl.exists() else None,
                'images': updated_images
            }
        
        # Update text_blocks section
        old_text_blocks = manifest.get('text_blocks', [])
        if isinstance(old_text_blocks, list):
            # Get unique text types
            text_types = set()
            for text_block in old_text_blocks:
                text_type = text_block.get('text_type')
                if text_type:
                    text_types.add(text_type)
            
            # Build text_type_files dictionary
            text_type_files = {}
            for text_type in text_types:
                text_file_path = text_dir / f"{pdf_stem}_llm_{text_type}.txt"
                if text_file_path.exists():
                    text_type_files[text_type] = str(text_file_path.absolute())
            
            manifest['text_blocks'] = {
                'folder_path': str(text_dir.absolute()),
                'paragraph_text_jsonl': str(paragraph_text_jsonl.absolute()) if paragraph_text_jsonl.exists() else None,
                'combined_text': str(combined_text.absolute()) if combined_text.exists() else None,
                'summary_text': str(summary_text.absolute()) if summary_text.exists() else None,
                'text_type_files': text_type_files,
                'text_blocks': old_text_blocks
            }
        
        # Save updated manifest using atomic write
        temp_manifest_path = manifest_path.with_suffix('.json.tmp')
        
        try:
            # Write to temporary file first
            with open(temp_manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Validate the written JSON before replacing
            with open(temp_manifest_path, 'r', encoding='utf-8') as f:
                json.load(f)  # Validate JSON is complete and valid
            
            # Atomic replace: rename temp file to final file
            temp_manifest_path.replace(manifest_path)
            
            logger.info(f"Updated manifest: {manifest_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save updated manifest {manifest_path}: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_manifest_path.exists():
                try:
                    temp_manifest_path.unlink()
                except:
                    pass
            return False
        
    except Exception as e:
        logger.error(f"Failed to update manifest {manifest_path}: {e}", exc_info=True)
        return False

def update_all_manifests(output_root: str | Path) -> Dict[str, Any]:
    """
    Update all existing manifest files in the output directory to the new structure.
    
    Args:
        output_root: Root directory containing document folders with manifests
        
    Returns:
        Dictionary with update results
    """
    output_root = Path(output_root)
    if not output_root.exists():
        logger.error(f"Output root directory does not exist: {output_root}")
        return {"success": False, "message": "Output directory does not exist", "updated": 0, "failed": 0}
    
    # Find all manifest files
    manifest_files = list(output_root.glob("**/llm_manifest.json"))
    
    if not manifest_files:
        logger.warning(f"No manifest files found in {output_root}")
        return {"success": True, "message": "No manifests found", "updated": 0, "failed": 0}
    
    logger.info(f"Found {len(manifest_files)} manifest files to update")
    
    updated = 0
    failed = 0
    
    for manifest_path in manifest_files:
        if update_existing_manifest(manifest_path):
            updated += 1
        else:
            failed += 1
    
    result = {
        "success": True,
        "total": len(manifest_files),
        "updated": updated,
        "failed": failed
    }
    
    logger.info(f"Manifest update completed: {updated} updated, {failed} failed out of {len(manifest_files)} total")
    return result

def _get_pdf_table_engines(tables: List[LLMTableResult]) -> Dict[str, int]:
    """Get engine breakdown for tables from a single PDF"""
    engine_count = {
        'partition_pdf': 0
    }
    
    for table in tables:
        engine = table.engine or 'partition_pdf'
        if engine in engine_count:
            engine_count[engine] += 1
        elif engine == 'unknown' and table.metadata:
            extraction_method = table.metadata.get('extraction_method', 'partition_pdf')
            if extraction_method in engine_count:
                engine_count[extraction_method] += 1
        else:
            # Default to partition_pdf
            engine_count['partition_pdf'] += 1
    
    return engine_count

def _create_llm_summary(results: List[LLMExtractionResult], successful: int, total: int) -> Dict[str, Any]:
    """Create LLM-ready summary"""
    summary = {
        "success": True,
        "extraction_type": "llm_ready",
        "processed": successful,
        "total": total,
        "pdfs": [
            {
                "path": str(r.pdf_path),
                "tables": len(r.tables),
                "images": len(r.images),
                "text_blocks": len(r.text),
                "extraction_times": r.metadata.get('extraction_times', {}) if r.metadata else {},
                "table_engines": _get_pdf_table_engines(r.tables)
            }
            for r in results
        ]
    }

    # Compute total extraction times
    total_times = {
        'tables_seconds': 0.0,
        'images_seconds': 0.0,
        'text_seconds': 0.0,
        'total_seconds': 0.0
    }
    
    for r in results:
        if r.metadata and 'extraction_times' in r.metadata:
            times = r.metadata['extraction_times']
            # Map metadata keys to summary keys
            # Metadata uses: 'tables', 'images', 'text', 'total_time'
            # Summary uses: 'tables_seconds', 'images_seconds', 'text_seconds', 'total_seconds'
            total_times['tables_seconds'] += times.get('tables', times.get('tables_seconds', 0.0))
            total_times['images_seconds'] += times.get('images', times.get('images_seconds', 0.0))
            total_times['text_seconds'] += times.get('text', times.get('text_seconds', 0.0))
            total_times['total_seconds'] += times.get('total_time', times.get('total_seconds', 0.0))
    
    summary['total_extraction_times'] = total_times
    
    # Compute total pages
    total_pages = 0
    for r in results:
        if r.metadata and 'page_count' in r.metadata:
            page_count = r.metadata.get('page_count')
            if page_count:
                total_pages += page_count
    summary['total_pages'] = total_pages if total_pages > 0 else None
    
    # Compute table extraction statistics (using only partition_pdf)
    engine_breakdown = {
        'partition_pdf': 0
    }
    
    for r in results:
        for table in r.tables:
            # All tables use partition_pdf
            engine = table.engine or 'partition_pdf'
            if engine in engine_breakdown:
                engine_breakdown[engine] += 1
            else:
                engine_breakdown['partition_pdf'] += 1
    
    # Add table statistics to summary
    summary['table_stats'] = {
        'engine_breakdown': engine_breakdown
    }
    
    return summary
