from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import pandas as pd

@dataclass
class TableResult:
    """Represents an extracted table from a PDF"""
    pdf_path: Path
    page: int
    index: int
    dataframe: pd.DataFrame
    
    def __str__(self) -> str:
        return f"Table {self.index} (Page {self.page}): {self.dataframe.shape[0]}×{self.dataframe.shape[1]}"

@dataclass
class ImageResult:
    """Represents an extracted image from a PDF"""
    pdf_path: Path
    page: int
    filename: Path
    caption: Optional[str]
    hash: str
    
    def __str__(self) -> str:
        caption_text = f", Caption: {self.caption}" if self.caption else ""
        return f"Image (Page {self.page}){caption_text}: {self.filename.name}"

@dataclass
class TextResult:
    """Represents extracted text from a PDF"""
    pdf_path: Path
    page: int
    text: str
    text_type: str  # 'main_text', 'header', 'footer', 'caption', etc.
    confidence: Optional[float] = None
    
    def __str__(self) -> str:
        return f"Text (Page {self.page}, {self.text_type}): {len(self.text)} chars"

@dataclass
class ExtractionResult:
    """Combined result of PDF extraction (tables, images, and text)"""
    pdf_path: Path
    tables: List[TableResult]
    images: List[ImageResult]
    text: List[TextResult]
    
    def __str__(self) -> str:
        return f"{self.pdf_path.name}: {len(self.tables)} tables, {len(self.images)} images, {len(self.text)} text blocks"