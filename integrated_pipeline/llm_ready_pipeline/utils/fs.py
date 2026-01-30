from pathlib import Path
from typing import List, Iterator

def discover_pdfs(path: str | Path) -> Iterator[Path]:
    """
    Find PDF files in a path (single file or directory).
    
    Args:
        path: Path to a PDF file or directory
        
    Yields:
        Path objects for each PDF found
    """
    path = Path(path)
    
    # Single file
    if path.is_file() and path.suffix.lower() == ".pdf":
        yield path
    
    # Directory - find all PDFs
    elif path.is_dir():
        for pdf_file in path.glob("**/*.pdf"):
            yield pdf_file

def ensure_dir(directory: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to create
        
    Returns:
        Path to the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory