import hashlib
from pathlib import Path

def sha256_file(file_path: str | Path) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA-256 hash as a hex string
    """
    file_path = Path(file_path)
    h = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
            
    return h.hexdigest()