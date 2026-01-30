import logging
import sys
from pathlib import Path
from typing import Optional

def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """
    Create a logger with the specified name and configuration.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional file to log to
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # If logger already exists and has handlers, return it
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # Create file handler if requested
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_log_level(level: int):
    """Set log level for all loggers in the application."""
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith(__name__.split('.')[0]):
            logging.getLogger(logger_name).setLevel(level)