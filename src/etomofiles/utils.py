"""
Utility functions for etomo file processing.

This module provides helper functions.
"""

from pathlib import Path


def validate_directory(directory: str | Path) -> None:
    """
    Validate that a directory exists and is accessible.
    
    Parameters
    ----------
    directory : str or Path
        Path to the directory to validate
        
    Raises
    ------
    ValueError
        If directory doesn't exist or is not a directory
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}") 