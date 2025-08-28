"""
etomofiles: A Python package for reading IMOD etomo alignment files.

This package provides utilities to read etomo alignment metadata
and expose it as pandas DataFrames.
"""

from .reader import read
from .io import read_tlt, read_xf, safe_read_tlt, safe_read_xf

__all__ = [
    "read",
    "read_tlt", 
    "read_xf",
    "safe_read_tlt",
    "safe_read_xf"
]