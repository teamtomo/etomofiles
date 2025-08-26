"""
Input/output utilities for reading IMOD files.
s.
This module provides functions to read IMOD files.
"""

import os
import numpy as np
from typing import Optional


def read_tlt(file: os.PathLike) -> Optional[np.ndarray]:
    """
    Read an IMOD tlt file into an (n, ) numpy array.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the tlt file
        
    Returns
    -------
    np.ndarray or None
        Array of tilt angles, or None if file doesn't exist or can't be read
    """
    try:
        if not os.path.exists(file):
            return None
        return np.loadtxt(fname=file, dtype=float).reshape(-1)
    except Exception:
        return None


def read_xf(file: os.PathLike) -> Optional[np.ndarray]:
    """
    Read an IMOD xf file into an (n, 6) numpy array.
    An xf file contains one line with six numbers per image in the tilt-series,
    each specifying a linear transformation:

        A11 A12 A21 A22 DX DY

    where the coordinate (X, Y) is transformed to (X', Y') by:

        X' = A11 * X + A12 * Y + DX
        Y' = A21 * X + A22 * Y + DY
        
    Parameters
    ----------
    file : os.PathLike
        Path to the xf file
        
    Returns
    -------
    np.ndarray or None
        Array of transformation parameters with shape (n, 6) where each row
        contains [a11, a12, a21, a22, dx, dy], or None if file doesn't exist or can't be read
    """
    try:
        if not os.path.exists(file):
            return None
        return np.loadtxt(fname=file, dtype=float).reshape(-1, 6)
    except Exception:
        return None
