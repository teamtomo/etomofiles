"""
Input/output utilities for reading IMOD files.

This module provides functions to read IMOD files.
"""

import os
import numpy as np
import warnings


def read_tlt(file: os.PathLike) -> np.ndarray:
    """
    Read an IMOD tlt file into an (n, ) numpy array.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the tlt file
        
    Returns
    -------
    np.ndarray
        Array of tilt angles
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file cannot be parsed as tilt data
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    
    try:
        return np.loadtxt(fname=file, dtype=float).reshape(-1)
    except Exception as e:
        raise ValueError(f"Cannot parse file {file}: {e}") from e


def read_xf(file: os.PathLike) -> np.ndarray:
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
    np.ndarray
        Array of transformation parameters with shape (n, 6) where each row
        contains [a11, a12, a21, a22, dx, dy]
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file cannot be parsed as transform data
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    
    try:
        return np.loadtxt(fname=file, dtype=float).reshape(-1, 6)
    except Exception as e:
        raise ValueError(f"Cannot parse file {file}: {e}") from e


def safe_read_tlt(file: os.PathLike) -> np.ndarray | None:
    """
    Safely read an IMOD tlt file, returning None if it fails.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the tlt file
        
    Returns
    -------
    np.ndarray | None
        Array of tilt angles, or None if file doesn't exist or can't be read.
        Issues a warning when returning None.
    """
    try:
        return read_tlt(file)
    except (FileNotFoundError, ValueError) as e:
        warnings.warn(f"Skipping {file}: {e}", RuntimeWarning)
        return None


def safe_read_xf(file: os.PathLike) -> np.ndarray | None:
    """
    Safely read an IMOD xf file, returning None if it fails.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the xf file
        
    Returns
    -------
    np.ndarray | None
        Array of transformation parameters, or None if file doesn't exist or can't be read.
        Issues a warning when returning None.
    """
    try:
        return read_xf(file)
    except (FileNotFoundError, ValueError) as e:
        warnings.warn(f"Skipping {file}: {e}", RuntimeWarning)
        return None
