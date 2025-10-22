"""Utility functions for etomo file processing and DataFrame conversion."""

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .data_model.etomo_data import EtomoDataFile # break circular import


def validate_directory(directory: str | Path) -> None:
    """Validate that a directory exists and is accessible.
    
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


def read_tlt(file: os.PathLike) -> np.ndarray:
    """Read an IMOD tlt file into an (n, ) numpy array.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the tlt file
        
    Returns
    -------
    np.ndarray
        Array of tilt angles
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    
    try:
        return np.loadtxt(fname=file, dtype=float).reshape(-1)
    except Exception as e:
        raise ValueError(f"Cannot parse file {file}: {e}") from e


def safe_read_tlt(file: os.PathLike) -> np.ndarray | None:
    """Safely read an IMOD tlt file, returning None if it fails.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the tlt file
        
    Returns
    -------
    np.ndarray | None
        Array of tilt angles, or None if file doesn't exist or can't be read.
    """
    try:
        return read_tlt(file)
    except (FileNotFoundError, ValueError) as e:
        warnings.warn(f"Skipping {file}: {e}", RuntimeWarning)
        return None


def read_xf(file: os.PathLike) -> np.ndarray:
    """Read an IMOD xf file into an (n, 6) numpy array.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the xf file
        
    Returns
    -------
    np.ndarray
        Array of transformation parameters with shape (n, 6)
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    
    try:
        return np.loadtxt(fname=file, dtype=float).reshape(-1, 6)
    except Exception as e:
        raise ValueError(f"Cannot parse file {file}: {e}") from e


def safe_read_xf(file: os.PathLike) -> np.ndarray | None:
    """Safely read an IMOD xf file, returning None if it fails.
    
    Parameters
    ----------
    file : os.PathLike
        Path to the xf file
        
    Returns
    -------
    np.ndarray | None
        Array of transformation parameters, or None if file doesn't exist or can't be read.
    """
    try:
        return read_xf(file)
    except (FileNotFoundError, ValueError) as e:
        warnings.warn(f"Skipping {file}: {e}", RuntimeWarning)
        return None


def etomo_to_dataframe(etomo_data: "EtomoDataFile") -> pd.DataFrame:
    """Convert EtomoDataFile to a pandas DataFrame.
    
    Parameters
    ----------
    etomo_data : EtomoDataFile
        Parsed etomo alignment data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - image_path: Path to the specific image in tilt series (e.g., TS_001.st[0])
        - idx_tilt: Index of the tilt image (0-based)
        - tilt_axis_angle: Tilt axis rotation angle
        - rawtlt: Raw tilt angle (degrees)
        - tlt: Processed/corrected tilt angle (degrees)
        - xtilt: Additional tilt information
        - xf_a11, xf_a12, xf_a21, xf_a22, xf_dx, xf_dy: xf transformation matrix elements
        - excluded: Boolean indicating if view was excluded
    """
    n_images = etomo_data.n_images
    basename = etomo_data.basename
    ext = etomo_data.tilt_series_extension
    
    df = pd.DataFrame({
        'image_path': [f"{basename}.{ext}[{i}]" for i in range(n_images)],
        'idx_tilt': range(n_images),
        'tilt_axis_angle': etomo_data.tilt_axis_angle,
        'rawtlt': _pad_array(etomo_data.rawtlt, n_images),
        'tlt': _pad_array(etomo_data.tlt, n_images),
        'xtilt': _pad_array(etomo_data.xtilt, n_images),
        'xf_a11': _pad_transform(etomo_data.xf, n_images, 0),
        'xf_a12': _pad_transform(etomo_data.xf, n_images, 1),
        'xf_a21': _pad_transform(etomo_data.xf, n_images, 2),
        'xf_a22': _pad_transform(etomo_data.xf, n_images, 3),
        'xf_dx': _pad_transform(etomo_data.xf, n_images, 4),
        'xf_dy': _pad_transform(etomo_data.xf, n_images, 5),
        'excluded': [i in etomo_data.excluded_views for i in range(1, n_images + 1)]
    })
    
    return df


def _pad_array(data: np.ndarray | None, n_images: int) -> list:
    """Pad array to n_images length with NaN.
    
    Parameters
    ----------
    data : np.ndarray | None
        Array to pad, or None
    n_images : int
        Target length
        
    Returns
    -------
    list
        Padded list of length n_images
    """
    if data is None:
        return [np.nan] * n_images
    
    result = [np.nan] * n_images
    for i, val in enumerate(data[:n_images]):
        result[i] = float(val)
    return result


def _pad_transform(data: np.ndarray | None, n_images: int, col: int) -> list:
    """Pad transform column to n_images length with NaN.
    
    Parameters
    ----------
    data : np.ndarray | None
        Transform array with shape (n, 6), or None
    n_images : int
        Target length
    col : int
        Column index to extract (0-5)
        
    Returns
    -------
    list
        Padded list of length n_images
    """
    if data is None:
        return [np.nan] * n_images
    
    result = [np.nan] * n_images
    for i, row in enumerate(data[:n_images]):
        result[i] = float(row[col])
    return result
