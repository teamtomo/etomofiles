"""Utility functions for etomo file processing and DataFrame conversion."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data_model.etomo_data import EtomoData


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


def etomo_to_dataframe(etomo_data: EtomoData) -> pd.DataFrame:
    """Convert EtomoData to a pandas DataFrame.
    
    Each row in the DataFrame represents one tilt image in the series.
    
    Parameters
    ----------
    etomo_data : EtomoData
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
    edf = etomo_data.edf_metadata
    tilt = etomo_data.tilt_angles
    xf = etomo_data.transforms
    
    n_images = edf.n_images
    ts_name = edf.ts_name
    ts_ext = edf.ts_ext
    
    # Build DataFrame - each row is an image in the tilt series
    df = pd.DataFrame({
        'image_path': [f"{ts_name}.{ts_ext}[{i}]" for i in range(n_images)],
        'idx_tilt': range(n_images),
        'tilt_axis_angle': edf.tilt_axis_angle,
        'rawtlt': _pad_array(tilt.rawtlt, n_images),
        'tlt': _pad_array(tilt.tlt, n_images),
        'xtilt': _pad_array(tilt.xtilt, n_images),
        'xf_a11': _pad_transform(xf.transforms, n_images, 0),
        'xf_a12': _pad_transform(xf.transforms, n_images, 1),
        'xf_a21': _pad_transform(xf.transforms, n_images, 2),
        'xf_a22': _pad_transform(xf.transforms, n_images, 3),
        'xf_dx': _pad_transform(xf.transforms, n_images, 4),
        'xf_dy': _pad_transform(xf.transforms, n_images, 5),
        'excluded': [i in edf.excluded_views for i in range(1, n_images + 1)]
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
