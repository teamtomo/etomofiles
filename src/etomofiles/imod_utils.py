"""IMOD utility functions for working with transformation matrices and file parsing."""

import os
from pathlib import Path
from typing import Set, TypedDict, Union

import mrcfile
import numpy as np
import pandas as pd

from .utils import read_xf


class EdfMetadata(TypedDict):
    """Structure for parsed .edf file metadata."""
    basename: str
    tilt_series_extension: str
    tilt_axis_angle: float | None
    excluded_views: Set[int]
    n_images: int


def parse_edf(directory: Path) -> EdfMetadata:
    """Parse IMOD .edf file and extract alignment metadata.
    
    Parameters
    ----------
    directory : Path
        Directory containing the .edf file and tilt series
        
    Returns
    -------
    EdfMetadata
        Dictionary containing:
        - basename: Tilt series name
        - tilt_series_extension: File extension (e.g., 'st', 'mrc')
        - tilt_axis_angle: Tilt axis rotation angle (degrees)
        - excluded_views: Set of view indices (1-indexed) to exclude
        - n_images: Total number of images in the tilt series
        
    Raises
    ------
    FileNotFoundError
        If no .edf file found or tilt series file not found
    ValueError
        If required fields missing or cannot determine image count
    """
    # Find .edf file
    edf_files = list(directory.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError("No .edf file found")
    
    edf_file = edf_files[0]
    
    # Read .edf file
    try:
        with open(edf_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        with open(edf_file, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # Parse .edf content
    basename = None
    tilt_series_extension = None
    tilt_axis_angle = None
    excluded_views: Set[int] = set()
    
    for line in content.splitlines():
        line = line.strip()
        if '=' not in line:
            continue
        
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        if key == 'Setup.DatasetName':
            basename = value
        elif key == 'Setup.RawImageStackExt':
            tilt_series_extension = value
        elif key == 'Setup.ImageRotationA':
            try:
                tilt_axis_angle = float(value) if value else None
            except ValueError:
                tilt_axis_angle = None
        elif key == 'Setup.AxisA.ExcludeProjections':
            if value:
                for part in value.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        excluded_views.update(range(start, end + 1))
                    elif part.isdigit():
                        excluded_views.add(int(part))
    
    # Validate required fields
    if not basename:
        raise ValueError("Setup.DatasetName not found in .edf file")
    if not tilt_series_extension:
        raise ValueError("Setup.RawImageStackExt not found in .edf file")
    
    # Get number of images from MRC header
    ts_file = directory / f"{basename}.{tilt_series_extension}"
    if not ts_file.exists():
        raise FileNotFoundError(f"Tilt series file not found: {ts_file}")
    
    try:
        with mrcfile.open(ts_file, header_only=True) as mrc:
            n_images = int(mrc.header.nz)
    except Exception:
        raise ValueError("Cannot determine number of images from stack")
    
    return EdfMetadata(
        basename=basename,
        tilt_series_extension=tilt_series_extension,
        tilt_axis_angle=tilt_axis_angle,
        excluded_views=excluded_views,
        n_images=n_images,
    )


def df_to_xf(
    data: Union[pd.DataFrame, np.ndarray, os.PathLike],
    yx: bool = False,
) -> np.ndarray:
    """Output xf as numpy array (ntilt, 2, 3).
    
    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or PathLike
        Transformation data:
        - DataFrame: Must have columns xf_a11, xf_a12, xf_a21, xf_a22, xf_dx, xf_dy
        - Path: Path to .xf file to read
    yx : bool, default False
        Matrix row ordering:
        - False: [[A11, A12, DX], [A21, A22, DY]] (xy convention)
        - True:  [[A22, A21, DY], [A12, A11, DX]] (yx convention)
    Returns
    -------
    np.ndarray
        Transformation matrices with shape (n_tilts, 2, 3)
    
    Notes
    -----
    The transformation applies as:
        X' = A11 * X + A12 * Y + DX
        Y' = A21 * X + A22 * Y + DY
    """
    if isinstance(data, pd.DataFrame):
        xf_cols = ['xf_a11', 'xf_a12', 'xf_a21', 'xf_a22', 'xf_dx', 'xf_dy']
        if not all(col in data.columns for col in xf_cols):
            raise ValueError(f"DataFrame must contain columns: {xf_cols}")
        xf_data = data[xf_cols].values
        
    elif isinstance(data, (str, Path, os.PathLike)):
        # Read from file
        xf_data = read_xf(data)
        
    else:
        raise TypeError(
            f"data must be DataFrame or path, got {type(data)}"
        )
    
    if xf_data is None or len(xf_data) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    
    if xf_data.ndim != 2 or xf_data.shape[1] != 6:
        raise ValueError(
            f"Transform data must have shape (n, 6), got {xf_data.shape}"
        )
    
    n_tilts = len(xf_data)
        
    if  yx:
        # yx: [[A22, A21, DY], [A12, A11, DX]]
        # Rearrange columns: [A22, A21, DY, A12, A11, DX]
        reordered = xf_data[:, [3, 2, 5, 1, 0, 4]]

    else:
        # xy [[A11, A12, DX], [A21, A22, DY]]
        # Rearrange columns: [A11, A12, DX, A21, A22, DY]
        reordered = xf_data[:, [0, 1, 4, 2, 3, 5]]
    
    # Reshape to (n, 2, 3)
    return reordered.reshape(n_tilts, 2, 3)
