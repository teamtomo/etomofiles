"""
Main reader module for etomo files.

This module provides the parser for etomo alignment data
into pandas DataFrames.
"""

from pathlib import Path
from typing import Tuple, Set
import numpy as np
import pandas as pd
import mrcfile

from .io import read_tlt, read_xf, safe_read_tlt, safe_read_xf
from .utils import validate_directory


def read(directory: str | Path) -> pd.DataFrame:
    """
    Read etomo alignment metadata from a directory into a pandas DataFrame.
    
    Parameters
    ----------
    directory : str or Path
        Path to the directory containing etomo alignment files
        
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
    directory = Path(directory)
    validate_directory(directory)
    
    # Parse .edf file for all primary metadata
    ts_name, ts_ext, tilt_axis_angle, excluded_views, n_images = _parse_edf_file(directory)
    
    # Read alignment files
    tlt_data = safe_read_tlt(directory / f"{ts_name}.tlt")
    xf_data = safe_read_xf(directory / f"{ts_name}.xf")
    rawtlt_data = safe_read_tlt(directory / f"{ts_name}.rawtlt")
    xtilt_data = safe_read_tlt(directory / f"{ts_name}.xtilt")
    
    # Build DataFrame - each row is an image in the tilt series
    df = pd.DataFrame({
        'image_path': [f"{ts_name}.{ts_ext}[{i}]" for i in range(n_images)],
        'idx_tilt': range(n_images),
        'tilt_axis_angle': tilt_axis_angle,
        'rawtlt': _pad_array(rawtlt_data, n_images),
        'tlt': _pad_array(tlt_data, n_images),
        'xtilt': _pad_array(xtilt_data, n_images),
        'xf_a11': _pad_transform(xf_data, n_images, 0),
        'xf_a12': _pad_transform(xf_data, n_images, 1),
        'xf_a21': _pad_transform(xf_data, n_images, 2),
        'xf_a22': _pad_transform(xf_data, n_images, 3),
        'xf_dx': _pad_transform(xf_data, n_images, 4),
        'xf_dy': _pad_transform(xf_data, n_images, 5),
        'excluded': [i in excluded_views for i in range(1, n_images + 1)]
    })
    
    return df


def _parse_edf_file(directory: Path) -> Tuple[str, str, float | None, Set[int], int]:
    """Parse .edf file to extract all primary metadata."""
    edf_files = list(directory.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError("No .edf file found")
    
    edf_file = edf_files[0]
    ts_name = None
    ts_ext = None
    tilt_axis_angle = None
    excluded_views = set()
    
    with open(edf_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'Setup.DatasetName':
                ts_name = value
            elif key == 'Setup.RawImageStackExt':
                ts_ext = value
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
    
    if not ts_name:
        raise ValueError("Setup.DatasetName not found in .edf file")
    if not ts_ext:
        raise ValueError("Setup.RawImageStackExt not found in .edf file")
    
    # Get number of images from MRC header
    ts_file = directory / f"{ts_name}.{ts_ext}"
    if not ts_file.exists():
        raise FileNotFoundError(f"Tilt series file not found: {ts_file}")
    
    try:
        with mrcfile.open(ts_file, header_only=True) as mrc:
            n_images = int(mrc.header.nz)
    except Exception:
        raise ValueError("Cannot determine number of images from stack")
    
    return ts_name, ts_ext, tilt_axis_angle, excluded_views, n_images


def _pad_array(data: np.ndarray | None, n_images: int) -> list:
    """Pad array to n_images length with NaN."""
    if data is None:
        return [np.nan] * n_images
    
    result = [np.nan] * n_images
    for i, val in enumerate(data[:n_images]):
        result[i] = float(val)
    return result


def _pad_transform(data: np.ndarray | None, n_images: int, col: int) -> list:
    """Pad transform column to n_images length with NaN."""
    if data is None:
        return [np.nan] * n_images
    
    result = [np.nan] * n_images
    for i, row in enumerate(data[:n_images]):
        result[i] = float(row[col])
    return result 