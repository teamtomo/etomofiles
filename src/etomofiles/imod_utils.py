"""IMOD utility functions for working with transformation matrices."""

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .utils import read_xf


def xf_to_array(
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
