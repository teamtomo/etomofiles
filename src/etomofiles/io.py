"""Main I/O module for reading etomo files."""

from pathlib import Path

import pandas as pd

from .data_model import EtomoDataFile
from .utils import etomo_to_dataframe, validate_directory


def read(directory: str | Path) -> pd.DataFrame:
    """Read etomo alignment metadata from a directory into a pandas DataFrame.
    
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
    # Validate directory
    directory = Path(directory)
    validate_directory(directory)

    # Load alignment data using data models
    etomo_data = EtomoDataFile.from_directory(directory)

    # Convert to DataFrame
    return etomo_to_dataframe(etomo_data)
