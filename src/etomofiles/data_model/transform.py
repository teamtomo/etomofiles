"""Transform data model for IMOD xf files."""

import os
import warnings
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class TransformData(BaseModel):
    """Data model for IMOD .xf transformation files.
    
    An xf file contains one line with six numbers per image in the tilt-series,
    each specifying a 2D affine transformation:
    
        A11 A12 A21 A22 DX DY
    
    where the coordinate (X, Y) is transformed to (X', Y') by:
    
        X' = A11 * X + A12 * Y + DX
        Y' = A21 * X + A22 * Y + DY
    
    Attributes:
        transforms: Array of transformation parameters with shape (n, 6) where each row
                   contains [a11, a12, a21, a22, dx, dy], or None if file not found
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    transforms: np.ndarray | None = Field(default=None)

    @classmethod
    def from_directory(cls, directory: Path, ts_name: str) -> "TransformData":
        """Load transform data from a directory.
        
        Parameters
        ----------
        directory : Path
            Directory containing the .xf file
        ts_name : str
            Base name of the tilt series (e.g., 'TS_001')
            
        Returns
        -------
        TransformData
            Parsed transform data (None if file doesn't exist)
        """
        xf_data = cls.safe_read_xf(directory / f"{ts_name}.xf")
        return cls(transforms=xf_data)
    
    @staticmethod
    def read_xf(file: os.PathLike) -> np.ndarray:
        """Read an IMOD xf file into an (n, 6) numpy array.
        
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

    @classmethod
    def safe_read_xf(cls, file: os.PathLike) -> np.ndarray | None:
        """Safely read an IMOD xf file, returning None if it fails.
        
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
            return cls.read_xf(file)
        except (FileNotFoundError, ValueError) as e:
            warnings.warn(f"Skipping {file}: {e}", RuntimeWarning)
            return None
    
    def __repr__(self) -> str:
        """Return string representation."""
        n_transforms = len(self.transforms) if self.transforms is not None else 0
        return f"TransformData({n_transforms} transforms)"
