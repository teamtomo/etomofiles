"""Tilt angle data model for IMOD tlt files."""

import os
import warnings
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class TiltAngleData(BaseModel):
    """Data model for IMOD tilt angle files (.tlt, .rawtlt, .xtilt).
    
    Attributes:
        tlt: Processed/corrected tilt angles (degrees) from .tlt file
        rawtlt: Raw tilt angles (degrees) from .rawtlt file
        xtilt: Additional tilt information from .xtilt file
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    tlt: np.ndarray | None = Field(default=None)
    rawtlt: np.ndarray | None = Field(default=None)
    xtilt: np.ndarray | None = Field(default=None)

    @classmethod
    def from_directory(cls, directory: Path, ts_name: str) -> "TiltAngleData":
        """Load tilt angle data from a directory.
        
        Parameters
        ----------
        directory : Path
            Directory containing the tilt angle files
        ts_name : str
            Base name of the tilt series (e.g., 'TS_001')
            
        Returns
        -------
        TiltAngleData
            Parsed tilt angle data (files that don't exist will be None)
        """
        tlt_data = cls.safe_read_tlt(directory / f"{ts_name}.tlt")
        rawtlt_data = cls.safe_read_tlt(directory / f"{ts_name}.rawtlt")
        xtilt_data = cls.safe_read_tlt(directory / f"{ts_name}.xtilt")
        
        return cls(
            tlt=tlt_data,
            rawtlt=rawtlt_data,
            xtilt=xtilt_data,
        )
    
    @staticmethod
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

    @classmethod
    def safe_read_tlt(cls, file: os.PathLike) -> np.ndarray | None:
        """Safely read an IMOD tlt file, returning None if it fails.
        
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
            return cls.read_tlt(file)
        except (FileNotFoundError, ValueError) as e:
            warnings.warn(f"Skipping {file}: {e}", RuntimeWarning)
            return None
    
    def __repr__(self) -> str:
        """Return string representation."""
        tlt_len = len(self.tlt) if self.tlt is not None else 0
        rawtlt_len = len(self.rawtlt) if self.rawtlt is not None else 0
        xtilt_len = len(self.xtilt) if self.xtilt is not None else 0
        return (f"TiltAngleData(tlt={tlt_len} angles, rawtlt={rawtlt_len} angles, "
                f"xtilt={xtilt_len} values)")
