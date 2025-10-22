"""Main container data model for IMOD etomo alignment data."""

from pathlib import Path
from typing import Set

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .. import utils
from ..imod_utils import parse_edf



class EtomoDataFile(BaseModel):
    """Data model for IMOD etomo alignment data.
    
    This is the main container that holds all parsed etomo file data.
    
    Attributes:
        basename: Tilt series name (e.g., 'TS_001')
        tilt_series_extension: File extension (e.g., 'st', 'mrc')
        tilt_axis_angle: Tilt axis rotation angle (degrees)
        excluded_views: Set of view indices (1-indexed) to exclude
        n_images: Total number of images in the tilt series
        xf: Transformation parameters from .xf file, shape (n, 6)
        tlt: tilt angles from .tlt file
        rawtlt: Raw tilt angles from .rawtlt file
        xtilt: x-axis tilt information from .xtilt file
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    basename: str
    tilt_series_extension: str
    tilt_axis_angle: float | None = None
    excluded_views: Set[int] = Field(default_factory=set)
    n_images: int = 0
    xf: np.ndarray | None = None
    tlt: np.ndarray | None = None
    rawtlt: np.ndarray | None = None
    xtilt: np.ndarray | None = None

    @classmethod
    def from_directory(cls, directory: Path) -> "EtomoDataFile":
        """Load etomo alignment data from a directory.
        
        Parameters
        ----------
        directory : Path
            Directory containing etomo alignment files
            
        Returns
        -------
        EtomoDataFile
            Parsed etomo alignment data
        """
        
        edf_metadata = parse_edf(directory)
        
        basename = edf_metadata["basename"]
        
        tlt_data = utils.safe_read_tlt(directory / f"{basename}.tlt")
        rawtlt_data = utils.safe_read_tlt(directory / f"{basename}.rawtlt")
        xtilt_data = utils.safe_read_tlt(directory / f"{basename}.xtilt")
        xf_data = utils.safe_read_xf(directory / f"{basename}.xf")
        
        return cls(
            basename=basename,
            tilt_series_extension=edf_metadata["tilt_series_extension"],
            tilt_axis_angle=edf_metadata["tilt_axis_angle"],
            excluded_views=edf_metadata["excluded_views"],
            n_images=edf_metadata["n_images"],
            xf=xf_data,
            tlt=tlt_data,
            rawtlt=rawtlt_data,
            xtilt=xtilt_data,
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (f"EtomoDataFile(basename='{self.basename}', "
                f"n_images={self.n_images}, "
                f"has_tlt={self.tlt is not None}, "
                f"has_xf={self.xf is not None})")
