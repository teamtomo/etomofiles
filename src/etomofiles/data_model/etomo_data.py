"""Main container data model for IMOD etomo alignment data."""

from pathlib import Path

from pydantic import BaseModel, Field

from .edf import EdfData
from .tlt import TiltAngleData
from .transform import TransformData


class EtomoData(BaseModel):
    """Data model for IMOD etomo alignment data.
    
    This is the main container that holds all parsed etomo file data including
    metadata from the .edf file, tilt angles from .tlt/.rawtlt/.xtilt files,
    and transformations from .xf files.
    
    Attributes:
        edf_metadata: Parsed .edf file metadata (dataset name, tilt axis, exclusions, etc.)
        tilt_angles: Tilt angle data from .tlt, .rawtlt, and .xtilt files
        transforms: Transformation data from .xf file
    """
    edf_metadata: EdfData
    tilt_angles: TiltAngleData
    transforms: TransformData

    @classmethod
    def from_directory(cls, directory: Path) -> "EtomoData":
        """Load etomo alignment data from a directory.
        
        Parameters
        ----------
        directory : Path
            Directory containing etomo alignment files (.edf, .tlt, .xf, etc.)
            
        Returns
        -------
        EtomoData
            Parsed etomo alignment data
            
        Raises
        ------
        FileNotFoundError
            If required files (.edf, tilt series) are not found
        ValueError
            If .edf file is malformed or missing required fields
        
        Examples
        --------
        >>> from pathlib import Path
        >>> etomo_data = EtomoData.from_directory(Path("TS_001/"))
        >>> etomo_data.edf_metadata.n_images
        41
        >>> etomo_data.tilt_angles.tlt
        array([-60., -58., ...])
        """
        # Parse .edf file first to get metadata
        edf_metadata = EdfData.from_directory(directory)
        
        # Load tilt angle files using the ts_name from .edf
        tilt_angles = TiltAngleData.from_directory(directory, edf_metadata.ts_name)
        
        # Load transformation file
        transforms = TransformData.from_directory(directory, edf_metadata.ts_name)
        
        return cls(
            edf_metadata=edf_metadata,
            tilt_angles=tilt_angles,
            transforms=transforms,
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (f"EtomoData(ts_name='{self.edf_metadata.ts_name}', "
                f"n_images={self.edf_metadata.n_images}, "
                f"has_tlt={self.tilt_angles.tlt is not None}, "
                f"has_transforms={self.transforms.transforms is not None})")
