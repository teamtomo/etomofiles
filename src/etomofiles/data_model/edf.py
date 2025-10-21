"""EDF metadata data model for IMOD etomo files."""

from pathlib import Path
from typing import Set

import mrcfile
from pydantic import BaseModel, Field


class EdfData(BaseModel):
    """Data model for IMOD .edf (etomo directive file) metadata.
    
    Attributes:
        ts_name: Tilt series dataset name
        ts_ext: Tilt series file extension (e.g., 'st', 'mrc')
        tilt_axis_angle: Rotation angle of tilt axis (degrees)
        excluded_views: Set of view indices (1-indexed) to exclude from reconstruction
        n_images: Total number of images in the tilt series
    """
    ts_name: str
    ts_ext: str
    tilt_axis_angle: float | None = None
    excluded_views: Set[int] = Field(default_factory=set)
    n_images: int = 0

    @classmethod
    def from_string(cls, edf_text: str, directory: Path) -> "EdfData":
        """Parse .edf file content.
        
        Parameters
        ----------
        edf_text : str
            Content of the .edf file
        directory : Path
            Directory containing the tilt series (needed to read image count)
            
        Returns
        -------
        EdfData
            Parsed EDF metadata
            
        Raises
        ------
        ValueError
            If required fields missing or cannot determine image count
        FileNotFoundError
            If tilt series file not found
        """
        ts_name = None
        ts_ext = None
        tilt_axis_angle = None
        excluded_views = set()
        
        # Parse .edf content
        for line in edf_text.splitlines():
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
        
        return cls(
            ts_name=ts_name,
            ts_ext=ts_ext,
            tilt_axis_angle=tilt_axis_angle,
            excluded_views=excluded_views,
            n_images=n_images,
        )

    @classmethod
    def from_file(cls, edf_file: Path) -> "EdfData":
        """Load EdfData from a .edf file.
        
        Parameters
        ----------
        edf_file : Path
            Path to the .edf file
            
        Returns
        -------
        EdfData
            Parsed EDF metadata
            
        Raises
        ------
        FileNotFoundError
            If .edf file or tilt series not found
        ValueError
            If .edf file is malformed
        """
        directory = edf_file.parent
        
        try:
            with open(edf_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            with open(edf_file, 'r', encoding='latin-1') as f:
                content = f.read()
        
        return cls.from_string(content, directory)
    
    @classmethod
    def from_directory(cls, directory: Path) -> "EdfData":
        """Parse .edf file from a directory containing etomo alignment files.
        
        Parameters
        ----------
        directory : Path
            Directory containing the .edf file and tilt series
            
        Returns
        -------
        EdfData
            Parsed EDF metadata
            
        Raises
        ------
        FileNotFoundError
            If no .edf file found or tilt series file not found
        ValueError
            If required fields missing or cannot determine image count
        """
        edf_files = list(directory.glob("*.edf"))
        if not edf_files:
            raise FileNotFoundError("No .edf file found")
        
        return cls.from_file(edf_files[0])
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (f"EdfData(ts_name='{self.ts_name}', ts_ext='{self.ts_ext}', "
                f"tilt_axis_angle={self.tilt_axis_angle}, n_images={self.n_images}, "
                f"excluded={len(self.excluded_views)} views)")
