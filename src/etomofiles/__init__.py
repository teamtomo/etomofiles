"""A Python package for reading IMOD etomo alignment files into pandas DataFrames.

This package provides utilities to read IMOD etomo alignment metadata
and expose it as pandas DataFrames.
"""

from importlib.metadata import PackageNotFoundError, version

from .data_model.tlt import TiltAngleData
from .data_model.transform import TransformData
from .io import read
from .imod_utils import xf_to_array

try:
    __version__ = version("etomofiles")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "read",
    "xf_to_array",
    "__version__",
]

# Expose low-level file readers for advanced users
read_tlt = TiltAngleData.read_tlt
read_xf = TransformData.read_xf
safe_read_tlt = TiltAngleData.safe_read_tlt
safe_read_xf = TransformData.safe_read_xf

__all__.extend(["read_tlt", "read_xf", "safe_read_tlt", "safe_read_xf"])
