"""A Python package for reading IMOD etomo alignment files into pandas DataFrames."""

from importlib.metadata import PackageNotFoundError, version

from .imod_utils import df_to_xf
from .io import read
from .utils import read_tlt, read_xf

try:
    __version__ = version("etomofiles")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "__version__",
    "df_to_xf",
    "read",
    "read_tlt",
    "read_xf",
]
