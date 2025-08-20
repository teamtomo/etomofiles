"""etomo metadata handling into dataframes"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("etomofiles")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Davide Torre"
__email__ = "davidetorre99@gmail.com"
