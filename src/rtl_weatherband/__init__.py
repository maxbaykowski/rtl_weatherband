"""NOAA Weather Radio streaming from csdr_server to Icecast."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("rtl_weatherband")
except PackageNotFoundError:
    __version__ = "0.0.0"
