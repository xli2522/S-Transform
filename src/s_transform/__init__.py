"""Public API for the S-transform package."""

from importlib.metadata import PackageNotFoundError, version

from .core import STransform, inverse_s_transform, s_transform

__all__ = ["STransform", "s_transform", "inverse_s_transform"]

try:
    __version__ = version("s-transform")
except PackageNotFoundError:
    # Source checkouts are importable through ``src`` before installation.
    __version__ = "0.1.0"
