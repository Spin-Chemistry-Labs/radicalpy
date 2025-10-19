"""Radicalpy package root."""

from pint import UnitRegistry

from . import plot  # noqa: F401 F403
from . import (
    classical,
    data,
    estimations,
    experiments,
    kinetics,
    relaxation,
    shared,
    simulation,
    tensornetwork,
    utils,
)

# pylint: disable=unused-import wildcard-import

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Fallback for Python < 3.8 if importlib-metadata backport is installed
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("radicalpy")
except PackageNotFoundError:
    # Handle cases where the package is not installed or metadata is missing
    __version__ = "unknown"

ureg = UnitRegistry()
Q_ = ureg.Quantity
