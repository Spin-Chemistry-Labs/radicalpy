"""Radicalpy package root."""

from pint import UnitRegistry

from . import plot  # noqa: F401 F403
from . import (
    classical,
    data,
    estimations,
    kinetics,
    relaxation,
    shared,
    simulation,
    utils,
)

# pylint: disable=unused-import wildcard-import


ureg = UnitRegistry()
Q_ = ureg.Quantity
