"""Radicalpy package root."""
from pint import UnitRegistry

from . import (  # noqa: F401 F403
    classical,
    data,
    estimations,
    kinetics,
    plot,
    relaxation,
    shared,
    simulation,
    utils,
)

# pylint: disable=unused-import wildcard-import


ureg = UnitRegistry()
Q_ = ureg.Quantity
