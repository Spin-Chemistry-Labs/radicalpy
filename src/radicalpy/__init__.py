from pint import UnitRegistry

from . import (
    classical,
    data,
    estimations,
    kinetics,
    plot,
    relaxation,
    simulation,
    utils,
)

ureg = UnitRegistry()
Q_ = ureg.Quantity
