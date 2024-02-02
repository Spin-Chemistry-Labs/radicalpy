"""Radicalpy package root."""
import logging

from pint import UnitRegistry

logger = logging.getLogger(__name__)
print(f"{logger.name=}")

# pylint: disable=unused-import wildcard-import
# from . import (classical, data, estimations, kinetics, plot,  # noqa: E402 F401 F403
# relaxation, shared, simulation, utils)

ureg = UnitRegistry()
Q_ = ureg.Quantity

print(f"radicalpy.__init__ : {__name__}")
