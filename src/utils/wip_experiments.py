#!/usr/bin/env python

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from types import SimpleNamespace

from radicalpy import Q_, ureg
from radicalpy.data import (
    CONSTANTS_JSON,
    SPIN_DATA,
    SPIN_DATA_JSON,
    Isotope,
    constants,
    isotopes,
)

with open(CONSTANTS_JSON) as f:
    CONSTANTS_DATA = json.load(f)


class Constant(Q_):
    def __new__(cls, *args):
        # By default, `pint.Quantity` aka `Q_` has `len(args) == 2`,
        # i.e. value and unit, but a `Constant` we want to init with
        # the single `dict` which contains all of that + other details
        # which we want to save
        if len(args) == 1:
            data = args[0]
            value = data.pop("value")
            unit = ureg(data.pop("unit"))
            obj = super().__new__(cls, value, unit)
            obj.details = SimpleNamespace(**data)
        else:
            obj = super().__new__(cls, *args)
        return obj


hbar_data = CONSTANTS_DATA["hbar"]
c_data = CONSTANTS_DATA["c"]
print(f"{hbar_data=}")
# hbar_value = hbar_data.pop("value")
# const = Constant(hbar_value, hbar_data)
hbar = Constant(hbar_data)
print(f"{type(hbar)=}")
print(f"{hbar=}")
print(f"{100.0 * hbar=}")
print(f"{100.0 * ureg('J s') + hbar=}")
print(f"{hbar.details=}")

c = Constant(c_data)
print(f"{type(c)=}")
print(f"{c=}")
print(f"{c.details=}")
print(f"{hbar * c=}")
prd = hbar * c
print(f"{prd=}")
print(f"{type(prd)=}")
# print(f"{prd.details=}") # THIS BREAKS!

print("=" * 80)

print(constants.hbar)
print(isotopes.E)
print(isotopes)

print("DONE!")
