#!/usr/bin/env python

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from types import SimpleNamespace

from radicalpy import Q_, ureg
from radicalpy.data import SPIN_DATA, gamma_mT


class Constant(float):
    def __new__(cls, some_arg=None):
        obj = float.__new__(cls, 10)
        obj._some_arg = some_arg
        return obj


const = Constant(some_arg="Hi")
print(f"{const=}")
print(f"{100.0 + const=}")
print(f"{const._some_arg=}")


class Isotope(str):
    def __new__(cls, o):
        obj = super().__new__(cls, o)
        data = SPIN_DATA[o]
        obj.gamma = data["gamma"]
        obj.multiplicity = data["multiplicity"]
        return obj


print(SPIN_DATA["E"])
print(gamma_mT("E"))
e = Isotope("E")

print(f"{e=}")
print(f"{e.gamma=}")
print(f"{e.multiplicity=}")
