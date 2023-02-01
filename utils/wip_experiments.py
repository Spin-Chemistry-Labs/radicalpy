#!/usr/bin/env python

# HFC experiments
import sys

sys.path.insert(0, "..")  ##############################################

from functools import singledispatchmethod  # noqa E402
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from radicalpy import data  # noqa E402


class Hfc:
    """The Hfc class represents isotropic and anisotropic HFC values.

    Args:
        hfc (float | list[list[float]]): The HFC value.  In case of a
            single `float`, only the isotropic value is set.  In case
            of a 3x3 matrix both the isotropic and anisotropic values
            are stored.
    """

    _anisotropic: Optional[NDArray]
    """Optional anisotropic HFC value."""

    isotropic: float
    """Isotropic HFC value."""

    def __repr__(self):  # noqa D105
        available = "not " if self._anisotropic is None else ""
        return f"{self.isotropic:.4} <anisotropic {available}available>"

    @singledispatchmethod
    def __init__(self, hfc: list[list[float]]):
        """Construct anisotropic `Hfc`."""
        self._anisotropic = np.array(hfc)
        if self._anisotropic.shape != (3, 3):
            raise ValueError("Anisotropic HFCs should be a 3x3 matrix or a float!")
        self.isotropic = self._anisotropic.trace() / 3

    @__init__.register
    def _(self, hfc: float):
        """Construct isotropic only `Hfc`."""
        self._anisotropic = None
        self.isotropic = hfc

    @property
    def anisotropic(self) -> NDArray:
        """Anisotropic value if available.

        Returns:
            NDarray: The anisotropic HFC values.
        """
        if self._anisotropic is None:
            raise ValueError("The molecule doesn't support anisotropic HFCs")
        return self._anisotropic


# with(open(DATA_DIR/"molecules/flavin_anion.json") as f:
#      flavin_dict = json.load(f)
flavin_dict = data.MOLECULE_DATA["flavin_anion"]
hfc_3d_data = flavin_dict["data"]["N5"]["hfc"]
hfc_3d_obj = Hfc(hfc_3d_data)
print(f"{hfc_3d_obj=}")
print(f"{hfc_3d_obj.isotropic=}")
print(f"{hfc_3d_obj.anisotropic=}")

adenine_dict = data.MOLECULE_DATA["adenine_cation"]
hfc_1d_data = adenine_dict["data"]["N6-H1"]["hfc"]
hfc_1d_obj = Hfc(hfc_1d_data)
print(f"{hfc_1d_obj=}")
print(f"{hfc_1d_obj.isotropic=}")
print(f"{hfc_1d_obj.anisotropic=}")
