#!/usr/bin/env python

import numpy as np

from .data import constants


def spherical_to_cartesian(theta, phi):
    return np.array(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )


def spin_quantum_number(multiplicity: int) -> float:
    return float(multiplicity - 1) / 2.0


def mT_to_MHz(mT: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return mT * 1e-9 * g_e * mu_B / h


def mT_to_angular_frequency(mT: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return mT * mu_B / hbar * g_e / 1e9