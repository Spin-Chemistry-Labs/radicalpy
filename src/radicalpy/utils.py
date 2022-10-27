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
    return 1e-9 * g_e * mu_B / h


def rotational_correlation_time_protein(Mr, temp, eta=0.89e-3):
    V = constants.value("V")
    rw = constants.value("rw")
    N_A = constants.value("N_A")
    k_B = constants.value("k_B")

    # Calculate Rh - effective hydrodynamic radius of the protein in m
    Rh = ((3 * V * Mr) / (4 * np.pi * N_A)) ** 0.33 + rw

    # Calculate isotropic rotational correlation time (tau_c) in s
    tau_c = (4 * np.pi * eta * Rh**3) / (3 * k_B * temp)
    return tau_c
