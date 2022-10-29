#!/usr/bin/env python

import numpy as np

from . import utils
from .data import constants


def rotational_correlation_time(radius, temp, eta=0.89e-3):
    k_B = constants.value("k_B")

    # Calculate isotropic rotational correlation time (tau_c) in s
    tau_c = (4 * np.pi * eta * radius**3) / (3 * k_B * temp)
    return tau_c


def rotational_correlation_time_protein(Mr, temp, eta=0.89e-3):
    V = constants.value("V")
    rw = constants.value("rw")
    N_A = constants.value("N_A")
    #k_B = constants.value("k_B")

    # Calculate Rh - effective hydrodynamic radius of the protein in m
    Rh = ((3 * V * Mr) / (4 * np.pi * N_A)) ** 0.33 + rw

    # Calculate isotropic rotational correlation time (tau_c) in s
    tau_c = rotational_correlation_time(Rh, temp, eta)
    #tau_c = (4 * np.pi * eta * Rh**3) / (3 * k_B * temp)
    return tau_c


def k_STD(J, tau_c):
    # J-modulation rate
    J_var_MHz = utils.mT_to_MHz(utils.mT_to_MHz(np.var(J)))
    return 4 * tau_c * J_var_MHz * 4 * np.pi**2 * 1e12
	

def k_STD_micelle(D, V, d=5e-10, J0=1e11, alpha=2e10):
    l = d + 1 / alpha * (np.log(2 * np.abs(J0) / (D * alpha**2)) + 1.15) 
    l -= 1j * (np.pi / 2 * alpha) * J0
    return 4 * np.pi * D * np.real(l) / V


def k_D(D, tau_c):
    # D-modulation rate
    D_var_MHz = utils.mT_to_MHz(utils.mT_to_MHz(np.var(D)))
    return tau_c * D_var_MHz * 4 * np.pi**2 * 1e12  # (s^-1) D-modulation rate


def k_ST_mixing(Bhalf: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B") * 1e-3
    h = constants.value("h")
    return -g_e * mu_B * Bhalf / h


def k_triplet_relaxation(B0, tau_c, D, E):
    g = constants.value("g_e")
    muB = constants.value("mu_B") * 1e-3
    h = constants.value("h")
    B0 = utils.mT_to_MHz(B0)
    
    nu_0 = (g * muB * B0) / h
    jnu0tc = (2 / 15) * ((4 * tau_c) / (1 + 4 * nu_0**2 * tau_c**2) + (tau_c) / (1 + nu_0**2 * tau_c**2))
    return (D**2 + 3 * E**2) * jnu0tc