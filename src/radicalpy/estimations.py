#!/usr/bin/env python

import numpy as np

from . import utils
from .data import MOLECULE_DATA, constants, gamma_mT, multiplicity


def Bhalf_theoretical(sim):
    assert len(sim.molecules) == 2
    sum_hfc2 = sum([m.effective_hyperfine**2 for m in sim.molecules])
    sum_hfc = sum([m.effective_hyperfine for m in sim.molecules])
    return np.sqrt(3) * (sum_hfc2 / sum_hfc)


def correlation_time_from_fit(*args):
    n = len(args) // 2
    As, taus = list(args)[:n], list(args)[n:]
    As_norm = As / np.array(As).sum()
    y = As_norm / taus
    return As, taus, np.trapz(y, dx=1)
	

def dipolar_interaction_1d(r: float) -> float:
    """Construct the Dipolar interaction constant.

    Construct the Dipolar interaction based on the inter-radical separation `r`.

    .. todo::
        equation 4 of https://pubs.acs.org/doi/10.1021/bi048445d.

    Returns:
        float: The dipolar coupling constant in milli Tesla (mT).

    """
    mu_0 = constants.value("mu_0")
    mu_B = constants.value("mu_B")
    g_e = constants.value("g_e")

    conversion = (3 * -g_e * mu_B * mu_0) / (8 * np.pi)
    return (-conversion / r**3) * 1000


def dipolar_interaction_3d(r: float, gamma: float = gamma_mT("E")) -> float:
    # , coefficient: float):
    #         kwargs = {"coefficient": coefficient} if coefficient is not None else {}
    dipolar1d = dipolar_interaction_1d(r)  # , **kwargs)
    dipolar = gamma * (2 / 3) * dipolar1d
    return dipolar * np.diag([-1, -1, 2])
	

def dipolar_interaction_monte_carlo(r: float, theta: float) -> float:
    """Construct the Dipolar interaction constant for Monte Carlo simulation.

    Construct the Dipolar interaction based on the inter-radical separation `r`.

    Returns:
        float: The dipolar coupling constant in milli Tesla (mT).

    """
	
    return dipolar_interaction_1d(r) * (3 * np.cos(theta)**2 - 1)

def exchange_interaction_monte_carlo(r: float) -> float:
    """Construct the exchange interaction constant for Monte Carlo simulation.

    .. todo::
        Write proper docs.
    """
    J0 = -570e-3
    alpha = 2e10
    return (-J0 * np.exp(-alpha * (r))) * 1e3
	

def exchange_interaction_protein(
    r: float, beta: float = 1.4e10, J0: float = 9.7e12
) -> float:
    """Construct the exchange interaction constant in a protein.

    .. todo::
        Write proper docs.
    """
    return J0 * np.exp(-beta * r) / 1000


def exchange_interaction_solution(r: float) -> float:
    """Construct the exchange interaction constant in a solution.

    .. todo::
        Write proper docs.
    """
    J0rad = 1.7e17
    rj = 0.049e-9
    gamma = 1.76e8  # TODO
    J0 = J0rad / gamma / 10  # convert to mT?????????
    return J0 * np.exp(-r / rj)


def exchange_interaction(r: float, model: str = "solution"):
    """Construct the exchange interaction constant in a solution.

    .. todo::
        Write proper docs.
    """
    methods = {
        "solution": exchange_interaction_solution,
        "protein": exchange_interaction_protein,
    }
    return methods[model](r)


def g_tensor_relaxation_rate_constant(tau_c, g1, g2):
    ge = constants.value("g_e")
    g1sum = sum([(gi - ge) ** 2 for gi in g1])
    g2sum = sum([(gi - ge) ** 2 for gi in g2])
    return (g1sum + g2sum) / (9 * tau_c)
	

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
    jnu0tc = (2 / 15) * (
        (4 * tau_c) / (1 + 4 * nu_0**2 * tau_c**2)
        + (tau_c) / (1 + nu_0**2 * tau_c**2)
    )
    return (D**2 + 3 * E**2) * jnu0tc


def rotational_correlation_time(radius, temp, eta=0.89e-3):
    k_B = constants.value("k_B")

    # Calculate isotropic rotational correlation time (tau_c) in s
    tau_c = (4 * np.pi * eta * radius**3) / (3 * k_B * temp)
    return tau_c


def rotational_correlation_time_protein(Mr, temp, eta=0.89e-3):
    V = constants.value("V")
    rw = constants.value("rw")
    N_A = constants.value("N_A")
    # k_B = constants.value("k_B")

    # Calculate Rh - effective hydrodynamic radius of the protein in m
    Rh = ((3 * V * Mr) / (4 * np.pi * N_A)) ** 0.33 + rw

    # Calculate isotropic rotational correlation time (tau_c) in s
    tau_c = rotational_correlation_time(Rh, temp, eta)
    # tau_c = (4 * np.pi * eta * Rh**3) / (3 * k_B * temp)
    return tau_c