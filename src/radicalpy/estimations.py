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
    """Construct the dipolar interaction constant for Monte Carlo simulation.

    Construct the Dipolar interaction based on the inter-radical separation `r`.

    Source: `Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544
    (2015)`_.

    Returns:
            float: The dipolar coupling constant in milli Tesla (mT).

    .. _Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544 (2015):
       https://doi.org/10.1021/acs.jpca.5b02183
    """
    return dipolar_interaction_1d(r) * (3 * np.cos(theta) ** 2 - 1)


def exchange_interaction_monte_carlo(r: float) -> float:
    """Construct the exchange interaction constant for Monte Carlo simulation.

    Source: `Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544
    (2015)`_.

    .. todo::
        Write proper docs.
    """
    J0 = 570e-3
    alpha = 2e10
    return (-J0 * np.exp(-alpha * (r - r.min()))) * 1e3


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


def k_STD(J: np.ndarray, tau_c: float) -> float:
    """ST-dephasing rate estimation for trajectories.

    Source: `Kattnig et al. New J Phys, 18, 063007 (2016)`_.

    Args:
            J (np.ndarray): The exchange interaction trajectory.
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The ST-dephasing rate (s^-1).

    .. _Kattnig et al. New J Phys, 18, 063007 (2016):
       https://iopscience.iop.org/article/10.1088/1367-2630/18/6/063007
    """
    J_var_MHz = utils.mT_to_MHz(utils.mT_to_MHz(np.var(J)))
    return 4 * tau_c * J_var_MHz * 4 * np.pi**2 * 1e12


def k_STD_microreactor(D: float, V: float, d=5e-10, J0=1e11, alpha=2e10) -> float:
    """ST-dephasing rate estimation for radical pairs in microreactors.

    Source: `Shushin, Chem. Phys. Letts., 181, 2–3, 274-278 (1991)`_.

    Args:
            D (float): The mutual diffusion coefficient (m^2 s^-1).
            V (float): The volume of the microreactor (e.g. micelle) (m^3).
            d (float): The distance of closest approach (m).
            J0 (float): The exchange factor (s^-1).
            alpha (float): The factor alpha (m^-1).


    Returns:
            float: The ST-dephasing rate (s^-1).

    .. _Shushin, Chem. Phys. Letts., 181, 2–3, 274-278 (1991):
       https://doi.org/10.1016/0009-2614(91)90366-H
    """
    l = d + 1 / alpha * (np.log(2 * np.abs(J0) / (D * alpha**2)) + 1.15)
    l -= 1j * (np.pi / 2 * alpha) * J0
    return 4 * np.pi * D * np.real(l) / V


def k_D(D: np.ndarray, tau_c: float) -> float:
    """D (dipolar)-dephasing rate estimation for trajectories.

    Source: `Kattnig et al. New J Phys, 18, 063007 (2016)`_.

    Args:
            D (np.ndarray): The dipolar interaction trajectory.
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The D-dephasing rate (s^-1).

    .. _Kattnig et al. New J Phys, 18, 063007 (2016):
       https://iopscience.iop.org/article/10.1088/1367-2630/18/6/063007
    """
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


def T1T2_relaxation_gtensor_term(g: list) -> float:
    return sum([(gi - np.mean(g)) ** 2 for gi in g])


def T1_relaxation_rate_gtensor(g_tensors: list, B: float, tau_c: float) -> float:
    """g-tensor anisotropy induced T1 relaxation rate estimation.

    Source: `Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic Field Effects on Chemical and Biochemical Reactions (2004)`_.

    Args:
            g_tensors (list): The principle components of g-tensor.
            B (float): The external magnetic field strength (mT).
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The T1 relaxation rate (s^-1)

    .. _Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic Field Effects on Chemical and Biochemical Reactions (2004):
       https://doi.org/10.1142/9789812562654_0015
    """
    hbar = rp.data.constants.value("hbar")
    muB = rp.data.constants.value("mu_B")
    omega = rp.data.gamma_mT("E") * B
    g_innerproduct = T1T2_relaxation_gtensor_term(g_tensors)
    return (
        (1 / 5)
        * ((muB * B) / hbar) ** 2
        * g_innerproduct
        * (tau_c / (1 + omega**2 * tau_c**2))
    )


def T2_relaxation_rate_gtensor(g_tensors, B, tau_c):
    """g-tensor anisotropy induced T2 relaxation rate estimation.

    Source: `Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic Field Effects on Chemical and Biochemical Reactions (2004)`_.

    Args:
            g_tensors (list): The principle components of g-tensor.
            B (float): The external magnetic field strength (mT).
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The T2 relaxation rate (s^-1).

    .. _Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic Field Effects on Chemical and Biochemical Reactions (2004):
       https://doi.org/10.1142/9789812562654_0015
    """
    hbar = rp.data.constants.value("hbar")
    muB = rp.data.constants.value("mu_B")
    omega = rp.data.gamma_mT("E") * B
    g_innerproduct = T1T2_relaxation_gtensor_term(g_tensors)
    return (
        (1 / 30)
        * ((muB * B) / hbar) ** 2
        * g_innerproduct
        * (4 * tau_c + (3 * tau_c / (1 + omega**2 * tau_c**2)))
    )


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


def viscosity_glycerol_mixture(frac_glyc: float, temp: float) -> float:
    """Calculates viscosity of aqueous glycerol solutions.
    Gives a good approximation for temperatures in the range 0-100°C.

    Source: `Volk et al. Experiments in Fluids, 59, 76, (2018)`_.

    Args:
            frac_glyc (float): The fraction of glycerol in solution (0.00-1.00).
            temp (float): The temperature in °C (0-100) (<0.07% accuracy between 15-30°C).

    Returns:
            float: The viscosity of the glycerol/water mixture in N s m^-2.

    .. _Volk et al. Experiments in Fluids, 59, 76, (2018):
       https://doi.org/10.1007/s00348-018-2527-y
    """
    vol_glyc = frac_glyc
    vol_water = 1 - frac_glyc
    density_glyc = 1273.3 - 0.6121 * temp
    density_water = 1000 * (1 - ((np.abs(temp - 3.98)) / 615) ** 1.71)

    mass_glyc = density_glyc * vol_glyc
    mass_water = density_water * vol_water
    tot_mass = mass_glyc + mass_water
    mass_frac = mass_glyc / tot_mass

    viscosity_glyc = 0.001 * 12100 * np.exp((-1233 + temp) * temp / (9900 + 70 * temp))
    viscosity_water = (
        0.001 * 1.790 * np.exp((-1230 - temp) * temp / (36100 + 360 * temp))
    )

    a = 0.705 - 0.0017 * temp
    b = (4.9 + 0.036 * temp) * a**2.5
    alpha = (
        1
        - mass_frac
        + (a * b * mass_frac * (1 - mass_frac)) / (a * mass_frac + b * (1 - mass_frac))
    )
    A = np.log(viscosity_water / viscosity_glyc)
    return viscosity_glyc * np.exp(A * alpha)
