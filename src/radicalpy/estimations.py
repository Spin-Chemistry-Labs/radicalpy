#!/usr/bin/env python

import numpy as np

from . import utils
from .data import constants, gamma_mT
from .simulation import HilbertSimulation


def Bhalf_theoretical(sim: HilbertSimulation) -> float:
    """Theoretical B1/2 for radical pairs in solution.

    Source: `Weller et al. Chem. Phys. Lett. 96, 1, 24-27 (1983)`_.

    Args:
            sim: The `sim` object containing the hyperfine coupling
                constants. (We'll change this to a list of molecules). It
                should contain exactly two molecules.

    Returns:
            float: The B1/2 value (mT).

    .. _Weller et al. Chem. Phys. Lett. 96, 1, 24-27 (1983):
       https://doi.org/10.1016/0009-2614(83)80109-2
    """
    assert len(sim.molecules) == 2
    sum_hfc2 = sum([m.effective_hyperfine**2 for m in sim.molecules])
    sum_hfc = sum([m.effective_hyperfine for m in sim.molecules])
    return np.sqrt(3) * (sum_hfc2 / sum_hfc)


def _relaxation_gtensor_term(g: list) -> float:
    return sum([(gi - np.mean(g)) ** 2 for gi in g])


def T1_relaxation_rate(g_tensors: list, B: float, tau_c: float) -> float:
    r"""T1 relaxation rate.

    Estimate T1 relaxation rate based on tau_c and g-tensor anisotropy.

    Source: `Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic Field Effects on Chemical and Biochemical Reactions (2004)`_.

    Args:
            g_tensors (list): The principle components of g-tensor.
            B (float): The external magnetic field strength (mT).
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The T1 relaxation rate (1/s)

    .. _Hayashi, Introduction to Dynamic Spin Chemistry\: Magnetic Field Effects on Chemical and Biochemical Reactions (2004):
       https://doi.org/10.1142/9789812562654_0015
    """
    hbar = constants.value("hbar")
    muB = constants.value("mu_B")
    omega = gamma_mT("E") * B
    g_innerproduct = _relaxation_gtensor_term(g_tensors)
    return (
        (1 / 5)
        * ((muB * B) / hbar) ** 2
        * g_innerproduct
        * (tau_c / (1 + omega**2 * tau_c**2))
    )


def T2_relaxation_rate(g_tensors: list, B: float, tau_c: float) -> float:
    """T2 relaxation rate.

    Estimate T2 relaxation rate based on tau_c and g-tensor anisotropy.

    Source: `Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic Field Effects on Chemical and Biochemical Reactions (2004)`_.

    Args:
            g_tensors (list): The principle components of g-tensor.
            B (float): The external magnetic field strength (mT).
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The T2 relaxation rate (1/s).
    """
    hbar = constants.value("hbar")
    muB = constants.value("mu_B")
    omega = gamma_mT("E") * B
    g_innerproduct = _relaxation_gtensor_term(g_tensors)
    return (
        (1 / 30)
        * ((muB * B) / hbar) ** 2
        * g_innerproduct
        * (4 * tau_c + (3 * tau_c / (1 + omega**2 * tau_c**2)))
    )


def aqueous_glycerol_viscosity(frac_glyc: float, temp: float) -> float:
    """Viscosity of aqueous glycerol solutions.

    Gives a good approximation for temperatures in the range 0-100°C.

    Source: `Volk et al. Experiments in Fluids, 59, 76, (2018)`_.

    Args:
            frac_glyc (float): The fraction of glycerol in solution (0.00-1.00).
            temp (float): The temperature in °C (0-100) (<0.07% accuracy between 15-30°C).

    Returns:
            float: The viscosity of the glycerol/water mixture in N s/m^2.

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


def correlation_time(*args: np.ndarray) -> float:
    """Estimate correlation time.

    The optimal parameters (amplitudes and taus) obtained from the multiexponential fit of the autocorrelation curve are used to estimate the correlation time.  See `radicalpy.utils.multiexponential_fit`.

    Args:
        args (np.ndarray): The amplitudes and taus from the
            multiexponential fit.

    Returns:
            float: The correlation time (s).
    """
    n = len(args) // 2
    As, taus = list(args)[:n], list(args)[n:]
    As_norm = As / np.array(As).sum()
    y = As_norm / taus
    return np.trapz(y, dx=1)


def dipolar_interaction_MC(
    r: float or np.ndarray, theta: float or np.ndarray
) -> (float or np.ndarray):
    """Dipolar interaction for Monte Carlo trajectories.

    Sources:

        - `O'Dea et al. J. Phys. Chem. A, 109, 5, 869-873 (2005)`_.
        - `Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544 (2015)`_.

    Args:
            r (float or np.ndarray): The interradical separation (m).
            theta (float or np.ndarray): The angle of molecular rotation (rad).

    Returns:
            (float or np.ndarray): The dipolar coupling constant in milli Tesla (mT).

    .. _O'Dea et al. J. Phys. Chem. A, 109, 5, 869-873 (2005):
       https://doi.org/10.1021/jp0456943
    .. _Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544 (2015):
       https://doi.org/10.1021/acs.jpca.5b02183
    """
    return dipolar_interaction_isotropic(r) * (3 * np.cos(theta) ** 2 - 1)


def dipolar_interaction_anisotropic(
    r: float or np.ndarray, gamma: float = gamma_mT("E")
) -> np.ndarray:
    """Anisotropic dipolar coupling.

    Point dipole approximation is used.

    Args:
            r (float or np.ndarray): The interradical separation (m).

    Returns:
            np.ndarray: The dipolar coupling tensor in millitesla (mT).
    .. todo:: np.ndarray not implemented.  `dipolar * diag` fails.
    """
    dipolar1d = dipolar_interaction_isotropic(r)
    dipolar = gamma * (2 / 3) * dipolar1d
    return dipolar * np.diag([-1, -1, 2])


def dipolar_interaction_isotropic(r: float or np.ndarray) -> (float or np.ndarray):
    """Isotropic dipolar coupling.

    Point dipole approximation is used.

    Source: `Santabarbara et al. Biochemistry, 44, 6, 2119–2128 (2005)`_.

    Args:
            r (float or np.ndarray): The interradical separation (m).

    Returns:
            (float or np.ndarray): The dipolar coupling constant in millitesla (mT).

    .. _Santabarbara et al. Biochemistry, 44, 6, 2119–2128 (2005):
       https://pubs.acs.org/doi/10.1021/bi048445d
    """
    mu_0 = constants.value("mu_0")
    mu_B = constants.value("mu_B")
    g_e = constants.value("g_e")

    conversion = (3 * -g_e * mu_B * mu_0) / (8 * np.pi)
    return (-conversion / r**3) * 1000


def exchange_interaction_in_protein(
    r: float or np.ndarray, beta: float = 14e9, J0: float = 9.7e9
) -> (float or np.ndarray):
    """Exchange interaction for radical pairs in proteins.

    Source: `Moser et al. Nature 355, 796–802 (1992)`_.

    Args:
            r (float or np.ndarray): The interradical separation (m).
            beta (float): The range parameter (1/m).
            J0 (float): The strength of the interaction (mT).

    Returns:
            (float or np.ndarray): The exchange interaction (mT).

    .. _Moser et al. Nature 355, 796–802 (1992):
       https://doi.org/10.1038/355796a0
    """
    return J0 * np.exp(-beta * r)


def exchange_interaction_in_solution(
    r: float or np.ndarray, beta: float = 0.049e-9, J0rad: float = 1.7e17
) -> (float or np.ndarray):
    """Exchange interaction for radical pairs in solution.

    Source: `McLauchlan et al. Mol. Phys. 73:2, 241-263 (1991)`_.

    Args:
            r (float or np.ndarray): The interradical separation (m).
            beta (float): The range parameter (m).
            J0rad (float): The strength of the interaction (rad/s).

    Returns:
            (float or np.ndarray): The exchange interaction (mT).

    .. _McLauchlan et al. Mol. Phys. 73:2, 241-263 (1991):
       https://doi.org/10.1080/00268979100101181
    """
    J0 = J0rad / gamma_mT("E")
    return J0 * np.exp(-r / beta)


def exchange_interaction_in_solution_MC(
    r: float or np.ndarray, beta: float = 2e10, J0: float = -570
) -> (float or np.ndarray):
    """Exchange interaction for Monte Carlo trajectories.

    Sources:

        - `O'Dea et al. J. Phys. Chem. A, 109, 5, 869-873 (2005)`_.
        - `Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544 (2015)`_.

    Args:
            r (float or np.ndarray): The interradical separation (m).
            beta (float): The range parameter (1/m).
            J0 (float): The strength of the interaction (mT).

    Returns:
            (float or np.ndarray): The exchange coupling constant in milli Tesla (mT).
    """
    return J0 * np.exp(-beta * (r - r.min()))


def g_tensor_relaxation_rate_constant(tau_c: float, g1: list, g2: list) -> float:
    """g-tensor relaxation rate.

    To be used with `radicalpy.relaxation.RandomFields`.

    Source: `Player et al. J. Chem. Phys. 153, 084303 (2020)`_.

    Args:
            tau_c (float): The rotational correlation time (s).
            g1 (list): The principle components of g-tensor of the first radical.
            g2 (list): The principle components of g-tensor of the second radical.

    Returns:
            float: The g-tensor relaxation rate (1/s).

    .. _Player et al. J. Chem. Phys. 153, 084303 (2020):
       https://doi.org/10.1063/5.0021643
    """
    ge = constants.value("g_e")
    g1sum = sum([(gi - ge) ** 2 for gi in g1])
    g2sum = sum([(gi - ge) ** 2 for gi in g2])
    return (g1sum + g2sum) / (9 * tau_c)


def k_D(D: np.ndarray, tau_c: float) -> float:
    """D (dipolar)-dephasing rate for trajectories.

    Source: `Kattnig et al. New J. Phys., 18, 063007 (2016)`_.

    Args:
            D (np.ndarray): The dipolar interaction trajectory (mT).
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The D-dephasing rate (1/s).
    """
    D_var_MHz = np.var(utils.mT_to_MHz(D))
    return tau_c * D_var_MHz * 4 * np.pi**2 * 1e12  # (1/s) D-modulation rate


def k_STD(J: np.ndarray, tau_c: float) -> float:
    """ST-dephasing rate for trajectories.

    Source: `Kattnig et al. New J. Phys., 18, 063007 (2016)`_.

    Args:
            J (np.ndarray): The exchange interaction trajectory (mT).
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: The ST-dephasing rate (1/s).

    .. _Kattnig et al. New J. Phys., 18, 063007 (2016):
       https://iopscience.iop.org/article/10.1088/1367-2630/18/6/063007
    """
    J_var_MHz = np.var(utils.mT_to_MHz(J))
    return 4 * tau_c * J_var_MHz * 4 * np.pi**2 * 1e12


def k_STD_microreactor(
    D: float, V: float, d: float = 5e-10, J0: float = 1e11, alpha: float = 2e10
) -> float:
    """ST-dephasing rate for radical pairs in microreactors.

    Source: `Shushin, Chem. Phys. Lett., 181, 2–3, 274-278 (1991)`_.

    Args:
            D (float): The mutual diffusion coefficient (m^2/s).
            V (float): The volume of the microreactor (e.g. micelle) (m^3).
            d (float): The distance of closest approach (m).
            J0 (float): The maximum exchange interaction (1/s).
            alpha (float): The characteristic length factor (1/m).

    Returns:
            float: The ST-dephasing rate (1/s).

    .. _Shushin, Chem. Phys. Lett., 181, 2–3, 274-278 (1991):
       https://doi.org/10.1016/0009-2614(91)90366-H
    """
    l = d + 1 / alpha * (np.log(2 * np.abs(J0) / (D * alpha**2)) + 1.15)
    l -= 1j * (np.pi / 2 * alpha) * J0
    return 4 * np.pi * D * np.real(l) / V


def k_ST_mixing(Bhalf: float) -> float:
    """Singlet-triplet mixing rate.

    Source: `Steiner et al. Chem. Rev. 89, 1, 51–147 (1989)`_.

    Args:
            Bhalf (float): The theoretical B1/2 value (mT).

    Returns:
            float: The ST-mixing rate (1/s).

    .. _Steiner et al. Chem. Rev. 89, 1, 51–147 (1989):
       https://doi.org/10.1021/cr00091a003
    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B") * 1e-3
    h = constants.value("h")
    return -g_e * mu_B * Bhalf / h


def k_triplet_relaxation(B0: float, tau_c: float, D: float, E: float) -> float:
    """Excited triplet state relaxation rate.

    Source: `Atkins et al. Mol. Phys., 27, 6 (1974)`_.

    Args:
            B0 (float): The external magnetic field (MHz).
            tau_c (float): The rotational correlation time (s).
            D (float): The zero field splitting (ZFS) parameter D (Hz).
            E (float): The zero field splitting (ZFS) parameter E (Hz).

    Returns:
            float: The excited triplet state relaxation rate (1/s).

    .. _Atkins et al. Mol. Phys., 27, 6 (1974):
       https://doi.org/10.1080/00268977400101361
    """
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


def rotational_correlation_time_for_molecule(
    radius: float, temp: float, eta: float = 0.89e-3
) -> float:
    """Rotational correlation time.

    Rotational correlation time is the average time it takes for a molecule (smaller than a protein) to rotate one radian. For proteins see `radicalpy.estimations.rotational_correlation_time_for_protein`.

    To calculate viscosity (eta) for glycerol-water mixtures see `radicalpy.estimations.aqueous_glycerol_viscosity`.

    Args:
            radius (float): The radius of a spherical molecule (m).
            temp (float): The temperature of the solution (K).
            eta (float): The viscosity of the solution (N s/m^2) (default: 0.89e-3 corresponds to water).

    Returns:
            float: The rotational correlation time (s).
    """
    k_B = constants.value("k_B")
    return (4 * np.pi * eta * radius**3) / (3 * k_B * temp)


def rotational_correlation_time_for_protein(
    Mr: float, temp: float, eta: float = 0.89e-3
) -> float:
    """Rotational correlation time.

    Rotational correlation time is the average time it takes for a protein to rotate one radian. For small molecules see `radicalpy.estimations.rotational_correlation_time_for_molecule`.

    To calculate viscosity (eta) for glycerol-water mixtures see `radicalpy.estimations.aqueous_glycerol_viscosity`.

    Source: `Cavanagh et al. Protein NMR Spectroscopy. Principles and Practice, Elsevier Academic Press (2007)`_.

    Args:
            Mr (float): The molecular weight of the protein (kDa).
            temp (float): The temperature of the solution (K).
            eta (float): The viscosity of the solution (N s/m^2) (default: 0.89e-3 corresponds to water).

    Returns:
            float: The rotational correlation time (s).

    .. _Cavanagh et al. Protein NMR  Spectroscopy. Principles and Practice, Elsevier Academic Press (2007):
       https://doi.org/10.1016/B978-0-12-164491-8.X5000-3
    """
    V = constants.value("V")
    rw = constants.value("rw")
    N_A = constants.value("N_A")

    # Calculate Rh - effective hydrodynamic radius of the protein in m
    Rh = ((3 * V * Mr) / (4 * np.pi * N_A)) ** 0.33 + rw
    return rotational_correlation_time_for_molecule(Rh, temp, eta)
