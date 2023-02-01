#!/usr/bin/env python

import numpy as np

from . import utils
from .data import Isotope
from .shared import constants as C
from .simulation import HilbertSimulation


def Bhalf_theoretical(sim: HilbertSimulation) -> float:
    """Theoretical B1/2 for radical pairs in solution.

    Source: `Weller et al. Chem. Phys. Lett. 96, 1, 24-27 (1983)`_.

    Args:
            sim: The `sim` object containing the hyperfine coupling
                constants. It should contain exactly two molecules.

    Returns:
            float: The B1/2 value (mT).

    .. _Weller et al. Chem. Phys. Lett. 96, 1, 24-27 (1983):
       https://doi.org/10.1016/0009-2614(83)80109-2
    .. todo:: Change `sim` to a list of molecules.
    """
    assert len(sim.molecules) == 2
    sum_hfc2 = sum(m.effective_hyperfine**2 for m in sim.molecules)
    sum_hfc = sum(m.effective_hyperfine for m in sim.molecules)
    return np.sqrt(3) * (sum_hfc2 / sum_hfc)


def _relaxation_gtensor_term(g: list) -> float:
    return sum((gi - np.mean(g)) ** 2 for gi in g)


def T1_relaxation_rate(
    g_tensors: list, B: float | np.ndarray, tau_c: float | np.ndarray
) -> float | np.ndarray:
    r"""T1 relaxation rate.

    Estimate T1 relaxation rate based on tau_c and g-tensor anisotropy.

    Source: `Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic
    Field Effects on Chemical and Biochemical Reactions (2004)`_.

    Args:
            g_tensors (list): The principle components of g-tensor.
            B (float or np.ndarray): The external magnetic field strength (T).
            tau_c (float or np.ndarray): The rotational correlation time (s).

    Returns:
            float or np.ndarray: The T1 relaxation rate (1/s)

    .. _Hayashi, Introduction to Dynamic Spin Chemistry\: Magnetic
       Field Effects on Chemical and Biochemical Reactions (2004):
       https://doi.org/10.1142/9789812562654_0015

    """
    omega = Isotope("E").gamma_mT * B
    g_innerproduct = _relaxation_gtensor_term(g_tensors)
    return (
        (1 / 5)
        * ((C.mu_B * B) / C.hbar) ** 2
        * g_innerproduct
        * (tau_c / (1 + omega**2 * tau_c**2))
    )


def T2_relaxation_rate(
    g_tensors: list, B: float | np.ndarray, tau_c: float | np.ndarray
) -> float | np.ndarray:
    """T2 relaxation rate.

    Estimate T2 relaxation rate based on tau_c and g-tensor anisotropy.

    Source: `Hayashi, Introduction to Dynamic Spin Chemistry: Magnetic
    Field Effects on Chemical and Biochemical Reactions (2004)`_.

    Args:
            g_tensors (list): The principle components of g-tensor.
            B (float or np.ndarray): The external magnetic field strength (T).
            tau_c (float or np.ndarray): The rotational correlation time (s).

    Returns:
            float or np.ndarray: The T2 relaxation rate (1/s).
    """
    omega = Isotope("E").gamma_mT * B
    g_innerproduct = _relaxation_gtensor_term(g_tensors)
    return (
        (1 / 30)
        * ((C.mu_B * B) / C.hbar) ** 2
        * g_innerproduct
        * (4 * tau_c + (3 * tau_c / (1 + omega**2 * tau_c**2)))
    )


def aqueous_glycerol_viscosity(
    frac_glyc: float | np.ndarray, temp: float
) -> float | np.ndarray:
    """Viscosity of aqueous glycerol solutions.

    Gives a good approximation for temperatures in the range 0-100°C.

    Source: `Volk et al. Experiments in Fluids, 59, 76, (2018)`_.

    Args:
            frac_glyc (float or np.ndarray): The fraction of glycerol
                in solution (0.00-1.00).
            temp (float): The temperature in °C (0-100) (<0.07%
                accuracy between 15-30°C).

    Returns:
            float or np.ndarray: The viscosity of the glycerol/water mixture in N s/m^2.

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

    The optimal parameters (amplitudes and taus) obtained from the
    multiexponential fit of the autocorrelation curve are used to
    estimate the correlation time.  See
    `radicalpy.utils.multiexponential_fit`.

    Args:
        args (np.ndarray): The amplitudes and taus from the
            multiexponential fit.

    Returns:
            float: The correlation time (s).
    .. todo:: Change to different approach.
    """
    n = len(args) // 2
    As, taus = list(args)[:n], list(args)[n:]
    As_norm = As / np.array(As).sum()
    y = As_norm / taus
    return np.trapz(y, dx=1)


def diffusion_coefficient(radius: float, temperature: float, eta: float):
    """Diffusion coefficient.

    The Stokes-Einstein relation.

    Source: `Einstein, Ann. der Physik, 17, 549-560 (1905)`_.

    Args:
            radius (float): The radius of the molecule (m).
            temperature (float): The temperature of the solution (K).
            eta (float): The viscosity of the solution (kg/m/s).

    Returns:
            float: The diffusion coefficient (m^2/s).

    .. _Einstein, Ann. der Physik, 17, 549-560 (1905):
       https://doi.org/10.1002/andp.19053220806
    """
    return (C.k_B * temperature) / (6 * np.pi * eta * radius)


def dipolar_interaction_MC(
    r: float | np.ndarray, theta: float | np.ndarray
) -> float | np.ndarray:
    """Dipolar interaction for Monte Carlo trajectories.

    Sources:

        - `O'Dea et al. J. Phys. Chem. A, 109, 5, 869-873 (2005)`_.
        - `Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544 (2015)`_.

    Args:
            r (float | np.ndarray): The interradical separation (m).
            theta (float | np.ndarray): The angle of molecular
                rotation (rad).

    Returns:
            float | np.ndarray: The dipolar coupling constant in
            milli Tesla (mT).

    .. _O'Dea et al. J. Phys. Chem. A, 109, 5, 869-873 (2005):
       https://doi.org/10.1021/jp0456943
    .. _Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544 (2015):
       https://doi.org/10.1021/acs.jpca.5b02183
    """
    return dipolar_interaction_isotropic(r) * (3 * np.cos(theta) ** 2 - 1)


def dipolar_interaction_anisotropic(
    r: float | np.ndarray, gamma: float = Isotope("E").gamma_mT
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


def dipolar_interaction_isotropic(r: float | np.ndarray) -> float | np.ndarray:
    """Isotropic dipolar coupling.

    Point dipole approximation is used.

    Source: `Santabarbara et al. Biochemistry, 44, 6, 2119–2128 (2005)`_.

    Args:
            r (float or np.ndarray): The interradical separation (m).

    Returns:
            (float or np.ndarray): The dipolar coupling constant in
            millitesla (mT).

    .. _Santabarbara et al. Biochemistry, 44, 6, 2119–2128 (2005):
       https://pubs.acs.org/doi/10.1021/bi048445d
    """
    conversion = (3 * -C.g_e * C.mu_B * C.mu_0) / (8 * np.pi)
    return (-conversion / r**3) * 1000


def exchange_interaction_in_protein(
    r: float | np.ndarray, beta: float = 14e9, J0: float = 9.7e9
) -> float | np.ndarray:
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
    r: float | np.ndarray, beta: float = 0.049e-9, J0rad: float = 1.7e17
) -> float | np.ndarray:
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
    J0 = J0rad / Isotope("E").gamma_mT
    return J0 * np.exp(-r / beta)


def exchange_interaction_in_solution_MC(
    r: np.ndarray, beta: float = 2e10, J0: float = -570
) -> np.ndarray:
    """Exchange interaction for Monte Carlo trajectories.

    Sources:

        - `O'Dea et al. J. Phys. Chem. A, 109, 5, 869-873 (2005)`_.
        - `Miura et al. J. Phys. Chem. A, 119, 22, 5534-5544 (2015)`_.

    Args:
        r (np.ndarray): The interradical separation (m).
        beta (float): The range parameter (1/m).
        J0 (float): The strength of the interaction (mT).

    Returns:
        np.ndarray: The exchange coupling constant in milli Tesla (mT).

    """
    return J0 * np.exp(-beta * (r - r.min()))


def g_tensor_relaxation_rate(tau_c: float, g1: list, g2: list) -> float:
    """g-tensor relaxation rate.

    To be used with `radicalpy.relaxation.RandomFields`.

    Source: `Player et al. J. Chem. Phys. 153, 084303 (2020)`_.

    Args:
            tau_c (float): The rotational correlation time (s).
            g1 (list): The principle components of g-tensor of the
                first radical.
            g2 (list): The principle components of g-tensor of the
                second radical.

    Returns:
            float: The g-tensor relaxation rate (1/s).

    .. _Player et al. J. Chem. Phys. 153, 084303 (2020):
       https://doi.org/10.1063/5.0021643
    """
    g1sum = sum((gi - C.g_e) ** 2 for gi in g1)
    g2sum = sum((gi - C.g_e) ** 2 for gi in g2)
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
    return tau_c * D_var_MHz * 4 * np.pi**2 * 1e12


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
    return -C.g_e * (C.mu_B * 1e-3) * Bhalf / C.h


def k_electron_transfer(
    separation: float, driving_force: float = -1, reorganisation_energy: float = 1
) -> float:
    """Electron transfer rate.

    The default values (when `-driving_force ==
    reorganisation_energy`) return the maximum electron transfer rate.

    Source: `Moser et al. Biochim. Biophys. Acta Bioenerg. 1797, 1573‐1586 (2010)`_.

    Args:
            separation (float): The edge-to-edge separation, R (Å).
            driving_force (float): The driving force, :math:`\Delta G` (eV).
            reorganisation_energy (float): The reorganisation energy,
                :math:`\lambda` (eV).

    Returns:
            float: The electron transfer rate (1/s).

    .. _Moser et al. Biochim. Biophys. Acta Bioenerg. 1797, 1573‐1586 (2010):
       https://doi.org/10.1016/j.bbabio.2010.04.441

    """
    return 10 ** (
        13
        - 0.6 * (separation - 3.6)
        - 3.1 * ((driving_force + reorganisation_energy) ** 2 / reorganisation_energy)
    )


def k_excitation(
    power: float, wavelength: float, volume: float, pathlength: float, epsilon: float
) -> float:
    """Groundstate excitation rate.

    Args:
            power (float): The excitation laser power (W).
            wavelength (float): The excitation wavelength (m).
            volume (float): The excitation beam volume (m^3).
            pathlength (float): The path length of the sample cell
                (m).
            epsilon (float): The extinction coefficient of the
                molecule (m^2/mol).

    Returns:
            float: The excitation rate (1/s).
    """
    nu = C.c / wavelength  # Frequency of excitation beam (1/s)
    I0 = power / (C.h * nu * C.N_A * volume)  # Initial intensity (I0)
    return I0 * np.log(10) * epsilon * pathlength


def k_recombination(MFE: float, k_escape: float) -> float:
    """Singlet recombination rate.

    Source: `Maeda et al. Mol. Phys., 117:19, 2709-2718 (2019)`_.

    Args:
            MFE (float): The magnetic field effect (0.00-1.00).
            k_escape (float): The free radical formation rate constant (1/s).

    Returns:
            float: The singlet recombination rate (1/s).

    .. _Maeda et al. Mol. Phys., 117:19, 2709-2718 (2019):
       https://doi.org/10.1080/00268976.2019.1580779
    """
    b = (1 - 6 * MFE) * k_escape
    return 0.5 * (-b + np.sqrt(b**2 + 48 * k_escape**2 * MFE))


def k_reencounter(encounter_dist: float, diff_coeff: float) -> float:
    """Re-encounter rate.

    Source: `Salikhov, J. Magn. Reson., 63, 271-279 (1985)`_.

    Args:
            encounter_dist (float): The effective re-encounter
                distance, R* (m).
            diff_coeff (float): The diffusion coefficient, D (m^2/s).

    Returns:
            float: The re-encounter rate (1/s).

    .. _Salikhov, J. Magn. Reson., 63, 271-279 (1985):
       https://doi.org/10.1016/0022-2364(85)90316-6

    """
    return (encounter_dist**2 / diff_coeff) ** -1


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
    B0 = utils.mT_to_MHz(B0)
    nu_0 = (C.g_e * (C.mu_B * 1e-3) * B0) / C.h
    jnu0tc = (2 / 15) * (
        (4 * tau_c) / (1 + 4 * nu_0**2 * tau_c**2)
        + (tau_c) / (1 + nu_0**2 * tau_c**2)
    )
    return (D**2 + 3 * E**2) * jnu0tc


def rotational_correlation_time_for_molecule(
    radius: float, temp: float, eta: float = 0.89e-3
) -> float:
    """Rotational correlation time.

    Rotational correlation time is the average time it takes for a
    molecule (smaller than a protein) to rotate one radian. For
    proteins see
    `radicalpy.estimations.rotational_correlation_time_for_protein`.

    To calculate viscosity (eta) for glycerol-water mixtures see
    `radicalpy.estimations.aqueous_glycerol_viscosity`.

    Args:
            radius (float): The radius of a spherical molecule (m).
            temp (float): The temperature of the solution (K).
            eta (float): The viscosity of the solution (N s/m^2)
                (default: 0.89e-3 corresponds to water).

    Returns:
            float: The rotational correlation time (s).
    """
    return (4 * np.pi * eta * radius**3) / (3 * C.k_B * temp)


def rotational_correlation_time_for_protein(
    Mr: float, temp: float, eta: float = 0.89e-3
) -> float:
    """Rotational correlation time.

    Rotational correlation time is the average time it takes for a
    protein to rotate one radian. For small molecules see
    `radicalpy.estimations.rotational_correlation_time_for_molecule`.

    To calculate viscosity (eta) for glycerol-water mixtures see
    `radicalpy.estimations.aqueous_glycerol_viscosity`.

    Source: `Cavanagh et al. Protein NMR Spectroscopy. Principles and
    Practice, Elsevier Academic Press (2007)`_.

    Args:
            Mr (float): The molecular weight of the protein (kDa).
            temp (float): The temperature of the solution (K).
            eta (float): The viscosity of the solution (N s/m^2)
                (default: 0.89e-3 corresponds to water).

    Returns:
            float: The rotational correlation time (s).

    .. _Cavanagh et al. Protein NMR  Spectroscopy. Principles and Practice, Elsevier Academic Press (2007):
       https://doi.org/10.1016/B978-0-12-164491-8.X5000-3
    """
    Rh = ((3 * C.V * Mr) / (4 * np.pi * C.N_A)) ** 0.33 + C.rw
    return rotational_correlation_time_for_molecule(Rh, temp, eta)
