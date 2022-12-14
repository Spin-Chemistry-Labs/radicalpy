#!/usr/bin/env python

from typing import Iterable

import numpy as np
from scipy.fftpack import fft, ifft, ifftshift
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from .data import constants


def angular_frequency_to_Gauss(ang_freq: float) -> float:
    """Convert units: Angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`) to Gauss (G).

    Args:
            ang_freq (float): The angular frequency in :math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`.

    Returns:
            float: The angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`) converted to Gauss (G).

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return ang_freq / (mu_B / hbar * -g_e / 1e10)


def angular_frequency_to_MHz(ang_freq: float) -> float:
    """Convert units: Angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`) to (:math:`\\text{MHz} \\cdot \\text{T}^{-1}`).

    Args:
            ang_freq (float): The angular frequency in :math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`.

    Returns:
            float: The angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`) converted to :math:`\\text{MHz} \\cdot \\text{T}^{-1}`.

    """
    return ang_freq / (2 * np.pi)


def angular_frequency_to_mT(ang_freq: float) -> float:
    """Convert units: Angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`) to millitesla (mT).

    Args:
            ang_freq (float): The angular frequency in :math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`.

    Returns:
            float: The angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`) converted to millitesla (mT).

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return ang_freq / (mu_B / hbar * -g_e / 1e9)


def autocorrelation(data: np.ndarray, factor=2) -> np.ndarray:
    """FFT-based autocorrelation of Monte Carlo or molecular dynamics trajectories.

    Args:
            data (np.ndarray): The time dependent trajectory.
                        factor (int): Data length reduction factor.

    Returns:
            np.ndarray: The autocorrelation of the trajectory.

    """
    datap = ifftshift((data - np.average(data)) / np.std(data))
    n = datap.shape[0]
    datap = np.r_[datap[: n // factor], np.zeros_like(datap), datap[n // factor :]]
    f = fft(datap)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    result = np.real(pi)[: n // factor] / np.arange(n, 0, -1)[: n // factor]
    result = np.delete(result, 0)
    return result


def Bhalf_fit(
    B: np.ndarray, MARY: np.ndarray
) -> (float, np.ndarray, np.ndarray, float, float):
    """Curve fitting: Lorentzian fit for MARY spectra.

    Args:
            B (np.ndarray): Magnetic field values (x-axis).
                        MARY (np.ndarray): Magnetic field effect data (y-axis).

    Returns:
            Bhalf (float): The magnetic field strength at half the saturation magnetic field.
                        x_model_MARY (np.ndarray): x-axis from fit.
                        y_model_MARY (np.ndarray): y-axis from fit.
                        MARY_fit_error (float): Standard error for the fit.
                        R2 (float): R-squared value for the fit.

    """
    popt_MARY, pcov_MARY = curve_fit(
        Lorentzian_fit,
        B,
        MARY,
        p0=[MARY[-1], int(len(B) / 2)],
        maxfev=1000000,
    )
    MARY_fit_error = np.sqrt(np.diag(pcov_MARY))

    A_opt_MARY, Bhalf_opt_MARY = popt_MARY
    x_model_MARY = np.linspace(min(B), max(B), len(B))
    y_model_MARY = Lorentzian_fit(x_model_MARY, *popt_MARY)
    Bhalf = np.abs(Bhalf_opt_MARY)

    y_pred_MARY = Lorentzian_fit(B, *popt_MARY)
    R2 = r2_score(MARY, y_pred_MARY)

    return Bhalf, x_model_MARY, y_model_MARY, MARY_fit_error, R2


def cartesian_to_spherical(
    x: float or np.ndarray, y: float or np.ndarray, z: float or np.ndarray
) -> (float or np.ndarray, float or np.ndarray, float or np.ndarray):
    """Convert units: Cartesian coordinates to spherical coordinates.

    Args:
            x (float or np.ndarray): Coordinate(s) in the x plane.
                        y (float or np.ndarray): Coordinate(s) in the y plane.
                        z (float or np.ndarray): Coordinate(s) in the z plane.

    Returns:
            r (float or np.ndarray): The radial distance(s).
                        theta (float or np.ndarray): The polar angle(s).
                        phi (float or np.ndarray): The azimuthal angle(s).

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def check_full_sphere_coordinates(theta: Iterable, phi: Iterable) -> (int, int):
    nth, nph = len(theta), len(phi)
    if not np.all(np.isclose(theta, np.linspace(0, np.pi, nth))):
        raise ValueError(
            "Not a full sphere: `theta` should be `linspace(0, np.pi, ntheta)`"
        )
    if not np.all(np.isclose(phi, np.linspace(0, 2 * np.pi, nph))):
        raise ValueError(
            "Not a full sphere: `phi` should be `linspace(0, np.pi, nphi)`"
        )
    return nth, nph


def Gauss_to_angular_frequency(Gauss: float) -> float:
    """Convert units: Gauss (G) to angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`).

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: Gauss (G) converted to angular frequency in :math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`.

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return Gauss * (mu_B / hbar * -g_e / 1e10)


def Gauss_to_MHz(Gauss: float) -> float:
    """Convert units: Gauss (G) to Megahertz (MHz).

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: Gauss (G) converted to Megahertz (MHz).

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return Gauss / (1e-10 * -g_e * mu_B / h)


def Gauss_to_mT(Gauss: float) -> float:
    """Convert units: Gauss (G) to millitesla (mT).

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: Gauss (G) converted to millitesla (mT).

    """
    return Gauss / 10


def get_idx(values, target):
    return np.abs(target - values).argmin()


def isotropic(anisotropic: np.ndarray or list) -> float:
    """Convert tensors: Anisotropic tensor to isotropic value.

    Args:
            anisotropic (np.ndarray or list): The 3x3 interaction tensor matrix.

    Returns:
            float: isotropic value.

    """
    return np.trace(anisotropic) / 3


def Lorentzian_fit(x: np.ndarray, A: np.ndarray, Bhalf: float) -> np.ndarray:
    """Curve fitting: Lorentzian function for MARY spectra.

    Args:
            x (np.ndarray): The x-axis values.
                        A (np.ndarray): The amplitudes (intensity scaling).
                        Bhalf (float): The magnetic field strength at half the saturation magnetic field.

    Returns:
            np.ndarray: Lorentzian fit for MARY spectrum.

    """
    return (A / Bhalf**2) - (A / (x**2 + Bhalf**2))


def MHz_to_angular_frequency(MHz: float) -> float:
    """Convert units: Megahertz (:math:`\\text{MHz} \\cdot \\text{T}^{-1}`) to angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`).

    Args:
            MHz (float): The frequency in Megahertz (:math:`\\text{MHz} \\cdot \\text{T}^{-1}`).

    Returns:
            float: Megahertz (:math:`\\text{MHz} \\cdot \\text{T}^{-1}`) converted to angular frequency in :math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`.

    """
    return MHz * (2 * np.pi)


def MHz_to_Gauss(MHz: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return MHz / (1e-10 * -g_e * mu_B / h)


def MHz_to_mT(MHz: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return MHz / (1e-9 * -g_e * mu_B / h)


def mT_to_angular_frequency(mT: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return mT * (mu_B / hbar * -g_e / 1e9)


def mT_to_Gauss(mT: float) -> float:
    return mT * 10


def mT_to_MHz(mT: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return mT * (1e-9 * -g_e * mu_B / h)


def multiexponential_fit(x, *args):
    n = len(args) // 2
    A, tau = list(args)[:n], list(args)[n:]
    return sum(a * np.exp(-t * x) for a, t in zip(A, tau))


def rotation_matrix_x(phi: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ]
    )


def rotation_matrix_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rotation_matrix_z(psi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1],
        ]
    )


def spectral_density(omega, tau_c):
    return tau_c / (1 + omega**2 * tau_c**2)


def spherical_average(singlet_yield, theta, phi):
    # Subtracting the spherical average from the singlet yield
    # Simpson's rule integration over theta (0, pi) and phi (0, 2pi)
    # n_theta = odd, n_phi = even

    theta, phi = _anisotropy_check(theta, phi)
    nth, nph = check_full_sphere_coordinates(theta, phi)

    wt = 4 * np.ones(nth)
    wt[2:-2:2] = 2
    wt[0] = wt[-1] = 1

    wp = 4 * np.ones(nph)
    wp[0:-1:2] = 2
    sintheta = np.sin(np.linspace(0, np.pi, nth))

    spherical_average = sum(
        singlet_yield[i, j] * sintheta[i] * wt[i] * wp[j]
        for i in range(nth)
        for j in range(nph)
    )

    return spherical_average * theta[1] * phi[1] / (4 * np.pi) / 9


def spherical_to_cartesian(theta, phi):
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def spin_quantum_number(multiplicity: int) -> float:
    return float(multiplicity - 1) / 2.0


def square_vectors(rhos):
    shape = rhos.shape
    if shape[-1] != shape[-2]:
        dim = int(np.sqrt(shape[-2]))
        rhos = rhos.reshape(shape[0], shape[1], dim, dim)
    return rhos


def yield_anisotropy(yields, theta, phi):
    delta_phi = yields.max() - yields.min()
    yield_av = spherical_average(yields, theta, phi)
    gamma = delta_phi / yield_av
    return delta_phi, gamma


def _anisotropy_check(
    theta: Iterable or float, phi: Iterable or float
) -> (Iterable, Iterable):
    if isinstance(theta, float):
        theta = [theta]
    if isinstance(phi, float):
        phi = [phi]
    if min(theta) < 0 or np.pi < max(theta):
        raise ValueError("Value of `theta` needs to be between 0 and pi!")
    if min(phi) < 0 or 2 * np.pi < max(phi):
        raise ValueError("Value of `phi` needs to be between 0 and 2*pi!")
    lt, lp = len(theta), len(phi)
    if lt > 1 and lp > 1:
        # theta odd, phi even
        if lt % 2 == 0:
            raise ValueError("Number of `len(theta)` needs to be odd!")
        if lp % 2 == 1:
            raise ValueError("Number of `len(phi)` needs to be even!")
    return theta, phi
