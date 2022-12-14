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
    """Tensor transformation: Anisotropic tensor to isotropic value.

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
    """Convert units: Megahertz (MHz) to Gauss (G).

    Args:
            MHz (float): The frequency in Megahertz (MHz).

    Returns:
            float: Megahertz (MHz) converted to Gauss (G).

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return MHz / (1e-10 * -g_e * mu_B / h)


def MHz_to_mT(MHz: float) -> float:
    """Convert units: Megahertz (MHz) to milltesla (mT).

    Args:
            MHz (float): The frequency in Megahertz (MHz).

    Returns:
            float: Megahertz (MHz) converted to millitesla (mT).

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return MHz / (1e-9 * -g_e * mu_B / h)


def mT_to_angular_frequency(mT: float) -> float:
    """Convert units: millitesla (mT) to angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`).

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: millitesla (mT) converted to angular frequency (:math:`\\text{rad} \\cdot \\text{s}^{-1} \\cdot \\text{T}^{-1}`).

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return mT * (mu_B / hbar * -g_e / 1e9)


def mT_to_Gauss(mT: float) -> float:
    """Convert units: millitesla (mT) to Gauss (G).

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: millitesla (mT) converted to Gauss (G).

    """
    return mT * 10


def mT_to_MHz(mT: float) -> float:
    """Convert units: millitesla (mT) to Megahertz (MHz).

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: millitesla (mT) converted to Megahertz (MHz).

    """
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return mT * (1e-9 * -g_e * mu_B / h)


def multiexponential_fit(x, *args):
    """Curve fitting: Multiexponential function for autocorrelation fitting.

    Args:
            x (np.ndarray): The x-axis values.
            args (np.ndarray): The amplitudes (A) and taus for fitting.

    Returns:
            np.ndarray: Multiexponential fit for autocorrelations.

    """
    n = len(args) // 2
    A, tau = list(args)[:n], list(args)[n:]
    return sum(a * np.exp(-t * x) for a, t in zip(A, tau))


def rotation_matrix_x(alpha: float) -> np.ndarray:
    """Tensor transformation: Rotation matrix to rotate a vector or matrix by an angle alpha about the x-axis in 3D.

    Args:
            alpha (float): The angle to rotate a vector about the x-axis.

    Returns:
            np.ndarray: Rotated vector or matrix.

    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )


def rotation_matrix_y(beta: float) -> np.ndarray:
    """Tensor transformation: Rotation matrix to rotate a vector or matrix by an angle beta about the y-axis in 3D.

    Args:
            beta (float): The angle to rotate a vector about the x-axis.

    Returns:
            np.ndarray: Rotated vector or matrix.

    """
    return np.array(
        [
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ]
    )


def rotation_matrix_z(gamma: float) -> np.ndarray:
    """Tensor transformation: Rotation matrix to rotate a vector or matrix by an angle gamma about the z-axis in 3D.

    Args:
            gamma (float): The angle to rotate a vector about the x-axis.

    Returns:
            np.ndarray: Rotated vector or matrix.

    """
    return np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )


def spectral_density(omega: float, tau_c: float) -> float:
    """Spectral density: The frequency at which the motion of the particle exists.

    Args:
            omega (float): The Larmor frequency of the electron.
            tau_c (float): The rotational correlation time.

    Returns:
            float: Spectral density frequency.

    """
    return tau_c / (1 + omega**2 * tau_c**2)


def spherical_average(
    product_yield: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> float:
    """Spherical average: The spherical average of anisotropic product yields.

    Args:
            product_yield (np.ndarray): The anisotropic product yields.
            theta (np.ndarray): The angles theta by which the anisotropic product yields were calculated.
            phi (np.ndarray): The angles phi by which the anisotropic product yields were calculated.

    Returns:
            float: The spherical average of the anisotropic product yields.

    """
    theta, phi = _anisotropy_check(theta, phi)
    nth, nph = check_full_sphere_coordinates(theta, phi)

    wt = 4 * np.ones(nth)
    wt[2:-2:2] = 2
    wt[0] = wt[-1] = 1

    wp = 4 * np.ones(nph)
    wp[0:-1:2] = 2
    sintheta = np.sin(np.linspace(0, np.pi, nth))

    spherical_average = sum(
        product_yield[i, j] * sintheta[i] * wt[i] * wp[j]
        for i in range(nth)
        for j in range(nph)
    )

    return spherical_average * theta[1] * phi[1] / (4 * np.pi) / 9


def spherical_to_cartesian(
    theta: float or np.ndarray, phi: float or np.ndarray
) -> np.ndarray:
    """Convert units: spherical coordinates to Cartesian coordinates.

    Args:
            theta (float or np.ndarray): The polar angle(s).
            phi (float or np.ndarray): The azimuthal angle(s).

    Returns:
            np.ndarray: The Cartesian coordinates.

    """
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )


def spin_quantum_number(multiplicity: int) -> float:
    """Conversion: Multiplicity to spin quantum number.

    Args:
            multiplicity (int): Spin multiplicity.

    Returns:
            float: Spin quantum number.

    """
    return float(multiplicity - 1) / 2.0


def square_vectors(rhos):
    shape = rhos.shape
    if shape[-1] != shape[-2]:
        dim = int(np.sqrt(shape[-2]))
        rhos = rhos.reshape(shape[0], shape[1], dim, dim)
    return rhos


def yield_anisotropy(
    product_yield: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> (float, float):
    """Calculate the yield anisotropy.

    Args:
            product_yield (np.ndarray): The anisotropic product yields.
            theta (np.ndarray): The angles theta by which the anisotropic product yields were calculated.
            phi (np.ndarray): The angles phi by which the anisotropic product yields were calculated.

    Returns:
            delta_phi (float): Maximum yield - minimum yield.
            gamma (float): delta_phi / spherical average.
    """
    delta_phi = product_yield.max() - product_yield.min()
    yield_av = spherical_average(product_yield, theta, phi)
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
