#!/usr/bin/env python

"""Utility functions.

.. todo:: Add module docstring.
"""
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.fftpack import fft, ifft, ifftshift
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from .shared import constants as C


def is_fast_run():
    """Is the `--fast` parameter set at execution.

    This function helps examples to be used as tests.  By running the
    example with the `--fast` option, a faster version of main can be
    called (e.g., by setting fewer number of time steps etc.).

    """
    return len(sys.argv) == 2 and sys.argv[1] == "--fast"


def Bhalf_fit(
    B: np.ndarray, MARY: np.ndarray
) -> Tuple[float, np.ndarray, float, float]:
    """B_1/2 fit for MARY spectra.

    Args:
            B (np.ndarray): Magnetic field values (x-axis).
            MARY (np.ndarray): Magnetic field effect data
                (y-axis). Use the `MARY` entry in the result of
                `radicalpy.simulation.HilbertSimulation.MARY`.

    Returns:
            (float, np.ndarray, float, float):
            - `Bhalf` (float): The magnetic field strength at half the
              saturation magnetic field.
            - `fit_result` (np.ndarray): y-axis from fit.
            - `fit_error` (float): Standard error for the fit.
            - `R2` (float): R-squared value for the fit.

    """
    popt_MARY, pcov_MARY = curve_fit(
        Lorentzian,
        B,
        MARY,
        p0=[MARY[-1], int(len(B) / 2)],
        maxfev=1000000,
    )
    fit_error = np.sqrt(np.diag(pcov_MARY))

    A_opt_MARY, Bhalf_opt_MARY = popt_MARY
    fit_result = Lorentzian(B, *popt_MARY)
    Bhalf = np.abs(Bhalf_opt_MARY)

    y_pred_MARY = Lorentzian(B, *popt_MARY)
    R2 = r2_score(MARY, y_pred_MARY)

    return Bhalf, fit_result, fit_error, R2


def Gauss_to_MHz(Gauss: float) -> float:
    """Convert Gauss to MHz.

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: The magnetic flux density converted to MHz.
    """
    return Gauss / (1e-10 * -C.g_e * C.mu_B / C.h)


def Gauss_to_angular_frequency(Gauss: float) -> float:
    """Convert Gauss to angular frequency.

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: The magnetic flux density converted to angular
            frequency (rad/s/T).
    """
    return Gauss * (C.mu_B / C.hbar * -C.g_e / 1e10)


def Gauss_to_mT(Gauss: float) -> float:
    """Convert Gauss to millitesla.

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: The magnetic flux density converted to millitesla
            (mT).
    """
    return Gauss / 10


def Lorentzian(B: np.ndarray, amplitude: float, Bhalf: float) -> np.ndarray:
    """Lorentzian function for MARY spectra.

    More information in `radicalpy.utils.Bhalf_fit` (where this is
    used).

    Args:
            B (np.ndarray): The x-axis magnetic field values.
            amplitude (float): The amplitude of the saturation field value.
            Bhalf (float): The magnetic field strength at half the
                saturation field value.

    Returns:
            np.ndarray: Lorentzian function for MARY spectrum.

    """
    return (amplitude / Bhalf**2) - (amplitude / (B**2 + Bhalf**2))


def mary_lorentzian(mod_signal: np.ndarray, lfe_magnitude: float):
    """Lorentzian MARY spectral shape.

    Used for brute force modulated MARY simulations.

    Args:
            mod_signal (np.ndarray): The modulated signal.
            lfe_magnitude (float): The magnitude of your low field effect (G).

    Returns:
            np.ndarray: The modulated MARY signal.
    """
    return 1 / (1 + mod_signal**2) - lfe_magnitude / (0.1 + mod_signal**2)


def MHz_in_angular_frequency(MHz: float) -> float:
    """Convert MHz into angular frequency.

    Args:
            MHz (float): The angular frequency in MHz/T

    Returns:
            float: The angular frequency converted to rad/s/T.
    """
    return MHz * (2 * np.pi)


def MHz_to_Gauss(MHz: float) -> float:
    """Convert Megahertz to Gauss.

    Args:
            MHz (float): The frequency in Megahertz (MHz).

    Returns:
            float: Megahertz (MHz) converted to Gauss (G).
    """
    return MHz / (1e-10 * -C.g_e * C.mu_B / C.h)


def MHz_to_mT(MHz: float) -> float:
    """Convert Megahertz to milltesla.

    Args:
            MHz (float): The frequency in Megahertz (MHz).

    Returns:
            float: Megahertz (MHz) converted to millitesla (mT).
    """
    return MHz / (1e-9 * -C.g_e * C.mu_B / C.h)


def modulated_signal(timeconstant: np.ndarray, theta: float, frequency: float):
    """Modulated MARY signal.

    Used for brute force modulated MARY simulations.

    Args:
            timeconstant (np.ndarray): The modulation time constant (s).
            theta (float): The modulation phase (rad).
            frequency (float): The modulation frequency (Hz).

    Returns:
            np.ndarray: The modulated signal.
    """
    return np.cos(frequency * timeconstant * (2 * np.pi) + theta)


def angular_frequency_in_MHz(ang_freq: float) -> float:
    """Convert angular frequency into MHz.

    Args:
            ang_freq (float): The angular frequency in rad/s/T.

    Returns:
            float: The angular frequency converted to MHz/T.
    """
    return ang_freq / (2 * np.pi)


def angular_frequency_to_Gauss(ang_freq: float) -> float:
    """Convert angular frequency to Gauss.

    Args:
            ang_freq (float): The angular frequency in rad/s/T.

    Returns:
            float: The angular frequency converted to Gauss (G).
    """
    return ang_freq / (C.mu_B / C.hbar * -C.g_e / 1e10)


def angular_frequency_to_mT(ang_freq: float) -> float:
    """Convert angular frequency to millitesla.

    Args:
            ang_freq (float): The angular frequency in rad/s/T.

    Returns:
            float: The angular frequency converted to millitesla (mT).
    """
    return ang_freq / (C.mu_B / C.hbar * -C.g_e / 1e9)


def autocorrelation(data: np.ndarray, factor: int = 1) -> np.ndarray:
    """Calculate the autocorrelation of a trajectory.

    An FFT-based implementation of the autocorrelation for Monte Carlo
    or molecular dynamics trajectories (or any other time dependent
    value).

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


def cartesian_to_spherical(
    x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Convert Cartesian coordinates to spherical coordinates.

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


def mT_to_Gauss(mT: float) -> float:
    """Convert millitesla to Gauss.

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: The magnetic flux density converted to Gauss (G).
    """
    return mT * 10


def mT_to_MHz(mT: float) -> float:
    """Convert millitesla to Megahertz.

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: The magnetic flux density converted to Megahertz (MHz).
    """
    return mT * (1e-9 * -C.g_e * C.mu_B / C.h)


def mT_to_angular_frequency(mT: float) -> float:
    """Convert millitesla to angular frequency.

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: The magnetic flux density converted to angular frequency (rad/s/T).
    """
    return mT * (C.mu_B / C.hbar * -C.g_e / 1e9)


def reference_signal(
    timeconstant: np.ndarray, harmonic: float, theta: float, frequency: float
):
    """Modulated MARY reference signal.

    Used for brute force modulated MARY simulations.

    Args:
            timeconstant (np.ndarray): The modulation time constant (s).
            harmonic (float): The harmonic of the modulation.
            theta (float): The modulation phase (rad).
            frequency (float): The modulation frequency (Hz).

    Returns:
            np.ndarray: The modulated reference signal.
    """
    return np.cos(harmonic * frequency * timeconstant * (2 * np.pi) + harmonic * theta)


def spectral_density(omega: float, tau_c: float) -> float:
    """Frequency at which the motion of the particle exists.

    Args:
            omega (float): The Larmor frequency of the electron.
            tau_c (float): The rotational correlation time.

    Returns:
            float: Spectral density frequency.
    """
    return tau_c / (1 + omega**2 * tau_c**2)


def anisotropy_check(
    theta: float | np.ndarray, phi: float | np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(theta, float):
        theta = np.array([theta])
    if isinstance(phi, float):
        phi = np.array([phi])
    if theta.min() < 0 or np.pi < theta.max():
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


def _check_full_sphere(theta: np.ndarray, phi: np.ndarray) -> Tuple[int, int]:
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


def spherical_average(
    product_yield: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> float:
    """Spherical average of anisotropic product yields.

    Args:
            product_yield (np.ndarray): The anisotropic product
                yields.
            theta (np.ndarray): The angles theta by which the
                anisotropic product yields were calculated.
            phi (np.ndarray): The angles phi by which the anisotropic
                product yields were calculated.

    Returns:
            float: The spherical average of the anisotropic product
                yields.
    """
    theta, phi = anisotropy_check(theta, phi)
    nth, nph = _check_full_sphere(theta, phi)

    wt = 4 * np.ones(nth)
    wt[2:-2:2] = 2
    wt[0] = wt[-1] = 1

    wp = 4 * np.ones(nph)
    wp[0:-1:2] = 2
    sintheta = np.sin(np.linspace(0, np.pi, nth))

    spherical_avg = sum(
        product_yield[i, j] * sintheta[i] * wt[i] * wp[j]
        for i in range(nth)
        for j in range(nph)
    )

    return spherical_avg * theta[1] * phi[1] / (4 * np.pi) / 9


def spherical_to_cartesian(
    theta: float | np.ndarray, phi: float | np.ndarray
) -> np.ndarray:
    """Spherical coordinates to Cartesian coordinates.

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
    ).T


def yield_anisotropy(
    product_yield: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> Tuple[float, float]:
    """Calculate the yield anisotropy.

    Args:
            product_yield (np.ndarray): The anisotropic product yields.
            theta (np.ndarray): The angles theta by which the
                anisotropic product yields were calculated.
            phi (np.ndarray): The angles phi by which the anisotropic
                product yields were calculated.

    Returns:
            (float, float):
            - delta_phi (float): Maximum yield - minimum yield.
            - gamma (float): delta_phi / spherical average.

    """
    delta_phi = product_yield.max() - product_yield.min()
    yield_av = spherical_average(product_yield, theta, phi)
    gamma = delta_phi / yield_av
    return delta_phi, gamma


def read_trajectory_files(path: Path, scale=1e-10):
    data = [np.genfromtxt(file_path) for file_path in Path(path).glob("*")]
    return scale * np.concatenate(data, axis=0)
