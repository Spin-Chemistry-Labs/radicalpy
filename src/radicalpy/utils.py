#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from .data import constants


def angular_frequency_to_Gauss(ang_freq: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return ang_freq / (mu_B / hbar * -g_e / 1e10)


def angular_frequency_to_MHz(ang_freq: float) -> float:
    return ang_freq / (2 * np.pi)


def angular_frequency_to_mT(ang_freq: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return ang_freq / (mu_B / hbar * -g_e / 1e9)


def Bhalf_fit(B, MARY):
    popt_MARY, pcov_MARY = curve_fit(
        Lorentzian_fit, B, MARY, p0=[MARY[-1], int(len(B) / 2)],
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


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def Gauss_to_angular_frequency(Gauss: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return Gauss * (mu_B / hbar * -g_e / 1e10)


def Gauss_to_MHz(Gauss: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return Gauss / (1e-10 * -g_e * mu_B / h)


def Gauss_to_mT(Gauss: float) -> float:
    return Gauss / 10


def isotropic(anisotropic: np.ndarray or list):
    return np.trace(anisotropic) / 3


def Lorentzian_fit(x, A, Bhalf):
    return (A / Bhalf**2) - (A / (x**2 + Bhalf**2))


def MHz_to_angular_frequency(MHz: float) -> float:
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


def spherical_to_cartesian(theta, phi):
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )


def spin_quantum_number(multiplicity: int) -> float:
    return float(multiplicity - 1) / 2.0
