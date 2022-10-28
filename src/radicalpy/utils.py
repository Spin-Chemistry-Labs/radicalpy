#!/usr/bin/env python

import numpy as np

from .data import constants


def angular_frequency_to_Gauss(ang_freq: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return ang_freq / mu_B / hbar * g_e / 1e10


def angular_frequency_to_MHz(ang_freq: float) -> float:
    return ang_freq / 2 * np.pi


def angular_frequency_to_mT(ang_freq: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return ang_freq / mu_B / hbar * g_e / 1e9


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def Gauss_to_angular_frequency(Gauss: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return Gauss * mu_B / hbar * g_e / 1e10


def Gauss_to_MHz(Gauss: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return Gauss / 1e-10 * g_e * mu_B / h


def Gauss_to_mT(Gauss: float) -> float:
    return Gauss / 10


def MHz_to_angular_frequency(MHz: float) -> float:
    return MHz * 2 * np.pi


def MHz_to_Gauss(MHz: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return MHz / 1e-10 * g_e * mu_B / h


def MHz_to_mT(MHz: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return MHz / 1e-9 * g_e * mu_B / h


def mT_to_angular_frequency(mT: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    hbar = constants.value("hbar")
    return mT * mu_B / hbar * g_e / 1e9


def mT_to_Gauss(mT: float) -> float:
    return mT * 10


def mT_to_MHz(mT: float) -> float:
    g_e = constants.value("g_e")
    mu_B = constants.value("mu_B")
    h = constants.value("h")
    return mT * 1e-9 * g_e * mu_B / h


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
