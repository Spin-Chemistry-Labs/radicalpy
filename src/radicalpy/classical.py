#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from . import estimations, utils


def get_delta_r(mutual_diffusion: float, delta_T: float) -> float:
    """Mean path between two radicals.

    Args:
            mutual_diffusion (float): The mutual diffusion coefficient (m^2 s^-1).
            delta_T (float): The time interval (s).

    Returns:
            float: The mean path between two radicals (m).
    """
    return np.sqrt(6 * mutual_diffusion * delta_T)


def _random_theta_phi():
    """Random sampling of theta and phi.

    Returns:
            Theta and phi (radians).
    """
    theta = np.pi * np.random.rand()
    arg = np.random.uniform(-1, 1)
    phi = 2 * np.sign(arg) * np.arcsin(np.sqrt(np.abs(arg)))
    return theta, phi


def randomwalk_3d(
    n_steps: float,
    x_0: float,
    y_0: float,
    z_0: float,
    delta_r: float,
    r_min: float,
    r_max: float = 0,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Monte Carlo random walk for radicals pairs in both solution and microreactor environments.

    Args:
            n_steps (float): The number of simulation steps.
            x_0 (float): The initial position in the x-axis (m).
            y_0 (float): The initial position in the x-axis (m).
            z_0 (float): The initial position in the x-axis (m).
            delta_r (float): The mean path between two radicals (m).
            r_min (float): The distance of closest approach (m).
            r_max (float): The diameter of the microreactor (m). Set to 0 for solution-based simulations.

    Returns:
            (np.ndarray, np.ndarray, np.ndarray)
            pos (np.ndarray): The positions of the moving radical (m).
            dist (np.ndarray): The mutual distances between the radical pairs (m).
            angle (np.ndarray): The angles (theta) of the vector trajectories of the moving radical (m).
    """
    pos = np.zeros([n_steps, 3])
    dist = np.zeros(n_steps)
    angle = np.zeros(n_steps)

    pos[0] = np.array([x_0, y_0, z_0])
    dist[0] = np.linalg.norm(pos[0])

    for i in range(1, n_steps):
        theta, phi = _random_theta_phi()
        new_pos = pos[i - 1] + delta_r * utils.spherical_to_cartesian(theta, phi)
        d = np.linalg.norm(new_pos)
        while (r_max > 0 and d >= r_max - r_min) or d <= r_min + r_min:
            theta, phi = _random_theta_phi()
            new_pos = pos[i - 1] + delta_r * utils.spherical_to_cartesian(theta, phi)
            d = np.linalg.norm(new_pos)
        angle[i] = theta
        pos[i] = new_pos
        dist[i] = d
    return pos, dist, angle
