#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from . import estimations, utils


def get_delta_r(mutual_diffusion: float, delta_T: float) -> float:
    """Mean path between two radicals.

    Args:
            mutual_diffusion (float): The mutual diffusion coefficient (m^2/s).
            delta_T (float): The time interval (s).

    Returns:
            float: The mean path between two radicals (m).
    """
    return np.sqrt(6 * mutual_diffusion * delta_T)


def kinetics(
    time: np.ndarray, initial_populations: list, states: list, rate_equations: dict
) -> np.ndarray:
    """Kinetic rate equation solver.

    Constructs the matrix propagator and performs a time evolution simulation.

    Args:
            time (np.ndarray): The timescale of the reaction kinetics (s).
            initial_populations (list): The initial populations of all states.
            states (list): The states involved in the chemical reaction.
            rate_equations (dict): The rate equations for all states.

    Returns:
            np.ndarray: The time evolution of all states.
    """
    shape = (len(states), len(states))
    arrange = [
        rate_equations[i][j] if (i in rate_equations and j in rate_equations[i]) else 0
        for i in states
        for j in states
    ]
    rates = np.reshape(arrange, shape)
    dt = time[1] - time[0]
    result = np.zeros([len(time), *rates[0].shape], dtype=float)
    propagator = sp.sparse.linalg.expm(sp.sparse.csc_matrix(rates) * dt)
    result[0] = initial_populations
    for t in range(1, len(time)):
        result[t] = propagator @ result[t - 1]
    return result


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
    """Simulate Monte Carlo random walk.

    The MC random walk is simulated for radicals pairs in both
    solution (`r_max = 0`) and microreactor (`r_max > 0`)
    environments.

    Args:
            n_steps (float): The number of simulation steps.
            x_0 (float): The initial position in the x-axis (m).
            y_0 (float): The initial position in the x-axis (m).
            z_0 (float): The initial position in the x-axis (m).
            delta_r (float): The mean path between two radicals (m).
            r_min (float): The distance of closest approach (m).
            r_max (float): The diameter of the microreactor (m). Set
                to 0 for solution-based, and to a positive value for
                microreactor-based simulations.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray):
            - pos: The positions of the moving radical (m).
            - dist: The mutual distances between the radical pairs (m).
            - angle: The angles (theta) of the vector trajectories of
              the moving radical (m).

    """
    if r_max != 0 and r_min > r_max:
        raise ValueError("r_min should be less than (or equal to) r_max.")
    if r_min < 0 or r_max < 0:
        raise ValueError("r_min and r_max should not be negative.")
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
