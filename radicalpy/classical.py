#!/usr/bin/env python


import numpy as np
import scipy as sp

from . import utils


def get_delta_r(mutual_diffusion: float, delta_T: float) -> float:
    """Mean path between two radicals.

    Args:
            mutual_diffusion (float): The mutual diffusion coefficient (m^2/s).
            delta_T (float): The time interval (s).

    Returns:
            float: The mean path between two radicals (m).
    """
    return np.sqrt(6 * mutual_diffusion * delta_T)


class RateEquations:
    """Results for `kinetics_solver`"""

    def __init__(self, rate_equations: dict, time: np.ndarray, initial_states: dict):
        self.rate_equations = rate_equations
        inner_keys = [list(v.keys()) for v in rate_equations.values()]
        outer_keys = list(rate_equations.keys())
        all_keys = list(set(sum(inner_keys, outer_keys)))
        self.indices = {k: i for i, k in enumerate(all_keys)}
        self._calc_(time, initial_states)

    @property
    def all_keys(self) -> list:
        return list(self.indices.keys())

    def are_valid_keys(self, keys: dict) -> bool:
        return set(keys).issubset(self.all_keys)

    def check_initial_states(self, initial_states):
        if not self.are_valid_keys(initial_states.keys()):
            raise ValueError("Unknown keys specified in `initial_states`")
        if sum(initial_states.values()) != 1:
            raise ValueError("Initial state values don't sum up to 1")

    def _calc_(self, time: np.ndarray, initial_states: dict):
        self.check_initial_states(initial_states)
        tmp = [
            (v, self.indices[i], self.indices[j])
            for i, d in self.rate_equations.items()
            for j, v in d.items()
        ]
        dt = time[1] - time[0]
        data, row_ind, col_ind = zip(*tmp)
        propagator = sp.sparse.linalg.expm(
            sp.sparse.csc_matrix((data, (row_ind, col_ind))) * dt
        )
        self.result = np.zeros([len(time), len(self.all_keys)], dtype=float)
        self.result[0] = [initial_states.get(k, 0) for k in self.all_keys]
        for t in range(1, len(time)):
            self.result[t] = propagator @ self.result[t - 1]

    def __getitem__(self, keys: list) -> np.ndarray:
        ks = [keys] if isinstance(keys, str) else keys
        return np.sum([self.result[:, self.indices[k]] for k in ks], axis=0)


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
    if np.linalg.norm(pos[0]) <= r_min:
        raise ValueError("Molecule starting distance is needs to be > r_min.")

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
