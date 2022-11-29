#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from . import estimations, utils


def get_delta_r(mutual_diffusion, delta_T):
    return np.sqrt(6 * mutual_diffusion * delta_T)


def _random_theta_phi():
    theta = np.pi * np.random.rand()
    # phi = 2 * np.pi * np.random.rand()
    arg = np.random.uniform(-1, 1)
    phi = 2 * np.sign(arg) * np.arcsin(np.sqrt(np.abs(arg)))
    return theta, phi


def randomwalk_3d(n_steps, x_0, y_0, z_0, delta_r, r_max=0):
    pos = np.zeros([n_steps, 3])
    dist = np.zeros(n_steps)
    angle = np.zeros(n_steps - 1)

    pos[0] = np.array([x_0, y_0, z_0])
    dist[0] = np.linalg.norm(pos[0])

    for i in range(1, n_steps):
        theta, phi = _random_theta_phi()
        new_pos = pos[i - 1] + delta_r * utils.spherical_to_cartesian(theta, phi)
        # _random_vector(delta_r)
        d = np.linalg.norm(new_pos)
        while r_max > 0 and d >= r_max:
            theta, phi = _random_theta_phi()
            new_pos = pos[i - 1] + delta_r * utils.spherical_to_cartesian(theta, phi)
            # _random_vector(delta_r)
            d = np.linalg.norm(new_pos)
        angle[i - 1] = theta
        pos[i] = new_pos
        dist[i] = d
    return pos, dist, angle


def monte_carlo_exchange_dipolar(n_steps, r_min, del_T, radA_x, dist, angle):

    r_min = radA_x[0]
    dist[0] = r_min
    r = dist
    r_tot = r_min + r

    theta = angle

    t_tot = n_steps * del_T * 1e9
    t = np.linspace(0, t_tot, n_steps)

    J = estimations.exchange_interaction_monte_carlo(r)
    D = estimations.dipolar_interaction_monte_carlo(r_tot, theta)

    return t, r_tot, J, D
