#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# from . import utils
import utils


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


def plot2(pos):
    f = 1e9
    pos *= f

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "auto"})
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    ax.plot(*pos.T, alpha=0.9, color="cyan")
    ax.plot(*pos[0], "bo", markersize=15)
    ax.plot(0, 0, 0, "mo", markersize=15)
    ax.set_title(
        "3D Monte Carlo random walk simulation for a radical pair in water", size=16
    )
    ax.set_xlabel("$X$ (nm)", size=14)
    ax.set_ylabel("$Y$ (nm)", size=14)
    ax.set_zlabel("$Z$ (nm)", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)
    plt.show()


def plot_sphere(pos):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x_frame = r_max * np.outer(np.sin(theta), np.cos(phi))
    y_frame = r_max * np.outer(np.sin(theta), np.sin(phi))
    z_frame = r_max * np.outer(np.cos(theta), np.ones_like(phi))

    f = 1e9

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "auto"})
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    ax.plot_wireframe(
        x_frame * f,
        y_frame * f,
        z_frame * f,
        color="k",
        alpha=0.1,
        rstride=1,
        cstride=1,
    )
    pos = f * pos
    ax.plot(*pos.T, alpha=0.9, color="cyan")
    ax.plot(*pos[0], "bo", markersize=15)
    ax.plot(0, 0, 0, "ro", markersize=15)
    #     ax.set_title("3D Monte Carlo random walk simulation for an encapsulated radical pair", size=16)
    ax.set_xlabel("$X$ (nm)", size=14)
    ax.set_ylabel("$Y$ (nm)", size=14)
    ax.set_zlabel("$Z$ (nm)", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)
    plt.show()


if __name__ == "__main__":
    n_steps = 300
    r_max = 1.5e-9
    # r_max = 0
    x0, y0, z0 = r_max / 2, 0, 0
    mut_D = 1e-5 / 10000  # dab
    del_T = 40e-12

    # np.random.seed(42)
    delta_r = get_delta_r(mut_D, del_T)
    pos, dist, ang = randomwalk_3d(n_steps, x0, y0, z0, delta_r, r_max)
    # plot2(pos)
    plot_sphere(pos)
