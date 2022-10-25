#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# from . import utils
import utils


def MC_randomwalk3D_cage(n_steps, r_max, x_0, y_0, z_0, mut_D, del_T):
    Dab = mut_D
    deltaT = del_T
    deltaR = np.sqrt(6 * Dab * deltaT)  # diffusional motion

    x, y, z, dist, angle = (
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps + 1),
    )
    x[0], y[0], z[0] = x_0, y_0, z_0

    for i in range(1, n_steps):
        theta = np.pi * np.random.rand()
        angle[i] = theta
        phi = 2 * np.pi * np.random.rand()

        x[i] = deltaR * np.cos(theta) * np.sin(phi)
        y[i] = deltaR * np.sin(theta) * np.sin(phi)
        z[i] = deltaR * np.cos(phi)

        dist_sq = (
            (x[i] + x[i - 1]) ** 2 + (y[i] + y[i - 1]) ** 2 + (z[i] + z[i - 1]) ** 2
        )
        dist[i] = np.sqrt(dist_sq)

        if dist_sq > r_max**2:
            x[i] = x[i - 1] - x[i]
            y[i] = y[i - 1] - y[i]
            z[i] = z[i - 1] - z[i]
        else:
            x[i] += x[i - 1]
            y[i] += y[i - 1]
            z[i] += z[i - 1]

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
    ax.plot(x * f, y * f, z * f, alpha=0.9, color="cyan")
    ax.plot(x[0] * f, y[0] * f, z[0] * f, "bo", markersize=15)
    ax.plot(0, 0, 0, "ro", markersize=15)
    #     ax.set_title("3D Monte Carlo random walk simulation for an encapsulated radical pair", size=16)
    ax.set_xlabel("$X$ (nm)", size=14)
    ax.set_ylabel("$Y$ (nm)", size=14)
    ax.set_zlabel("$Z$ (nm)", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)
    plt.show()

    return x, y, z, dist, angle


def get_delta_r(mutual_diffusion, delta_T):
    return np.sqrt(6 * mutual_diffusion * delta_T)


def randomwalk_3d(n_steps, max_r, x_0, y_0, z_0, delta_r):
    pos = np.zeros([n_steps, 3])
    dist = np.zeros(n_steps)
    angle = np.zeros(n_steps + 1)
    pos[0] = np.array([x_0, y_0, z_0])

    for i in range(1, n_steps):
        theta = np.pi * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        angle[i] = theta
        # why subtract? pos[i] == 0
        new_pos = pos[i - 1] + delta_r * utils.spherical_to_cartesian(theta, phi)
        dist = np.linalg.norm(new_pos)
        # while while dist >= max_r:
        #     new_pos = pos[i-1] + delta_r * get_rot(phi, theta)
        #     dist = np.linalg.norm(new_pos)
        pos[i] = new_pos
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


if __name__ == "__main__":
    n_steps = 1000
    r_max = 1.5e-9
    x0, y0, z0 = r_max / 2, 0, 0
    mut_D = 1e-5 / 10000  # dab
    del_T = 40e-12

    np.random.seed(42)
    pos, dist, ang = randomwalk_3d(
        n_steps, r_max, x0, y0, z0, get_delta_r(mut_D, del_T)
    )
    plot2(pos)
