#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


def get_rot(phi, theta):
    return np.array(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )


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
        new_pos = pos[i - 1] + delta_r * get_rot(phi, theta)
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
