#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp


def main():
    np.random.seed(42)

    # t = np.arange(0, 8e-6, 40e-12)
    t = np.arange(0, 50e-9, 40e-12)
    r_min = 0.5e-9 / 2
    r_max = 2e-9 / 2
    r_max = 1.5e-9
    r = (r_min) + np.random.sample() * ((r_max) - (r_min))
    x0, y0, z0 = r, 0, 0
    mutual_diffusion = 1e-6 / 10000

    delta_r = rp.classical.get_delta_r(mutual_diffusion, t[1] - t[0])
    pos, dist, ang = rp.classical.randomwalk_3d(
        len(t), x0, y0, z0, delta_r, r_min, r_max
    )

    rp.plot.monte_carlo_caged(pos, r_max)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)

    J = rp.estimations.exchange_interaction_in_solution_MC(dist)
    D = rp.estimations.dipolar_interaction_MC(dist, ang)

    t_convert = 1e-6

    # plt.set_facecolor("none")
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(t / t_convert, dist * 1e9, "r")
    plt.title("Time evolution of radical pair separation", size=16)
    plt.xlabel("$t$ ($\mu s$)", size=14)
    plt.ylabel("$r$ (nm)", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)

    # plt.set_facecolor("none")
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(t / t_convert, J)
    plt.title("Time evolution of the exchange interaction", size=16)
    plt.xlabel("$t$ ($\mu s$)", size=14)
    plt.ylabel("$J$ (mT)", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{2}.png"
    plt.savefig(path)

    # plt.facecolor("none")
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(t / t_convert, D, "g")
    plt.title("Time evolution of the dipolar interaction", size=16)
    plt.xlabel("$t$ ($\mu s$)", size=14)
    plt.ylabel("$D$ (mT)", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path)

    acf_j = rp.utils.autocorrelation(J, factor=2)

    t = np.linspace(0, t[-1], len(acf_j))

    # ax.set_facecolor("none")
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.xscale("log")
    plt.rc("axes", edgecolor="k")
    plt.plot(t, acf_j, "b", label="J")
    plt.xlabel(r"$\tau$ (s)", size=14)
    plt.ylabel(r"$g_J(\tau)$", size=14)
    plt.title("Autocorrelation: exchange interaction", size=16)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{4}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
