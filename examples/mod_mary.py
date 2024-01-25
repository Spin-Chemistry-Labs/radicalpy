#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.experiments import modulated_mary_brute_force


def main(
    Bs = np.linspace(-5, 5, 500),
    modulation_depths = [2, 1.5, 1, 0.5, 0.1],
    modulation_frequency = 3,
    time_constant = 0.3,
    harmonics = [1, 2, 3, 4, 5],
    lfe_magnitude = 0.02,
):
    S = modulated_mary_brute_force(
        Bs,
        modulation_depths,
        modulation_frequency,
        time_constant,
        harmonics,
        lfe_magnitude)


    harmonic = 0

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for i, md in enumerate(modulation_depths):
        plt.plot(Bs, S[harmonic, i, :], label=f"{md} G", linewidth=3)
    plt.legend()
    plt.xlabel(r"$B_0$ / G", size=14)
    plt.ylabel(r"ModMARY signal / au", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)

    harmonic = 1

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for i, md in enumerate(modulation_depths):
        plt.plot(Bs, S[harmonic, i, :], label=f"{md} G", linewidth=3)
    plt.legend()
    plt.xlabel(r"$B_0$ / G", size=14)
    plt.ylabel(r"ModMARY signal / au", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)

    harmonic = 2

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for i, md in enumerate(modulation_depths):
        plt.plot(Bs, S[harmonic, i, :], label=f"{md} G", linewidth=3)
    plt.legend()
    plt.xlabel(r"$B_0$ / G", size=14)
    plt.ylabel(r"ModMARY signal / au", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{2}.png"
    plt.savefig(path)

    harmonic = 3

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for i, md in enumerate(modulation_depths):
        plt.plot(Bs, S[harmonic, i, :], label=f"{md} G", linewidth=3)
    plt.legend()
    plt.xlabel(r"$B_0$ / G", size=14)
    plt.ylabel(r"ModMARY signal / au", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path)

    harmonic = 4

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for i, md in enumerate(modulation_depths):
        plt.plot(Bs, S[harmonic, i, :], label=f"{md} G", linewidth=3)
    plt.legend()
    plt.xlabel(r"$B_0$ / G", size=14)
    plt.ylabel(r"ModMARY signal / au", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{4}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
