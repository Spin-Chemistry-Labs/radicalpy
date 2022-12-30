#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from radicalpy.estimations import (
    T1_relaxation_rate,
    T2_relaxation_rate,
    aqueous_glycerol_viscosity,
    rotational_correlation_time_for_protein,
)


def main():
    # Calculate the effect of different glycerol-water ratios at various temperatures.
    ratio = np.arange(0, 0.5, 0.01)
    temperatures = [5, 25, 37]
    eta = {}
    for t in temperatures:
        eta[t] = aqueous_glycerol_viscosity(ratio, t)

    f = 1e3
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for t in temperatures:
        plt.plot(ratio, eta[t] * f, label=f"{t}$^\circ$C")
    plt.xlabel("Glycerol fraction", size=14)
    plt.ylabel("Viscosity ($mN \, s \, m^{-2}$)", size=14)
    plt.legend()
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)

    # Calculate the rotational correlation time for glucose oxidase (160 kDa) at various temperatures and viscosities.
    Mr = 160
    tau_c = {}
    base = 273
    for t in temperatures:
        tau_c[t] = rotational_correlation_time_for_protein(Mr, base + t, eta[t])

    f2 = 1e6
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for t in temperatures:
        plt.plot(ratio, tau_c[t] * f2, label=f"{t}$^\circ$C")
    plt.xlabel("Glycerol fraction", size=14)
    plt.ylabel(r"$\tau _C$ ($\mu s$)", size=14)
    plt.legend()
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)

    # Calculate the T1 and T2 relaxation times for glucose oxidase at various temperatures in the geomagnetic field.
    g = [
        2.00429,
        2.00389,
        2.00216,
    ]  # Nohr et al. Methods in Enzymology, 620, 251-275, 2019.
    B = 50e-6

    t1, t2 = {}, {}
    for t in temperatures:
        t1[t] = T1_relaxation_rate(g, B, tau_c[t])
        t2[t] = T2_relaxation_rate(g, B, tau_c[t])

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for t in temperatures:
        plt.plot(ratio, 1 / t1[t], label=f"{t}$^\circ$C")
    plt.xlabel("Glycerol fraction", size=14)
    plt.ylabel(r"1 / T1 ($s$)", size=14)
    plt.legend()
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{2}.png"
    plt.savefig(path)

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    for t in temperatures:
        plt.plot(ratio, 1 / t2[t], label=f"{t}$^\circ$C")
    plt.xlabel("Glycerol fraction", size=14)
    plt.ylabel(r"1 / T2 ($s$)", size=14)
    plt.legend()
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
