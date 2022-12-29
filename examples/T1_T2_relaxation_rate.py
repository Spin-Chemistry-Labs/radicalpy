#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp

def main():
    # Calculate the effect of different glycerol-water ratios at various temperatures.
    ratio = np.arange(0, 0.5, 0.01)
    eta_5, eta_25, eta_37 = np.zeros(len(ratio)), np.zeros(len(ratio)), np.zeros(len(ratio))
    for i, r in enumerate(ratio):
        eta_5[i] = rp.estimations.aqueous_glycerol_viscosity(r, 5)
        eta_25[i] = rp.estimations.aqueous_glycerol_viscosity(r, 25)
        eta_37[i] = rp.estimations.aqueous_glycerol_viscosity(r, 37)

    f = 1e3
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(ratio, eta_5 * f, label="5$^\circ$C")
    plt.plot(ratio, eta_25 * f, label="25$^\circ$C")
    plt.plot(ratio, eta_37 * f, label="37$^\circ$C")
    plt.xlabel("Glycerol fraction", size=14)
    plt.ylabel("Viscosity ($mN s m^{-2}$)", size=14)
    plt.legend()
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


    # Calculate the rotational correlation time for glucose oxidase (160 kDa) at various temperatures and viscosities.
    Mr = 160
    tauc_5 = rp.estimations.rotational_correlation_time_for_protein(Mr, 278, eta_5)
    tauc_25 = rp.estimations.rotational_correlation_time_for_protein(Mr, 298, eta_25)
    tauc_37 = rp.estimations.rotational_correlation_time_for_protein(Mr, 310, eta_37)

    f2 = 1e6
    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(ratio, tauc_5 * f2, label="5$^\circ$C")
    plt.plot(ratio, tauc_25 * f2, label="25$^\circ$C")
    plt.plot(ratio, tauc_37 * f2, label="37$^\circ$C")
    plt.xlabel("Glycerol fraction", size=14)
    plt.ylabel(r"$\tau _C$ ($\mu s$)", size=14)
    plt.legend()
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)


    # Calculate the T1 and T2 relaxation times for glucose oxidase at various temperatures in the geomagnetic field.
    g = [2.00429, 2.00389, 2.00216] # Nohr et al. Methods in Enzymology, 620, 251-275, 2019.
    B = 50e-6

    t1_5 = rp.estimations.T1_relaxation_rate(g, B, tauc_5)
    t1_25 = rp.estimations.T1_relaxation_rate(g, B, tauc_25)
    t1_37 = rp.estimations.T1_relaxation_rate(g, B, tauc_37)

    t2_5 = rp.estimations.T2_relaxation_rate(g, B, tauc_5)
    t2_25 = rp.estimations.T2_relaxation_rate(g, B, tauc_25)
    t2_37 = rp.estimations.T2_relaxation_rate(g, B, tauc_37)

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(ratio, 1 / t1_5, label="5$^\circ$C")
    plt.plot(ratio, 1 / t1_25, label="25$^\circ$C")
    plt.plot(ratio, 1 / t1_37, label="37$^\circ$C")
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
    plt.plot(ratio, 1 / t2_5, label="5$^\circ$C")
    plt.plot(ratio, 1 / t2_25, label="25$^\circ$C")
    plt.plot(ratio, 1 / t2_37, label="37$^\circ$C")
    plt.xlabel("Glycerol fraction", size=14)
    plt.ylabel(r"1 / T2 ($s$)", size=14)
    plt.legend()
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path)

if __name__ == "__main__":
    main()
