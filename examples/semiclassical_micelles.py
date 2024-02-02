#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.data import Molecule
from radicalpy.simulation import LiouvilleSimulation, State


def main(data_path="./examples/data/md_fad_trp_aot"):
    all_data = rp.utils.read_trajectory_files(data_path)

    time = np.linspace(0, len(all_data), len(all_data)) * 5e-12 * 1e9
    j = rp.estimations.exchange_interaction_in_solution_MC(all_data[:, 1], J0=5)

    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="black")
    color = "tab:red"
    plt.plot(time, all_data[:, 1] * 1e9, color=color)
    ax2 = ax.twinx()
    color2 = "tab:blue"
    plt.plot(time, -j, color=color2)
    ax.set_xlabel("Time (ns)", size=24)
    ax.set_ylabel("Radical pair separation (nm)", size=24, color=color)
    ax2.set_ylabel("Exchange interaction (mT)", size=24, color=color2)
    ax.tick_params(axis="y", labelsize=18, labelcolor=color)
    ax.tick_params(axis="x", labelsize=18, labelcolor="k")
    ax2.tick_params(labelsize=18, labelcolor=color2)
    fig.set_size_inches(7, 5)
    plt.show()

    # Calculate the autocorrelation, tau_c, and k_STD
    acf_j = rp.utils.autocorrelation(j, factor=1)
    zero_point_crossing_j = np.where(np.diff(np.sign(acf_j)))[0][0]
    t_j_max = max(time[:zero_point_crossing_j]) * 1e-9
    t_j = np.linspace(5e-12, t_j_max, zero_point_crossing_j)

    acf_j_fit = rp.estimations.autocorrelation_fit(t_j, j, 5e-12, t_j_max)
    acf_j_fit["tau_c"]
    kstd = rp.estimations.k_STD(j, acf_j_fit["tau_c"])
    k_STD = np.mean(kstd)  # singlet-triplet dephasing rate

    fig = plt.figure(2)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.xscale("log")
    # .rc("axes", edgecolor="black")
    plt.plot(t_j, acf_j[0:zero_point_crossing_j], color="tab:blue", linewidth=3)
    plt.plot(t_j, acf_j_fit["fit"], color="black", linestyle="dashed", linewidth=2)
    ax.set_xlabel(r"$\tau$ (s)", size=24)
    ax.set_ylabel(r"$g_J(\tau)$", size=24)
    plt.tick_params(labelsize=18)
    fig.set_size_inches(7, 5)
    plt.show()

    kq = 5e6  # triplet excited state quenching rate
    krec = 8e6  # recombination rate
    kesc = 5e5  # escape rate

    flavin = rp.simulation.Molecule.semiclassical_schulten_wolynes("flavin_anion")
    trp = rp.simulation.Molecule.semiclassical_schulten_wolynes("tryptophan_cation")
    sim = rp.simulation.HilbertSimulation([flavin, trp], basis="Zeeman")

    time = np.arange(0, 10e-6, 10e-9)
    Bs = np.arange(0, 50, 1)

    results = sim.MARY(
        init_state=State.TRIPLET,
        obs_state=State.TRIPLET_AND_FREE_RADICAL,
        time=time,
        B0=Bs,
        D=0,
        J=0,
        kinetics=[
            rp.kinetics.Haberkorn(krec, State.SINGLET),
            rp.kinetics.FreeRadical(kesc),
            rp.kinetics.ElectronTransfer(kq),
        ],
        relaxations=[rp.relaxation.SingletTripletDephasing(kstd)],
    )

    # Calculate time evolution of the B1/2
    bhalf_time = np.zeros((len(results)))
    fit_time = np.zeros((len(Bs), len(results)))
    fit_error_time = np.zeros((2, len(results)))
    R2_time = np.zeros((len(results)))

    for i in range(2, len(results), 1):
        (
            bhalf_time[i],
            fit_time[:, i],
            fit_error_time[:, i],
            R2_time[i],
        ) = rp.utils.Bhalf_fit(B, results["MARY"])

    # Plotting
    factor = 1e6

    plt.figure(3)
    for i in range(2, len(time), 35):
        plt.plot(time[i] * factor, bhalf_time[i], "ro", linewidth=3)
        plt.errorbar(
            time[i] * factor,
            bhalf_time[i],
            fit_error_time[1, i],
            color="k",
            linewidth=2,
        )
    plt.xlabel("Time ($\mu s$)", size=18)
    plt.ylabel("$B_{1/2}$ (mT)", size=18)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    ax.set_facecolor("none")
    ax.grid(False)
    X, Y = np.meshgrid(Bs, time)
    ax.plot_surface(
        X,
        Y * factor,
        results["MARY"],
        facecolors=cmap.to_rgba(mary_2.real),
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel("$B_0$ (mT)", size=18)
    ax.set_ylabel("Time ($\mu s$)", size=18)
    ax.set_zlabel("$\Delta \Delta A$", size=18)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
