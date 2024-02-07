#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import testing as npt  # ####################################

import radicalpy as rp
from radicalpy.data import Molecule
from radicalpy.estimations import (autocorrelation, autocorrelation_fit,
                                   exchange_interaction_in_solution_MC, k_STD)
from radicalpy.experiments import semiclassical_mary
from radicalpy.kinetics import Haberkorn
from radicalpy.relaxation import SingletTripletDephasing
from radicalpy.simulation import SemiclassicalSimulation, State
from radicalpy.utils import Bhalf_fit, read_trajectory_files


def plot_exchange_interaction_in_solution(ts, trajectory_data, j):
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="black")
    color = "tab:red"
    plt.plot(ts, trajectory_data[:, 1] * 1e9, color=color)
    ax2 = ax.twinx()
    color2 = "tab:blue"
    plt.plot(ts, -j, color=color2)
    ax.set_xlabel("Time (ns)", size=24)
    ax.set_ylabel("Radical pair separation (nm)", size=24, color=color)
    ax2.set_ylabel("Exchange interaction (mT)", size=24, color=color2)
    ax.tick_params(axis="y", labelsize=18, labelcolor=color)
    ax.tick_params(axis="x", labelsize=18, labelcolor="k")
    ax2.tick_params(labelsize=18, labelcolor=color2)
    fig.set_size_inches(7, 5)
    plt.show()


def plot_autocorrelation_fit(t_j, acf_j, acf_j_fit, zero_point_crossing_j):
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


def main(data_path="./examples/data/md_fad_trp_aot"):
    flavin = Molecule.all_nuclei("flavin_anion")
    trp = Molecule.all_nuclei("tryptophan_cation")
    sim = SemiclassicalSimulation([flavin, trp], basis="Zeeman")
    print(f"{sim.molecules[0].semiclassical_tau=}")
    print(f"{sim.molecules[1].semiclassical_tau=}")

    trajectory_data = read_trajectory_files(data_path)
    ts = np.linspace(0, len(trajectory_data), len(trajectory_data)) * 5e-12 * 1e9
    j = exchange_interaction_in_solution_MC(trajectory_data[:, 1], J0=5)

    # plot_exchange_interaction_in_solution(ts, trajectory_data, j)

    # Calculate the autocorrelation, tau_c, and k_STD
    acf_j = autocorrelation(j, factor=1)
    zero_point_crossing_j = np.where(np.diff(np.sign(acf_j)))[0][0]
    t_j_max = max(ts[:zero_point_crossing_j]) * 1e-9
    t_j = np.linspace(5e-12, t_j_max, zero_point_crossing_j)

    acf_j_fit = autocorrelation_fit(t_j, j, 5e-12, t_j_max)
    acf_j_fit["tau_c"]
    kstd = rp.estimations.k_STD(j, acf_j_fit["tau_c"])
    # k_STD = np.mean(kstd)  # singlet-triplet dephasing rate ##################################

    # plot_autocorrelation_fit(t_j, acf_j, acf_j_fit, zero_point_crossing_j)

    triplet_excited_state_quenching_rate = 5e6
    recombination_rate = 8e6
    free_radical_escape_rate = 5e5

    ts = np.arange(0, 10e-6, 10e-9)
    Bs = np.arange(0, 50, 1)

    results = semiclassical_mary(
        sim=sim,
        num_samples=10,
        init_state=State.TRIPLET,
        # obs_state=State.TRIPLET,
        time=ts,
        B0=Bs,
        D=0,
        J=0,
        triplet_excited_state_quenching_rate=triplet_excited_state_quenching_rate,
        free_radical_escape_rate=free_radical_escape_rate,
        kinetics=[Haberkorn(recombination_rate, State.SINGLET)],
        relaxations=[SingletTripletDephasing(kstd)],
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
        ) = Bhalf_fit(Bs, results["MARY"])

    # Plotting
    factor = 1e6

    plt.figure(3)
    for i in range(2, len(ts), 35):
        plt.plot(ts[i] * factor, bhalf_time[i], "ro", linewidth=3)
        plt.errorbar(
            ts[i] * factor,
            bhalf_time[i],
            fit_error_time[1, i],
            color="k",
            linewidth=2,
        )
    plt.xlabel("Time ($\mu s$)", size=18)
    plt.ylabel("$B_{1/2}$ (mT)", size=18)
    plt.tick_params(labelsize=14)
    plt.gcf().set_size_inches(10, 5)
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    ax.set_facecolor("none")
    ax.grid(False)
    X, Y = np.meshgrid(Bs, ts)
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
