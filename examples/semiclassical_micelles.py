#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.data import Molecule
from radicalpy.estimations import (
    autocorrelation_fit,
    exchange_interaction_in_solution_MC,
    k_STD,
)
from radicalpy.experiments import semiclassical_mary
from radicalpy.kinetics import Haberkorn
from radicalpy.plot import (
    plot_3d_results,
    plot_autocorrelation_fit,
    plot_bhalf_time,
    plot_exchange_interaction_in_solution,
)
from radicalpy.relaxation import SingletTripletDephasing
from radicalpy.simulation import SemiclassicalSimulation, State
from radicalpy.utils import (
    Bhalf_fit,
    autocorrelation,
    is_fast_run,
    read_trajectory_files,
)


def main(
    ts=np.arange(0, 10e-6, 10e-9),
    Bs=np.arange(0, 30, 0.1),
    num_samples=100,
):
    flavin = Molecule.all_nuclei("flavin_anion")
    trp = Molecule.all_nuclei("tryptophan_cation")
    sim = SemiclassicalSimulation([flavin, trp])

    trajectory_data = read_trajectory_files(
        "./examples/data/md_fad_trp_aot", scale=1e-10
    )
    trajectory_ts = (
        np.linspace(0, len(trajectory_data), len(trajectory_data)) * 5e-12 * 1e9
    )
    j = exchange_interaction_in_solution_MC(trajectory_data[:, 1], J0=5)

    plot_exchange_interaction_in_solution(trajectory_ts, trajectory_data, j)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    acf_j = autocorrelation(j, factor=1)
    zero_point_crossing_j = np.where(np.diff(np.sign(acf_j)))[0][0]
    t_j_max = max(trajectory_ts[:zero_point_crossing_j]) * 1e-9
    t_j = np.linspace(5e-12, t_j_max, zero_point_crossing_j)

    acf_j_fit = autocorrelation_fit(t_j, j, 5e-12, t_j_max)
    plot_autocorrelation_fit(t_j, acf_j, acf_j_fit, zero_point_crossing_j)
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    kstd = k_STD(-j, acf_j_fit["tau_c"])
    # kstd = 11681368.059456564
    triplet_excited_state_quenching_rate = 5e6
    recombination_rate = 8e6
    free_radical_escape_rate = 5e5

    results = semiclassical_mary(
        sim=sim,
        num_samples=num_samples,
        init_state=State.TRIPLET,
        ts=ts,
        Bs=Bs,
        D=0,
        J=0,
        triplet_excited_state_quenching_rate=triplet_excited_state_quenching_rate,
        free_radical_escape_rate=free_radical_escape_rate,
        kinetics=[Haberkorn(recombination_rate, State.SINGLET)],
        relaxations=[SingletTripletDephasing(kstd)],
        scale_factor=0.005,
    )

    # Calculate time evolution of the B1/2
    bhalf_time = np.zeros((len(results["MARY"])))
    fit_time = np.zeros((len(Bs), len(results["MARY"])))
    fit_error_time = np.zeros((2, len(results["MARY"])))
    R2_time = np.zeros((len(results["MARY"])))

    for i in range(2, len(results["MARY"])):
        (
            bhalf_time[i],
            fit_time[:, i],
            fit_error_time[:, i],
            R2_time[i],
        ) = Bhalf_fit(Bs, results["MARY"][i, :])

    plot_bhalf_time(ts, bhalf_time, fit_error_time)
    path = __file__[:-3] + f"_{2}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "$B_0$ / mT"
    ylabel = "Time / $\mu s$"
    zlabel = "$\Delta \Delta A$"
    plot_3d_results(
        xdata=Bs,
        ydata=ts,
        zdata=results["MARY"],
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        factor=1e6,
    )
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    if is_fast_run():
        main(num_samples=4)
    else:
        main()
