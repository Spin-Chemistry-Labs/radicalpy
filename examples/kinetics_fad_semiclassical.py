#! /usr/bin/env python
import matplotlib.pyplot as plt  # TODO REMOVE THIS
import numpy as np
from radicalpy.classical import Rate, RateEquations
from radicalpy.experiments import semiclassical_kinetics_mary
from radicalpy.simulation import LiouvilleSimulation, Molecule, SemiclassicalSimulation
from radicalpy.utils import Bhalf_fit

def plot_3d_results(results, factor=1e6):
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    ax.set_facecolor("none")
    ax.grid(False)
    X, Y = np.meshgrid(results["Bs"], results["ts"])
    ax.plot_surface(
        X,
        Y * factor,
        results["MARY"],
        facecolors=cmap.to_rgba(results["MARY"].real),
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel("$B_0$ (mT)", size=18)
    ax.set_ylabel("Time ($\mu s$)", size=18)
    ax.set_zlabel("$\Delta \Delta A$", size=18)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    plt.show()

def plot_bhalf_time(ts, bhalf_time, fit_error_time, factor=1e6):
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

def main():
    # Kinetic simulation of FAD at pH 2.1.
    # For FAD quenching: uncomment the three quenching kinetic parameters.

    # FAD kinetic parameters
    kex = Rate(1e4, "kex")  # groundstate excitation rate
    kfl = Rate(3.55e8, "kfl")  # fluorescence rate
    kic = Rate(1.28e9, "kic")  # internal conversion rate
    kisc = Rate(3.64e8, "kisc")  # intersystem crossing rate
    kd = Rate(3e5, "kd")  # protonated triplet to ground state
    k1 = Rate(7e6, "k1")  # protonated triplet to RP
    km1 = Rate(2.7e9, "km1")  # RP to protonated triplet
    krt = Rate(1e9, "krt")  # triplet state relaxation rate
    kbet = Rate(1.3e7, "kbet")  # singlet recombination rate
    # kr = Rate(1.7e6, "kr")  # RP relaxation rate
    pH = 2.1  # pH of the solution
    Hp = Rate(10**-pH, "H^+")  # concentration of hydrogen ions

    # Rate equations
    S0, S1, T1p, T10, T1m = "S0", "S1", "T1+", "T10", "T1-"
    SS, STp, ST0, STm = "SS", "ST+", "ST0", "ST-"
    TpS, TpTp, TpT0, TpTm = "T+S", "T+T+", "T+T0", "T+T-"
    T0S, T0Tp, T0T0, T0Tm = "T0S", "T0T+", "T0T0", "T0T-"
    TmS, TmTp, TmT0, TmTm = "T-S", "T-T+", "T-T0", "T-T-"

    base = {}
    base[S0] = {
        S0: -kex,
        S1: kfl + kic,
        T1p: kd,
        T10: kd,
        T1m: kd,
        SS: kbet,
    }
    base[S1] = {
        S0: kex,
        S1: -(kfl + kic + 3 * kisc),
    }
    base[T1p] = {
        S1: kisc,
        T1p: -(kd + k1 + krt),
        T10: krt,
        TpTp: km1 * Hp,
    }
    base[T10] = {
        S1: kisc,
        T1p: krt,
        T10: -(kd + k1 + 2 * krt),
        T1m: krt,
        T0T0: km1 * Hp,
    }
    base[T1m] = {
        S1: kisc,
        T10: krt,
        T1m: -(kd + k1 + krt),
        TmTm: km1 * Hp,
    }
    base[SS] = {
        SS: -(kbet),
    }
    base[STp] = {
        STp: -(kbet + km1 * Hp) / 2,
    }
    base[ST0] = {
        ST0: -(kbet + km1 * Hp) / 2,
    }
    base[STm] = {
        STm: -(kbet + km1 * Hp) / 2,
    }
    base[TpS] = {
        TpS: -(kbet + km1 * Hp) / 2,
    }
    base[TpTp] = {
        T1p: k1,
        TpTp: -km1 * Hp,
    }
    base[TpT0] = {
        TpT0: -km1 * Hp,
    }
    base[TpTm] = {
        TpTm: -km1 * Hp,
    }
    base[T0S] = {
        T0S: -(kbet + km1 * Hp) / 2,
    }
    base[T0Tp] = {
        T0Tp: -km1 * Hp,
    }
    base[T0T0] = {
        T10: k1,
        T0T0: -km1 * Hp,
    }
    base[T0Tm] = {
        T0Tm: -km1 * Hp,
    }
    base[TmS] = {
        TmS: -(kbet + km1 * Hp) / 2,
    }
    base[TmTp] = {
        TmTp: -km1 * Hp,
    }
    base[TmT0] = {
        TmT0: -km1 * Hp,
    }
    base[TmTm] = {
        T1m: k1,
        TmTm: -km1 * Hp,
    }

    rate_eq = RateEquations(base)
    mat = rate_eq.matrix.todense()
    rho0 = np.array([0, 0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    time = np.arange(0, 10e-6, 10e-9)
    Bs = np.arange(0, 40, 1)

    flavin = Molecule.all_nuclei("flavin_anion")
    adenine = Molecule.all_nuclei("adenine_cation")
    sim = SemiclassicalSimulation([flavin, adenine])

    num_samples = 400
    results = semiclassical_kinetics_mary(
        sim,
        num_samples,
        rho0,
        ts=time,
        Bs=Bs,
        D=0,
        J=0,
        kinetics=mat,
        relaxations=[]
    )

    plot_3d_results(results, factor=1e6)

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

    plot_bhalf_time(time, bhalf_time, fit_error_time)


if __name__ == "__main__":
    main()
