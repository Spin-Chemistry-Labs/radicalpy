#! /usr/bin/env python
import matplotlib.pyplot as plt  # TODO REMOVE THIS
import numpy as np
from scipy.io import loadmat  # # TODO REMOVE THIS!

from radicalpy.classical import Rate, RateEquations


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

    initial_states = {
        T1p: 1 / 3,
        T10: 1 / 3,
        T1m: 1 / 3,
    }
    time = np.linspace(0, 6e-6, 200)

    rate_eq = RateEquations(base)
    mat = rate_eq.matrix.todense()

    return  #############
    plt.clf()
    fig = plt.figure()
    scale = 1e6
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("FAD (pH 2.3) Transient Absorption", size=18)
    axs[0].plot(time * scale, data, color="blue", linewidth=2)
    plt.xscale("linear")
    axs[0].legend([r"$F (B_0 = 0)$", r"$F (B_0 \neq 0)$"])
    axs[1].set_xlabel("Time ($\mu s$)", size=14)
    axs[0].set_ylabel("$\Delta A$", size=14)
    axs[1].set_ylabel("$\Delta \Delta A$", size=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
