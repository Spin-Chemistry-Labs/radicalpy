#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.classical import Rate, RateEquations


def main():
    # Kinetic simulation of FAD at pH 2.1.
    # For FAD quenching: uncomment the three quenching kinetic parameters.

    # FAD kinetic parameters
    kex = Rate(1e4, "kex")  # groundstate excitation rate
    kfl = Rate(3.55e8, "kfl")  # fluorescence rate
    kic = Rate(1.28e9, "kic")  # internal conversion rate
    kisc = Rate(3.64e8, "kisc")  # intersystem crossing rate
    khfc = Rate(8e7, "khfc")  # spin-state mixing rate
    kd = Rate(3e5, "kd")  # protonated triplet to ground state
    k1 = Rate(7e6, "k1")  # protonated triplet to RP
    km1 = Rate(2.7e9, "km1")  # RP to protonated triplet
    krt = Rate(1e9, "krt")  # triplet state relaxation rate
    kbet = Rate(1.3e7, "kbet")  # singlet recombination rate
    kr = Rate(1.7e6, "kr")  # RP relaxation rate
    pH = 2.3  # pH of the solution
    Hp = Rate(10 ** (-1 * pH), "H^+")  # concentration of hydrogen ions

    # Quenching kinetic parameters
    kq = Rate(0, "kq")  # 1e9  # quenching rate
    kp = Rate(0, "kp")  # 3.3e3  # free radical recombination
    Q = Rate(0, "Q")  # 1e-3  # quencher concentration

    # Rate equations
    S0, S1, T1p, T10, T1m, FR = "S0", "S1", "T1+", "T10", "T1-", "FR"
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
        FR: kp,
    }
    base[S1] = {
        S1: -(kfl + kic + 3 * kisc),
        S0: kex,
    }
    base[T1p] = {
        T1p: -(kd + k1 + krt),
        Tp: km1 * Hp,
        "T*0": 2 * krt,
        "S*": 2 * kisc,
    }
    base["T*0"] = {
        "T*0": -(kd + k1 + 2 * krt),
        "T0": km1 * Hp,
        "T*+/-": krt,
        "S*": kisc,
    }
    base["Quencher"] = {
        "Quencher": -kp,
        "S": kq * Q,
        "T+/-": kq * Q,
        "T0": kq * Q,
    }

    off = {}
    off["S"] = {
        "S": -(3 * khfc + kbet),
        "T+/-": khfc,
        "T0": khfc,
    }
    off["T+/-"] = {
        "T+/-": -(2 * khfc + km1 * Hp),
        "S": 2 * khfc,
        "T0": 2 * khfc,
        "T*+/-": k1,
    }
    off["T0"] = {
        "T0": -(3 * khfc + km1 * Hp),
        "S": khfc,
        "T+/-": khfc,
        "T*0": k1,
    }

    on = {}
    on["S"] = {
        "S": -(2 * kr + khfc + kbet),
        "T+/-": kr,
        "T0": khfc,
    }
    on["T+/-"] = {
        "T+/-": -(2 * kr + km1 * Hp),
        "S": 2 * kr,
        "T0": 2 * kr,
        "T*+/-": k1,
    }
    on["T0"] = {
        "T0": -(2 * kr + khfc + km1 * Hp),
        "S": khfc,
        "T+/-": kr,
        "T*0": k1,
    }

    initial_states = {
        "T+/-": 2 / 3,
        "T0": 1 / 3,
    }
    time = np.linspace(0, 6e-6, 200)

    result_off = RateEquations({**base, **off}, time, initial_states)
    result_on = RateEquations({**base, **on}, time, initial_states)
    fac = 0.07

    keys = ["S", "T+/-", "T0", "Quencher"] + 2 * ["T*+/-", "T*0"]
    field_off = fac * result_off[keys]
    field_on = fac * result_on[keys]
    delta_delta_A = field_on - field_off

    plt.clf()
    fig = plt.figure()
    scale = 1e6
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("FAD (pH 2.3) Transient Absorption", size=18)
    axs[0].plot(time * scale, field_off, color="blue", linewidth=2)
    axs[0].plot(time * scale, field_on, color="green", linewidth=2)
    axs[1].plot(time * scale, delta_delta_A, color="orange", linewidth=2)
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
