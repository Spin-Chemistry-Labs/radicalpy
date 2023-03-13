#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp


def main():
    # Kinetic simulation of FAD at pH 2.3.
    # For FAD quenching: uncomment the three quenching kinetic parameters.

    # FAD kinetic parameters
    kex = 1e4  # groundstate excitation rate
    kfl = 3.55e8  # fluorescence rate
    kic = 1.28e9  # internal conversion rate
    kisc = 3.64e8  # intersystem crossing rate
    khfc = 8e7  # spin-state mixing rate
    kd = 3e5  # protonated triplet to ground state
    k1 = 7e6  # protonated triplet to RP
    km1 = 2.7e9  # RP to protonated triplet
    krt = 1e9  # triplet state relaxation rate
    kbet = 1.3e7  # singlet recombination rate
    kr = 1.7e6  # RP relaxation rate
    pH = 2.3  # pH of the solution
    Hp = 10 ** (-1 * pH)  # concentration of hydrogen ions

    # Quenching kinetic parameters
    kq = 0  # 1e9  # quenching rate
    kp = 0  # 3.3e3  # free radical recombination
    Q = 0  # 1e-3  # quencher concentration

    # Rate equations
    base = {}
    base["S0"] = {
        "S0": -kex,
        "T+/-": kd,
        "T0": kd,
        "S": kbet,
        "S*": kfl + kic,
        "Quencher": kp,
    }
    base["S*"] = {
        "S*": -(kfl + kic + 3 * kisc),
        "S0": kex,
    }
    base["T*+/-"] = {
        "T*+/-": -(kd + k1 + krt),
        "T+/-": km1 * Hp,
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
        "S0": 0,
        "S*": 0,
        "T*+/-": 0,
        "T*0": 0,
        "S": 0,
        "T+/-": 2 / 3,
        "T0": 1 / 3,
        "Quencher": 0,
    }
    time = np.linspace(0, 6e-6, 200)

    rates_off = {**base, **off}
    rates_on = {**base, **on}
    result_off = rp.classical.kinetics(time, initial_states, rates_off)
    result_on = rp.classical.kinetics(time, initial_states, rates_on)
    fac = 0.07

    triplet_off = result_off[:, 2] + result_off[:, 3]
    radical_pair_off = result_off[:, 4] + result_off[:, 5] + result_off[:, 6]
    free_radical_off = result_off[:, 7]
    field_off = fac * (radical_pair_off + 2 * triplet_off + free_radical_off)

    triplet_on = result_on[:, 2] + result_on[:, 3]
    radical_pair_on = result_on[:, 4] + result_on[:, 5] + result_on[:, 6]
    free_radical_on = result_on[:, 7]
    field_on = fac * (radical_pair_on + 2 * triplet_on + free_radical_on)
    delta_delta_A = field_on - field_off

    fluor_off = result_off[:, 0]
    fluor_on = result_on[:, 0]
    fluor_del_A = fluor_on - fluor_off

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

    plt.clf()
    fig = plt.figure()
    scale = 1e6
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("FAD (pH 2.3) Fluorescence", size=18)
    axs[0].plot(time * scale, fluor_off, color="blue", linewidth=2)
    axs[0].plot(time * scale, fluor_on, color="green", linewidth=2)
    axs[1].plot(time * scale, fluor_del_A, color="orange", linewidth=2)
    plt.xscale("linear")
    axs[0].legend([r"$F (B_0 = 0)$", r"$F (B_0 \neq 0)$"])
    axs[1].set_xlabel("Time ($\mu s$)", size=14)
    axs[0].set_ylabel("$F$", size=14)
    axs[1].set_ylabel("$\Delta F$", size=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
