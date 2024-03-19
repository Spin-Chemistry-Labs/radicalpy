#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.classical import RateEquations


def main():
    # Kinetic simulation of intermolecular magnetic field effects.
    # The examples given here are FMN-HEWL and FMN-Trp

    # kinetic parameters
    kex = 1.36e4  # groundstate excitation rate
    kds = 1.09e8  # excited singlet state decay kinetics (fluorescence + IC)
    kisc = 1.09e8  # intersystem crossing rate
    ket = 1.2e9  # 1/M/s
    kdt = 3.85e5  # excited triplet state decay kinetics
    ksep = 2e8  # geminate RP to free radical separation
    khfc = 8e7  # ST-mixing rate
    kr = 2e6  # RP relaxation rate
    kre = 1.87e10  # re-encounter of free radicals to form geminate RPs
    kbet = 1e8  # spin selective reverse electron transfer of RP to groundstate
    ka = 0.7  # FMN/lysozyme acceptor recombination rate
    kd = 4.9  # FMN/lysozyme donor recombination rate
    # ka = 70  # FMN/Trp
    # kd = 12  # FMN/Trp

    # Rate equations
    base = {}
    base["A"] = {
        "A": -kex,
        "A*": kds,
        "AT*": kdt,
        "S": kbet,
        "AFR": ka,
    }
    base["A*"] = {
        "A*": -(kisc + kds),
        "A": kex,
    }
    base["AT*"] = {
        "AT*": -(ket + kdt),
        "A*": kisc,
    }
    base["AFR"] = {
        "AFR": -(kre + ka),
        "S": ksep,
        "T+": ksep,
        "T0": ksep,
        "T-": ksep,
    }
    base["D"] = {
        "D": -ket,
        "S": kbet,
        "DFR": kd,
    }
    base["DFR"] = {
        "DFR": -(kre + kd),
        "S": ksep,
        "T+": ksep,
        "T0": ksep,
        "T-": ksep,
    }

    off = {}
    off["S"] = {
        "S": -(3 * khfc + kbet + ksep),
        "T+": khfc,
        "T0": khfc,
        "T-": khfc,
        "DFR": 1 / 4 * kre,
    }
    off["T+"] = {
        "T+": -(2 * khfc + ksep),
        "S": khfc,
        "T0": khfc,
        "DFR": 1 / 4 * kre,
        "D": 1 / 3 * ket,
    }
    off["T0"] = {
        "T0": -(3 * khfc + ksep),
        "S": khfc,
        "T+": khfc,
        "T-": khfc,
        "DFR": 1 / 4 * kre,
        "D": 1 / 3 * ket,
    }
    off["T-"] = {
        "T-": -(2 * khfc + ksep),
        "S": khfc,
        "T0": khfc,
        "DFR": 1 / 4 * kre,
        "D": 1 / 3 * ket,
    }

    on = {}
    on["S"] = {
        "S": -(khfc + 2 * kr + kbet + ksep),
        "T+": kr,
        "T0": khfc,
        "T-": kr,
        "DFR": 1 / 4 * kre,
    }
    on["T+"] = {
        "T+": -(2 * kr + ksep),
        "S": kr,
        "T0": kr,
        "DFR": 1 / 4 * kre,
        "D": 1 / 3 * ket,
    }
    on["T0"] = {
        "T0": -(khfc + 2 * kr + ksep),
        "S": khfc,
        "T+": kr,
        "T-": kr,
        "DFR": 1 / 4 * kre,
        "D": 1 / 3 * ket,
    }
    on["T-"] = {
        "T-": -(2 * kr + ksep),
        "S": kr,
        "T0": kr,
        "DFR": 1 / 4 * kre,
        "D": 1 / 3 * ket,
    }

    initial_states = {
        "T+": 1 / 3,
        "T0": 1 / 3,
        "T-": 1 / 3,
    }
    time = np.linspace(0, 1e-3, 2000000)

    result_off = RateEquations({**base, **off}, time, initial_states)
    result_on = RateEquations({**base, **on}, time, initial_states)

    fluor_field_off = result_off["A"]
    fluor_field_on = result_on["A"]
    fluor_delta_A = fluor_field_on - fluor_field_off
    keys = ["S", "T+", "T0", "T-"]
    rp_field_off = result_off[keys]
    rp_field_on = result_on[keys]
    rp_delta_delta_A = rp_field_on - rp_field_off

    plt.clf()
    fig = plt.figure()
    scale = 1e9
    t = 600
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("FMN-HEWL Transient Absorption", size=18)
    axs[0].plot(time[0:t] * scale, rp_field_off[0:t], color="blue", linewidth=2)
    axs[0].plot(time[0:t] * scale, rp_field_on[0:t], color="green", linewidth=2)
    axs[1].plot(time[0:t] * scale, rp_delta_delta_A[0:t], color="orange", linewidth=2)
    plt.xscale("linear")
    axs[0].legend([r"$F (B_0 = 0)$", r"$F (B_0 \neq 0)$"])
    axs[1].set_xlabel("Time ($ns$)", size=14)
    axs[0].set_ylabel("$\Delta A$", size=14)
    axs[1].set_ylabel("$\Delta \Delta A$", size=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)

    plt.clf()
    fig = plt.figure()
    scale = 1e3
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("FMN-HEWL Fluorescence", size=18)
    axs[0].plot(time * scale, fluor_field_off, color="blue", linewidth=2)
    axs[0].plot(time * scale, fluor_field_on, color="green", linewidth=2)
    axs[1].plot(time * scale, fluor_delta_A, color="orange", linewidth=2)
    plt.xscale("linear")
    axs[0].legend([r"$F (B_0 = 0)$", r"$F (B_0 \neq 0)$"])
    axs[1].set_xlabel("Time ($ms$)", size=14)
    axs[0].set_ylabel("$F$", size=14)
    axs[1].set_ylabel("$\Delta F$", size=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
