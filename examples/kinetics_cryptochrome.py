#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp


def main():
    # Kinetic simulation of intramolecular magnetic field effects.
    # The example given here is cryptochrome

    # kinetic parameters
    kex = 1e3  # groundstate excitation rate
    kds = 1.09e8  # excited singlet state decay kinetics (fluorescence + IC)
    ket = 1e10  # 1/s
    krp2 = 2e6  # proton transfer from RP1 to RP2
    khfc = 8e7  # ST-mixing rate
    kr = 2e6  # RP relaxation rate
    kbet = 2e6  # spin selective reverse electron transfer of RP1 to groundstate
    kbet2 = 5e4  # spin selective reverse electron transfer of RP2 to groundstate
    ka = 90  # acceptor recombination rate
    kd = 120  # donor recombination rate

    # Rate equations
    base = {}
    base["A-D"] = {
        "A-D": -kex,
        "A*-D": kds,
        "S": kbet,
        "RP2": kbet2,
        "AR-D": ka,
        "A-DR": kd,
    }
    base["A*-D"] = {
        "A*-D": -(ket + kds),
        "A-D": kex,
    }
    base["RP2"] = {
        "RP2": -(ka + kd + kbet2),
        "S": 1 / 4 * krp2,
        "T+": 1 / 4 * krp2,
        "T0": 1 / 4 * krp2,
        "T-": 1 / 4 * krp2,
    }
    base["AR-D"] = {
        "AR-D": -ka,
        "RP2": kd,
    }
    base["A-DR"] = {
        "A-DR": -kd,
        "RP2": ka,
    }

    off = {}
    off["S"] = {
        "S": -(3 * khfc + kbet + 1 / 4 * krp2),
        "T+": khfc,
        "T0": khfc,
        "T-": khfc,
        "A*-D": ket,
    }
    off["T+"] = {
        "T+": -(2 * khfc + 1 / 4 * krp2),
        "S": khfc,
        "T0": khfc,
    }
    off["T0"] = {
        "T0": -(3 * khfc + 1 / 4 * krp2),
        "S": khfc,
        "T+": khfc,
        "T-": khfc,
    }
    off["T-"] = {
        "T-": -(2 * khfc + 1 / 4 * krp2),
        "S": khfc,
        "T0": khfc,
    }

    on = {}
    on["S"] = {
        "S": -(khfc + 2 * kr + kbet + 1 / 4 * krp2),
        "T+": kr,
        "T0": khfc,
        "T-": kr,
        "A*-D": ket,
    }
    on["T+"] = {
        "T+": -(2 * kr + 1 / 4 * krp2),
        "S": kr,
        "T0": kr,
    }
    on["T0"] = {
        "T0": -(khfc + 2 * kr + 1 / 4 * krp2),
        "S": khfc,
        "T+": kr,
        "T-": kr,
    }
    on["T-"] = {
        "T-": -(2 * kr + 1 / 4 * krp2),
        "S": kr,
        "T0": kr,
    }

    initial_states = {
        "A-D": 0,
        "A*-D": 0,
        "RP2": 0,
        "AR-D": 0,
        "A-DR": 0,
        "S": 1,
        "T+": 0,
        "T0": 0,
        "T-": 0,
    }
    time = np.linspace(0, 100e-6, 500)

    rates_off = {**base, **off}
    rates_on = {**base, **on}
    result_off = rp.classical.kinetics(time, initial_states, rates_off)
    result_on = rp.classical.kinetics(time, initial_states, rates_on)

    fluor_field_off = result_off[:, 0]
    fluor_field_on = result_on[:, 0]
    fluor_delta_A = fluor_field_on - fluor_field_off
    rp_field_off = result_off[:, 2] + result_off[:, 3]
    rp_field_on = result_on[:, 2] + result_on[:, 3]
    rp_delta_delta_A = rp_field_on - rp_field_off

    plt.clf()
    fig = plt.figure()
    scale = 1e6
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("Cryptochrome Transient Absorption", size=18)
    axs[0].plot(time * scale, rp_field_off, color="blue", linewidth=2)
    axs[0].plot(time * scale, rp_field_on, color="green", linewidth=2)
    axs[1].plot(time * scale, rp_delta_delta_A, color="orange", linewidth=2)
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
    fig.suptitle("Cryptochrome Fluorescence", size=18)
    axs[0].plot(time * scale, fluor_field_off, color="blue", linewidth=2)
    axs[0].plot(time * scale, fluor_field_on, color="green", linewidth=2)
    axs[1].plot(time * scale, fluor_delta_A, color="orange", linewidth=2)
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
