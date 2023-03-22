#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from radicalpy.classical import RateEquations


def main():
    # Simple example of a RP for the paper.

    # kinetic parameters
    ke = 1e6  # geminate RP to free radical separation
    kst = 8e7  # ST-mixing rate
    krlx = 2e6  # RP relaxation rate
    kr = 1e8  # spin selective reverse electron transfer of RP to groundstate

    # Rate equations
    off = {}
    off["S"] = {"S": -(3 * kst + kr + ke), "T+": kst, "T0": kst, "T-": kst}
    off["T+"] = {"T+": -(2 * kst + ke), "S": kst, "T0": kst}
    off["T0"] = {"T0": -(3 * kst + ke), "S": kst, "T+": kst, "T-": kst}
    off["T-"] = {"T-": -(2 * kst + ke), "S": kst, "T0": kst}

    on = {}
    on["S"] = {"S": -(kst + 2 * krlx + kr + ke), "T+": krlx, "T0": kst, "T-": krlx}
    on["T+"] = {"T+": -(2 * krlx + ke), "S": krlx, "T0": krlx}
    on["T0"] = {"T0": -(kst + 2 * krlx + ke), "S": kst, "T+": krlx, "T-": krlx}
    on["T-"] = {"T-": -(2 * krlx + ke), "S": krlx, "T0": krlx}

    initial_states = {"T+": 1 / 3, "T0": 1 / 3, "T-": 1 / 3}
    time = np.linspace(0, 1e-6, 10000)

    result_off = RateEquations(off, time, initial_states)
    result_on = RateEquations(on, time, initial_states)

    keys = ["S", "T+", "T0", "T-"]
    rp_field_off = result_off[keys]
    rp_field_on = result_on[keys]
    rp_delta_delta_A = rp_field_on - rp_field_off

    plt.clf()
    fig = plt.figure()
    scale = 1e6
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle("Triplet born radical pair", size=18)
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


if __name__ == "__main__":
    main()
