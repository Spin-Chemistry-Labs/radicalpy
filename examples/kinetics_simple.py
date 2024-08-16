#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.classical import Rate, RateEquations


def main():
    # Simple example of a RP for the paper.

    # kinetic parameters
    ke = Rate(1e6, "k_{E}")  # geminate RP to free radical separation
    kst = Rate(8e7, "k_{ST}")  # ST-mixing rate
    krlx = Rate(2e6, "k_{Rlx}")  # RP relaxation rate
    kr = Rate(1e8, "k_{R}")  # reverse electron transfer of RP to groundstate

    # Rate equations
    S, Tp, T0, Tm, GS, FR = "S", "T_+", "T_0", "T_-", "GS", "FR"
    off = {}
    off[S] = {S: -(3 * kst + kr + ke), Tp: kst, T0: kst, Tm: kst}
    off[Tp] = {Tp: -(2 * kst + ke), S: kst, T0: kst}
    off[T0] = {T0: -(3 * kst + ke), S: kst, Tp: kst, Tm: kst}
    off[Tm] = {Tm: -(2 * kst + ke), S: kst, T0: kst}
    off[GS] = {S: kr}
    off[FR] = {S: ke, Tp: ke, T0: ke, Tm: ke}

    on = {}
    on[S] = {S: -(kst + 2 * krlx + kr + ke), Tp: krlx, T0: kst, Tm: krlx}
    on[Tp] = {Tp: -(2 * krlx + ke), S: krlx, T0: krlx}
    on[T0] = {T0: -(kst + 2 * krlx + ke), S: kst, Tp: krlx, Tm: krlx}
    on[Tm] = {Tm: -(2 * krlx + ke), S: krlx, T0: krlx}
    on[GS] = {S: kr}
    on[FR] = {S: ke, Tp: ke, T0: ke, Tm: ke}

    initial_states = {Tp: 1 / 3, T0: 1 / 3, Tm: 1 / 3}
    time = np.linspace(0, 1e-6, 10000)

    re_off = RateEquations(off)
    re_on = RateEquations(on)
    result_off = re_off.time_evolution(time, initial_states)
    result_on = re_on.time_evolution(time, initial_states)

    keys = [S, Tp, T0, Tm]
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
    axs[1].set_xlabel(r"Time ($\mu s$)", size=14)
    axs[0].set_ylabel(r"$\Delta A$", size=14)
    axs[1].set_ylabel(r"$\Delta \Delta A$", size=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)

    # for eq in latexify(off):
    #     print(eq)
    # print(latex_eqlist_to_align(latexify(off)))
    # reaction_scheme(__file__, on)


if __name__ == "__main__":
    main()
