#! /usr/bin/env python

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from radicalpy.classical import Rate, RateEquations, latex_eqlist_to_align, latexify


def graph(rate_equations: dict):
    data = [
        (f"${v1}$", f"${v2}$", f"${edge.label}$")
        for v1, rhs_data in rate_equations.items()
        for v2, edge in rhs_data.items()
    ]
    for t in [e for _, _, e in data]:
        print(t)
    G = nx.DiGraph([(v1, v2) for v1, v2, e in data])
    pos = nx.spring_layout(G, seed=42)
    options = {
        "font_size": 36,
        "node_size": 5000,
        # "node_color": "white",
        "edgecolors": "black",
        "linewidths": 2,
        "width": 3,
    }

    plt.clf()
    nx.draw_networkx(G, pos, with_labels=True, **options)
    # nx.draw_networkx_edges(
    #     G,
    #     pos,
    #     connectionstyle="arc3, rad=0.1",
    # )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={(v1, v2): e for v1, v2, e in data},
        verticalalignment="top",
    )
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    path = f"{__file__[:-3]}_graph.png"
    plt.savefig(path)


def main():
    # Simple example of a RP for the paper.

    # kinetic parameters
    ke = Rate(1e6, "k_{E}")  # geminate RP to free radical separation
    kst = Rate(8e7, "k_{ST}")  # ST-mixing rate
    krlx = Rate(2e6, "k_{Rlx}")  # RP relaxation rate
    kr = Rate(1e8, "k_{R}")  # reverse electron transfer of RP to groundstate

    # Rate equations
    S, Tp, T0, Tm = "S", "T_+", "T_0", "T_-"
    off = {}
    off[S] = {S: -(3 * kst + kr + ke), Tp: kst, T0: kst, Tm: kst}
    off[Tp] = {Tp: -(2 * kst + ke), S: kst, T0: kst}
    off[T0] = {T0: -(3 * kst + ke), S: kst, Tp: kst, Tm: kst}
    off[Tm] = {Tm: -(2 * kst + ke), S: kst, T0: kst}

    on = {}
    on[S] = {S: -(kst + 2 * krlx + kr + ke), Tp: krlx, T0: kst, Tm: krlx}
    on[Tp] = {Tp: -(2 * krlx + ke), S: krlx, T0: krlx}
    on[T0] = {T0: -(kst + 2 * krlx + ke), S: kst, Tp: krlx, Tm: krlx}
    on[Tm] = {Tm: -(2 * krlx + ke), S: krlx, T0: krlx}

    initial_states = {Tp: 1 / 3, T0: 1 / 3, Tm: 1 / 3}
    time = np.linspace(0, 1e-6, 10000)

    result_off = RateEquations(off, time, initial_states)
    result_on = RateEquations(on, time, initial_states)

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
    axs[1].set_xlabel("Time ($\mu s$)", size=14)
    axs[0].set_ylabel("$\Delta A$", size=14)
    axs[1].set_ylabel("$\Delta \Delta A$", size=14)
    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)

    print(latex_eqlist_to_align(latexify(off)))
    graph(off)


if __name__ == "__main__":
    main()
