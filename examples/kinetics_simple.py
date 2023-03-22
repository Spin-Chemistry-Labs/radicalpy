#! /usr/bin/env python
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

print(f"cwd: os.getcwd()")
sys.path.append(".")
from radicalpy.classical import RateEquations


class Rate:
    """Rate class.

    Extends float with the `Rate.label` which is the LaTeX
    representation of the rate.

    """

    value: float
    label: str
    """LaTeX representation of the rate constant."""

    def __repr__(self):
        return f"{self.label} = {self.value}"

    def __init__(self, value: float, label: str):  # noqa D102
        self.value = value
        self.label = label

    def __rmul__(self, v):
        return Rate(self.value * v, f"{v} {self.label}")

    def __mul__(self, v):
        return self.__rmul__(v)

    @staticmethod
    def _get_value_lable(v):
        return (v.value, v.label) if isinstance(v, Rate) else (v, v)

    def __radd__(self, v):
        value, label = self._get_value_lable(v)
        return Rate(self.value + value, f"{label} + {self.label}")

    def __add__(self, v):
        value, label = self._get_value_lable(v)
        return Rate(self.value + value, f"{self.label} + {label}")

    def __neg__(self):
        return Rate(-self.value, f"-({self.label})")


def latexify(rate_equations: dict):
    print(rate_equations)
    result = []
    for k, v in rate_equations.items():
        lhs = f"\\frac{{d[{k}]}}{{dt}} "
        rhs = " + ".join([f"{edge.label} [{vertex}]" for vertex, edge in v.items()])
        result.append(f"${lhs} = {rhs}$")
    return result


def main():
    # Simple example of a RP for the paper.

    # kinetic parameters
    ke = Rate(1e6, "k_\\text{E}")  # geminate RP to free radical separation
    kst = Rate(8e7, "k_\\text{ST}")  # ST-mixing rate
    krlx = Rate(2e6, "k_\\text{Rlx}")  # RP relaxation rate
    kr = Rate(1e8, "k_\\text{R}")  # reverse electron transfer of RP to groundstate

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

    for eq in latexify(off):
        print(eq)
    exit()

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

    latex = latexify(off)
    print("--------- space -----------")
    print(latex)


if __name__ == "__main__":
    main()
