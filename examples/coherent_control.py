#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.experiments import coherent_control
from radicalpy.shared import constants as C
from radicalpy.simulation import HilbertSimulation, State
from radicalpy.utils import is_fast_run, make_resonance_sticks


def main():
    TRPstick_field, TRPstick_int = make_resonance_sticks(
        freq=100e6, hfcH=(-4.0, -3.6, 13.6, 28.3), hfcN=(4.0,)
    )
    TRPstick_field_rads = 2 * np.pi * 1e6 * TRPstick_field

    gval = 2.0034
    # 1 G = 1e-4 T
    B_T = TRPstick_field * 1e-4
    # frequency in Hz: (g μB B)/h
    freq_Hz = (gval * C.mu_B * B_T) / C.h
    fields_MHz = freq_Hz / 1e6

    fig, ax = plt.subplots()
    for f, h in zip(fields_MHz, TRPstick_int):
        ax.vlines(f, 0, h, color="k")
    ax.set_xlabel("Frequency / MHz")
    ax.set_yticks([])
    plt.show()
    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)

    FMNstick_field, FMNstick_int = make_resonance_sticks(
        freq=100e6,
        hfcH=(
            -1.58,
            -1.58,
            3.9,
            3.9,
        ),
        hfcN=(
            3.93,
            2.12,
        ),
    )
    FMNstick_field_rads = 2 * np.pi * 1e6 * FMNstick_field

    gval = 2.0034
    # 1 G = 1e-4 T
    B_T = FMNstick_field * 1e-4
    # frequency in Hz: (g μB B)/h
    freq_Hz = (gval * C.mu_B * B_T) / C.h
    fields_MHz = freq_Hz / 1e6

    fig, ax = plt.subplots()
    for f, h in zip(fields_MHz, FMNstick_int):
        ax.vlines(f, 0, h, color="k")
    ax.set_xlabel("Frequency / MHz")
    ax.set_yticks([])
    plt.show()
    # path = __file__[:-3] + f"_{2}.png"
    # plt.savefig(path)

    m1 = rp.simulation.Molecule("1", [])
    m2 = rp.simulation.Molecule("2", [])
    sim = HilbertSimulation([m1, m2])

    time = np.arange(0, 2e-6, 1e-9)

    out = coherent_control(
        sim=sim,
        init_state=State.SINGLET,
        obs_state=State.TRIPLET,
        time=time,
        sticks_A_freq=TRPstick_field_rads,
        sticks_A_int=TRPstick_int,
        sticks_B_freq=FMNstick_field_rads,
        sticks_B_int=FMNstick_int,
        B1_G=200,
        g_e=2,
        k_s=2e6,
        J=0,
    )

    plt.plot(out["time"] * 1e6, out["u"] / (C.g_e * C.mu_B / C.hbar / 1000))
    plt.xlabel("Time / $\mu s$")
    plt.ylabel("Target population")
    plt.show()
    # path = __file__[:-3] + f"_{3}.png"
    # plt.savefig(path)

    plt.plot(out["time"] * 1e6, out["target"])
    plt.xlabel("Time / $\mu s$")
    plt.ylabel("Target population")
    plt.show()
    # path = __file__[:-3] + f"_{4}.png"
    # plt.savefig(path)

    plt.plot(out["time"] * 1e6, out["population"])
    plt.xlabel("Time / $\mu s$")
    plt.ylabel("Target population")
    plt.show()
    # path = __file__[:-3] + f"_{5}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main()
    else:
        main()
