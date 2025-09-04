#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy import relaxation
from radicalpy.experiments import odmr
from radicalpy.plot import plot_general
from radicalpy.simulation import State
from radicalpy.utils import is_fast_run


def main(Bmax=28, Bmin=14, dB=0.2, tmax=3e-6, dt=10e-9):
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["N5"])  # , "H27", "H29"])
    trp = rp.simulation.Molecule.fromdb("tryptophan_cation", [])  # , "Hbeta1"])
    sim = rp.simulation.LiouvilleSimulation([flavin, trp])
    time = np.arange(0, tmax, dt)
    B0 = 21.6
    B1 = 0.3
    B1_freq = np.arange(Bmin, Bmax, dB)
    D = 0
    J = 0
    krec = 1.1e7
    kesc = 7e6
    kSTD = 1e8
    kr = 7e7

    results = odmr(
        sim,
        init_state=State.TRIPLET,
        obs_state=State.SINGLET,
        time=time,
        D=D,
        J=J,
        B0=B0,
        B1=B1,
        B1_freq=B1_freq,
        kinetics=[
            rp.kinetics.Haberkorn(krec, State.SINGLET),
            rp.kinetics.HaberkornFree(kesc),
        ],
        relaxations=[
            relaxation.SingletTripletDephasing(kSTD),
            relaxation.RandomFields(kr),
        ],
    )

    B0 = 0

    results2 = odmr(
        sim,
        init_state=State.TRIPLET,
        obs_state=State.SINGLET,
        time=time,
        D=D,
        J=J,
        B0=B0,
        B1=B1,
        B1_freq=B1_freq,
        kinetics=[
            rp.kinetics.Haberkorn(krec, State.SINGLET),
            rp.kinetics.HaberkornFree(kesc),
        ],
        relaxations=[
            relaxation.SingletTripletDephasing(kSTD),
            relaxation.RandomFields(kr),
        ],
    )

    odmr_mfe = (
        (results["product_yields"] - results2["product_yields"])
        / results2["product_yields"]
    ) * 100

    colours = plt.colormaps.get_cmap("cividis").resampled(len(time)).colors

    plt.figure(1)
    for i in range(0, len(time), 1):
        plt.plot(
            rp.utils.mT_to_MHz(B1_freq),
            odmr_mfe[:, i],
            "-",
            linewidth=3,
            color=colours[i],
        )
    plt.xlabel(r"Frequency / MHz", size=14)
    plt.ylabel(r"ODMR / %", size=14)
    plt.legend()
    plt.tick_params(labelsize=12)
    plt.gcf().set_size_inches(10, 5)
    plt.show()
    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(Bmax=10, Bmin=0, dB=2, tmax=1e-6, dt=10e-8)
    else:
        main()
