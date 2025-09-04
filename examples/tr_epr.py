#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy import relaxation
from radicalpy.experiments import epr
from radicalpy.simulation import State
from radicalpy.utils import is_fast_run


def main(Bmax=336, Bmin=333, dB=0.01, tmax=3e-6, dt=5e-9):
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", [])  # , "H27", "H29"])
    trp = rp.simulation.Molecule.fromdb("tryptophan_cation", [])  # , "Hbeta1"])
    sim = rp.simulation.LiouvilleSimulation([flavin, trp])
    time = np.arange(0, tmax, dt)
    B0 = np.arange(Bmin, Bmax, dB)
    B1 = 0.01
    B1_freq = rp.utils.MHz_to_mT(9373)
    print(B1_freq)
    D = 0
    J = -0.001  # (np.sum((sim.projection_operator(state=State.SINGLET) - sim.projection_operator(state=State.TRIPLET))))
    krec = 1e7
    kesc = 1e7
    kSTD = 0
    kr = 0

    results = epr(
        sim,
        init_state=State.SINGLET,
        obs_state=State.EPR,
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

    trepr_result = results["product_yields"]

    colours = plt.colormaps.get_cmap("cividis").resampled(len(time)).colors

    plt.figure(1)
    for i in range(0, len(time), 1):
        plt.plot(
            B0,
            trepr_result[:,i],
            "-",
            linewidth=3,
            color=colours[i]
            )
    plt.xlabel(r"$B_0$ / mT", size=14)
    plt.legend()
    plt.tick_params(labelsize=18)
    plt.gcf().set_size_inches(10, 5)
    plt.show()

    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(Bmax=10, Bmin=0, dB=2, tmax=1e-6, dt=10e-8)
    else:
        main()
