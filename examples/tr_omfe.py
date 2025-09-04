#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy import relaxation
from radicalpy.experiments import omfe
from radicalpy.simulation import State
from radicalpy.utils import is_fast_run


def main(tmax=3e-6, dt=5e-9):
    # radical1 = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.8])
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["N5"])  # , "H27", "H29"])
    trp = rp.simulation.Molecule.fromdb("tryptophan_cation", [])  # , "Hbeta1"])
    sim = rp.simulation.HilbertSimulation([flavin, trp])

    a = sim.nuclei[0].hfc.isotropic  # mT
    time = np.arange(0, tmax, dt)
    dB = 0.01
    B1 = 0.1
    Bmin = rp.utils.MHz_to_mT(0)
    Bmax = rp.utils.MHz_to_mT(80)
    print(Bmax)
    B1_freq = np.arange(Bmin, Bmax, dB)
    k = 2.8e6

    results = omfe(
        sim,
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        D=0,
        J=0,
        B1=B1,
        B1_freq=B1_freq,
        B1_axis="x",
        B1_freq_axis="x",
        kinetics=[rp.kinetics.Exponential(k)],
    )

    B1 = 0

    results2 = omfe(
        sim,
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        D=0,
        J=0,
        B1=B1,
        B1_freq=B1_freq,
        B1_axis="x",
        B1_freq_axis="x",
        kinetics=[rp.kinetics.Exponential(k)],
    )

    omfe_mfe = (
        (results["product_yields"] - results2["product_yields"])
        / results2["product_yields"]
    ) * 100
    n = len(time)
    colours = plt.colormaps.get_cmap("cividis").resampled(n).colors

    plt.figure(1)
    for i in range(0, n, 1):
        plt.plot(B1_freq / a, omfe_mfe[:, i], "-", linewidth=3, color=colours[i])
    plt.xlabel(r"$B_1$ / a", size=14)
    plt.ylabel(r"OMFE / %", size=14)
    plt.legend()
    plt.tick_params(labelsize=12)
    plt.gcf().set_size_inches(10, 5)
    plt.show()

    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(tmax=1e-6, dt=10e-8)
    else:
        main()
