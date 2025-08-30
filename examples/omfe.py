#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy import relaxation
from radicalpy.experiments import omfe
from radicalpy.simulation import State
from radicalpy.utils import is_fast_run


def main(Bmin=0, tmax=3e-6, dt=10e-9):
    radical1 = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.8])
    trp = rp.simulation.Molecule.fromdb("tryptophan_cation", [])  # , "Hbeta1"])
    sim = rp.simulation.HilbertSimulation([radical1, trp])

    # gamma = sim.radicals[0].gamma_mT  # rad / s / mT
    a = sim.nuclei[0].hfc.isotropic  # mT
    time = np.arange(0, tmax, dt)
    dB = 0.05
    B1 = 0.3
    Bmax = rp.utils.MHz_to_mT(100)
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
        B1_freq_axis="z",
        kinetics=[rp.kinetics.Exponential(k)],
    )
    # MARY = results["MARY"]
    # HFE = results["HFE"]
    # LFE = results["LFE"]

    # np.save("./examples/data/fad_mary/results_5nuc_liouville_relaxation.npy", results)

    # Bhalf, fit_result, fit_error, R2 = rp.utils.Bhalf_fit(B1, MARY)

    plt.plot(
        B1_freq / a,
        results["product_yield_sums"],
        color="red",
        linewidth=2,
    )

    plt.xlabel("$\omega_{rf}$ / a")
    plt.ylabel("Singlet Yield")
    # plt.title("")
    # plt.legend([r"Simulation", r"Fit"])

    # print(f"HFE = {HFE: .2f} %")
    # print(f"LFE = {LFE: .2f} %")
    # print(f"B1/2 = {Bhalf: .2f} mT")
    # print(f"B1/2 fit error = {fit_error[1]: .2f} mT")
    # print(f"R^2 for B1/2 fit = {R2: .3f}")

    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(Bmax=10, Bmin=0, dB=2, tmax=1e-6, dt=10e-8)
    else:
        main()
