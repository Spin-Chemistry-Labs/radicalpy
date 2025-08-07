#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy import relaxation
from radicalpy.experiments import odmr
from radicalpy.simulation import State
from radicalpy.utils import is_fast_run


def main(Bmax=20, dB=0.5, tmax=10e-6, dt=10e-9):
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["H25"])  # , "H27", "H29"])
    trp = rp.simulation.Molecule.fromdb("tryptophan_cation", ["H1"])  # , "Hbeta1"])
    sim = rp.simulation.LiouvilleSimulation([flavin, trp])
    time = np.arange(0, tmax, dt)
    B0 = 3.14  # FIX MEEEEE!
    B1 = np.arange(0, Bmax, dB)
    krec = 1.1e7
    kesc = 7e6
    kSTD = 1e8
    kr = 7e7

    results = odmr(
        sim,
        init_state=State.TRIPLET,
        obs_state=State.TRIPLET,
        time=time,
        D=0,
        J=0,
        B0=B0,
        B1=B1,
        kinetics=[
            rp.kinetics.Haberkorn(krec, State.SINGLET),
            rp.kinetics.HaberkornFree(kesc),
        ],
        relaxations=[
            relaxation.SingletTripletDephasing(kSTD),
            relaxation.RandomFields(kr),
        ],
    )
    MARY = results["MARY"]
    HFE = results["HFE"]
    LFE = results["LFE"]

    # np.save("./examples/data/fad_mary/results_5nuc_liouville_relaxation.npy", results)

    Bhalf, fit_result, fit_error, R2 = rp.utils.Bhalf_fit(B1, MARY)

    plt.plot(B1, MARY, color="red", linewidth=2)
    plt.plot(B1, fit_result, "k--", linewidth=1, label="Lorentzian fit")

    plt.xlabel("$B_0 (mT)$")
    plt.ylabel("MFE (%)")
    plt.title("")
    plt.legend([r"Simulation", r"Fit"])

    print(f"HFE = {HFE: .2f} %")
    print(f"LFE = {LFE: .2f} %")
    print(f"B1/2 = {Bhalf: .2f} mT")
    print(f"B1/2 fit error = {fit_error[1]: .2f} mT")
    print(f"R^2 for B1/2 fit = {R2: .3f}")

    path = __file__[:-3] + f"_{15}.png"
    plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(Bmax=10, dB=2, tmax=1e-6, dt=10e-8)
    else:
        main()
