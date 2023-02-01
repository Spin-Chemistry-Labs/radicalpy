#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy import relaxation
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["H25", "N5"])
    trp = rp.simulation.Molecule.fromdb("tryptophan_cation", ["N1"])
    sim = rp.simulation.LiouvilleSimulation([flavin, trp])
    time = np.arange(0, 5e-6, 5e-9)
    Bs = np.arange(0, 30, 0.1)
    krec = 1e6
    kesc = 1e6
    kSTD = 1e7

    results = sim.MARY(
        init_state=State.TRIPLET,
        obs_state=State.TRIPLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        kinetics=[
            rp.kinetics.Haberkorn(krec, State.SINGLET),
            rp.kinetics.HaberkornFree(kesc),
        ],
        relaxations=[relaxation.SingletTripletDephasing(kSTD)],
    )
    MARY = results["MARY"]
    HFE = results["HFE"]
    LFE = results["LFE"]

    Bhalf, fit_result, fit_error, R2 = rp.utils.Bhalf_fit(Bs, MARY)

    plt.plot(Bs, MARY, color="red", linewidth=2)
    plt.plot(Bs, fit_result, "k--", linewidth=1, label="Lorentzian fit")

    plt.xlabel("$B_0 (mT)$")
    plt.ylabel("MFE (%)")
    plt.title("")
    plt.legend([r"Simulation", r"Fit"])

    print(f"HFE = {HFE: .2f} %")
    print(f"LFE = {LFE: .2f} %")
    print(f"B1/2 = {Bhalf: .2f} mT")
    print(f"B1/2 fit error = {fit_error[1]: .2f} mT")
    print(f"R^2 for B1/2 fit = {R2: .3f}")

    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
