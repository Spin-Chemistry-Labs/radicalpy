#! /usr/bin/env python

import numpy as np
import radicalpy as rp
from radicalpy import relaxation
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25", "N5"])
    trp = rp.simulation.Molecule("tryptophan_cation", ["N1"])
    # trp = rp.simulation.Molecule("trp")
    sim = rp.simulation.LiouvilleSimulation([flavin, trp])
    # sim = rp.simulation.HilbertSimulation([flavin, trp])
    time = np.arange(0, 2e-6, 1e-6)
    Bs = np.arange(0, 2, 1)
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
            # rp.kinetics.Exponential(k),
            rp.kinetics.Haberkorn(krec, State.SINGLET),
            rp.kinetics.HaberkornFree(kesc),
        ],
        relaxations=[relaxation.SingletTripletDephasing(kSTD)],
    )
    HFE = results["HFE"]
    LFE = results["LFE"]

    print(f"HFE = {HFE: .2f} %")
    print(f"LFE = {LFE: .2f} %")

    # MARY = results["MARY"]
    # Bhalf, x_model_MARY, y_model_MARY, MARY_fit_error, R2 = rp.utils.Bhalf_fit(Bs, MARY)
    # print(f"B1/2 = {Bhalf: .2f} mT")
    # print(f"B1/2 fit error = {MARY_fit_error[1]: .2f} mT")
    # print(f"R^2 for B1/2 fit = {R2: .3f}")


if __name__ == "__main__":
    main()
