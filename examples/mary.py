#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])
    time = np.arange(0, 15-6, 5e-9)
    Bs = np.arange(0, 10, 1)
    k = 3e6

    MARY = sim.MARY(
        init_state=State.SINGLET,
        obs_state=State.TRIPLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        kinetics=[
            rp.kinetics.Exponential(k)
        ],
    )

    x = MARY["B"] * 1e3

    plt.plot(x, MARY["MARY"], color="red", linewidth=2)
    plt.xlabel("$B_0 (mT)$")
    plt.ylabel("MFE (%)")
    plt.title(f"B={B}")
    plt.legend([r"$P_i(t)$", r"$\Phi_i$"]); path = __file__[:-3] + f"_{0}.png"; plt.savefig(path)
    #plt.show()


if __name__ == "__main__":
    main()
