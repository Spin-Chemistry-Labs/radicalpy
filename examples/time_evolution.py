#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    # sim = rp.simulation.HilbertSimulation([flavin, Z])
    sim = rp.simulation.LiouvilleSimulation([flavin, Z])
    time = np.arange(0, 15e-6, 5e-9)
    Bs = np.arange(0, 3, 1)
    k = 1e6

    MARY = sim.MARY(
        init_state=State.SINGLET,
        obs_state=State.TRIPLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        # kinetics=[rp.kinetics.Exponential(k)],
        kinetics=[
            rp.kinetics.Haberkorn(k, State.SINGLET),
            rp.kinetics.Haberkorn(k, State.TRIPLET),
        ],
    )

    Bi = 1
    B = Bs[Bi]
    x = MARY["time"] * 1e6

    plt.plot(x, MARY["time_evolutions"][Bi], color="red", linewidth=2)
    plt.fill_between(x, MARY["product_yields"][Bi], color="blue", alpha=0.2)
    plt.xlabel("Time ($\mu s$)")
    plt.ylabel("Probability")
    plt.title(f"B={B}")
    plt.legend([r"$P_i(t)$", r"$\Phi_i$"])
    plt.show()


if __name__ == "__main__":
    main()
