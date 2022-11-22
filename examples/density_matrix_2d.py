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
    rhos = MARY["rhos"]

    Bi = 1
    B = Bs[Bi]
    plt.imshow(np.abs(rhos[Bi, 0]))
    plt.title(f"B={B} mT")
    labels = rp.plot.spin_state_labels(sim)
    print(labels)
    # plt.xlabel(labels)
    plt.show()


if __name__ == "__main__":
    main()
