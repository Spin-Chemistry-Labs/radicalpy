#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.experiments import mary
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    # sim = rp.simulation.HilbertSimulation([flavin, Z])
    sim = rp.simulation.LiouvilleSimulation([flavin, Z])
    time = np.arange(0, 15e-6, 5e-9)
    Bs = np.arange(0, 10, 1)
    k = 1e6

    MARY = mary(
        sim,
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

    Bi = 9
    ti = 10
    B = Bs[Bi]
    plt.imshow(np.abs(rhos[Bi, ti]))
    plt.title(f"B={B} mT")
    plt.tick_params(bottom=False, left=False)
    plt.colorbar()

    ax = plt.gca()
    labels = rp.plot.spin_state_labels(sim)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    # plt.show()
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
