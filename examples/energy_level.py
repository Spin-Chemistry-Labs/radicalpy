#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", [])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])

    rp.plot.energy_levels(sim, B=np.arange(0.01, 1, 0.01), J=-0.05, D=0)
    plt.show()
    # path = __file__[:-3] + f"_{0}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    main()
