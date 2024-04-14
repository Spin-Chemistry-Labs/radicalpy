#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])
    H = sim.total_hamiltonian(B0=1, D=0, J=0)

    eigval = np.linalg.eigh(H)
    E = np.real(eigval[0])  # 0 = eigenvalues, 1 = eigenvectors

    rp.plot.energy_levels(sim, B=np.arange(0.01, 1, 0.01), J=0, D=0)
    plt.show()


if __name__ == "__main__":
    main()
