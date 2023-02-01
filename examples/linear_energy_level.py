#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])
    H = sim.total_hamiltonian(B=1, D=0, J=0)

    eigval = np.linalg.eigh(H)
    E = np.real(eigval[0])  # 0 = eigenvalues, 1 = eigenvectors

    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.eventplot(E, orientation="vertical", color="red", linewidth=3)
    ax.set_ylabel("Spin state energy (J)", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
