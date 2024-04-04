#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.4])
    Z = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.5])
    sim = rp.simulation.HilbertSimulation([flavin, Z])
    H = sim.total_hamiltonian(B0=0, D=0, J=0)
    # plt.spy(H)
    # plt.show()
    time = np.arange(0, 3e-6, 5e-9)
    rhos = sim.time_evolution(State.SINGLET, time, H)

    # k = 3e6
    # kinetics = [rp.kinetics.Exponential(k)]

    time_evol = sim.product_probability(State.SINGLET, rhos)
    # sim.apply_hilbert_kinetics(time, time_evol, kinetics)
    # product_yield, product_yield_sum = sim.product_yield(time_evol, time, k)

    np.savetxt("./examples/data/fad_kinetics/timeevolution_yield_2nuc.txt", time_evol)
    np.savetxt("./examples/data/fad_kinetics/timeevolution_time_2nuc.txt", time)

    x = time * 1e6

    plt.plot(x, time_evol, color="red", linewidth=2)
    # plt.fill_between(x, product_yield, color="blue", alpha=0.2)
    plt.xlabel("Time ($\mu s$)")
    plt.ylabel("Probability")
    plt.ylim([0, 1])
    # plt.legend([r"$P_i(t)$", r"$\Phi_i$"])

    # print(f"PY = {product_yield_sum}")

    # path = __file__[:-3] + f"_{0}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    main()
