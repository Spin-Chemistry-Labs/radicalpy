#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.data import Molecule
from radicalpy.experiments import magnetic_field_loop
from radicalpy.kinetics import Haberkorn, HaberkornFree
from radicalpy.relaxation import SingletTripletDephasing
from radicalpy.simulation import HilbertSimulation, LiouvilleSimulation, State
from radicalpy.utils import is_fast_run


def main():
    m1 = Molecule.fromdb("flavin_anion", ["N5"])
    m2 = Molecule.fromdb("tryptophan_cation", [])
    simH = HilbertSimulation([m1, m2])
    sim = LiouvilleSimulation([m1, m2])
    H = simH.total_hamiltonian(B0=0, J=0, D=0)

    gamma = 3e6
    sigmax = simH.spin_operator(0, "x") + simH.spin_operator(1, "x")
    sigmay = simH.spin_operator(0, "y") + simH.spin_operator(1, "y")
    sigmaz = simH.spin_operator(0, "z") + simH.spin_operator(1, "z")

    Ls = [np.sqrt(gamma) * sigmax, np.sqrt(gamma) * sigmay, np.sqrt(gamma) * sigmaz]
    linblad = sim.linblad_liouvillian(H=H, Ls=Ls)

    dB = 0.5
    B0 = np.arange(0.0, 20.0 + 1e-9, dB)
    time = np.arange(0, 3e-6, 10e-9)
    kinetics = [Haberkorn(3e6, State.SINGLET), HaberkornFree(1e6)]
    relaxations = [SingletTripletDephasing(1e7)]

    init_state = State.SINGLET
    obs_state = State.TRIPLET

    sim.apply_liouville_hamiltonian_modifiers(linblad, kinetics + relaxations)
    rhos = magnetic_field_loop(sim, init_state, time, linblad, B0, B_axis="z")
    product_probabilities = sim.product_probability(obs_state, rhos)
    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    x = time * 1e6
    n = 0

    fig = plt.figure(1)
    plt.plot(x, product_probabilities[n, :], linewidth=2, label=r"$P_i(t)$")
    plt.fill_between(x, product_yields[n, :], alpha=0.2, label=r"$\Phi_i$")
    plt.xlabel("Time / $\mu s$", size=14)
    plt.ylabel("Probability", size=14)
    # plt.ylim([0, 1])
    plt.legend(fontsize=14)
    fig.set_size_inches([7, 4])
    plt.show()
    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main()
    else:
        main()
