#! /usr/bin/env python

import numpy as np
from radicalpy.data import Isotope, Molecule, Nucleus, Triplet
from radicalpy.experiments import steady_state_mary
from radicalpy.kinetics import Haberkorn, HaberkornFree
from radicalpy.simulation import Basis, HilbertSimulation, State


def main():
    # gamma = Isotope("E").gamma_mT
    Bs = np.arange(0, 2500, 1)
    D = -6.2  # * gamma
    E = 35  #  * gamma
    J = 499.55
    k0 = 1
    ks = 1.1e9
    kd = 2.8e9
    m = Triplet()
    sim = HilbertSimulation(molecules=[m, m], basis=Basis.ZEEMAN)
    print(sim)
    H = sim.zero_field_splitting_hamiltonian(
        D=D,
        E=E,
    )
    print(H)
    steady_state_mary(
        sim,
        obs_state=State.TP_SINGLET,
        Bs=Bs,
        D=D,
        E=E,
        J=J,
        kinetics=[Haberkorn(ks, State.TP_SINGLET), HaberkornFree(kd)],
    )
    # print(H)


if __name__ == "__main__":
    main()
