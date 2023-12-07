#! /usr/bin/env python

from radicalpy.data import Isotope, Molecule, Nucleus, Triplet
from radicalpy.experiments import steady_state_mary
from radicalpy.simulation import Basis, LiouvilleSimulation


def main():
    gamma = Isotope("E").gamma_mT
    D = -6.2 * gamma
    E = 35 * gamma
    m = Triplet()
    sim = LiouvilleSimulation(molecules=[m, m])
    print(sim)
    # sim = LiouvilleSimulation(molecules=[m, m], basis=Basis.ZEEMAN)
    # # print(sim)
    # steady_state_mary(
    #     sim,
    #     obs_state=State.TP_SINGLET,
    #     B=Bs,
    #     D=D,
    #     E=E,
    #     J=0,
    #     kinetics=[
    #         rp.kinetics.Haberkorn(krec, State.TP_SINGLET),
    #         rp.kinetics.HaberkornFree(kesc),
    #     ],
    # )
    # print(H)


if __name__ == "__main__":
    main()
