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

    rho0 = sim.projection_operator(State.SINGLET)
    obs = sim.projection_operator(State.SINGLET)

    sigmax = sim.spin_operator(0, "x") + sim.spin_operator(1, "x")
    sigmay = sim.spin_operator(0, "y") + sim.spin_operator(1, "y")
    sigmaz = sim.spin_operator(0, "z") + sim.spin_operator(1, "z")

    Sxyz = sigmax + sigmay + sigmaz

    def S(omega: float, tau_c: float) -> float:
        omega= float(omega)
        return tau_c / (1.0 + (omega * tau_c) * (omega * tau_c))

    tau_c = 1e7
    NPS = lambda omega: S(omega, tau_c)

    time = np.arange(0, 5e-6, 1e-9)

    L = sim.bloch_redfield_time_evolution(
        H, rho0, time, bath=[Sxyz], noise=[NPS], obs=[obs]
    )
    L_result = L['expect'][0] / np.trace(rho0)

    fig = plt.figure(1)
    plt.plot(time * 1e6, L_result)
    plt.xlabel("Time / $\mu s$", size=14)
    plt.ylabel("Probability", size=14)
    plt.ylim([0, 1])
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
