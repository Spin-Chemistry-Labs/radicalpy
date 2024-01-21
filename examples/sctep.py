#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.data import Isotope, Molecule, Nucleus, Triplet
from radicalpy.experiments import steady_state_mary
from radicalpy.kinetics import Haberkorn, HaberkornFree
from radicalpy.simulation import Basis, LiouvilleSimulation, State


def main(
    Bs=np.arange(0, 2500, 10),
    D=-6.2,
    E=35,
    J=499.55,
    k0=1,
    ks=1.1e9,
    kd=2.8e9,
):
    m = Triplet()
    sim = LiouvilleSimulation(molecules=[m, m], basis=Basis.ZEEMAN)
    rhos, Phi_s = steady_state_mary(
        sim,
        obs=State.TP_SINGLET,
        Bs=Bs,
        D=D,
        E=E,
        J=J,
        theta=np.pi / 4,
        phi=0,
        kinetics=[Haberkorn(ks, State.TP_SINGLET), HaberkornFree(kd)],
    )
    rhos *= k0
    Phi_s *= k0 * ks

    MFE = ((np.abs(Phi_s) - np.abs(Phi_s[0])) / np.abs(Phi_s[0])) * 100
    # print(H)
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(Bs / J, MFE, linewidth=3, color="tab:red")
    ax.set_xlabel("g$μ_B$$B_0$ / J", size=18)
    ax.set_ylabel("MFE (%)", size=18)
    ax.axvline(x=1.5, color="k", linestyle="--")
    ax.axvline(x=3, color="k", linestyle="--")
    plt.tick_params(labelsize=14)
    fig.set_size_inches(8, 5)
    plt.show()


if __name__ == "__main__":
    main()
