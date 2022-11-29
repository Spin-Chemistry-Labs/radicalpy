#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def main():

    theta = np.linspace(0, np.pi, 17)
    phi = np.linspace(0, 2 * np.pi, 34)

    flavin = rp.simulation.Molecule("flavin_anion", ["N5"])
    Z = rp.simulation.Molecule("zorro", [])
    sim = rp.simulation.HilbertSimulation([flavin, Z])

    time = np.arange(0, 5e-6, 5e-9)
    B0 = 0.05
    k = 1e6

    results = sim.anisotropy(
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        theta=theta,
        phi=phi,
        B=B0,
        D=0,
        J=0,
        kinetics=[rp.kinetics.Exponential(k)],
    )

    Y = results["product_yield_sums"]
    delta_phi_s, gamma_s = rp.utils.yield_anisotropy(Y, theta, phi)
    Y_av = rp.utils.spherical_average(Y, theta, phi)
    Y = Y - Y_av

    rp.plot.anisotropy_surface(theta, phi, Y)
    
    print(f"{Y_av=}")
    print(f"{delta_phi_s=}")
    print(f"{gamma_s=}")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
