#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.experiments import anisotropy
from radicalpy.simulation import State
from radicalpy.utils import is_fast_run


# def main(theta_steps=35, phi_steps=58, tmax=15e-6, dt=5e-9):
def main(theta_steps=17, phi_steps=32, tmax=5e-6, dt=5e-9):
    theta = np.linspace(0, np.pi, theta_steps)
    phi = np.linspace(0, 2 * np.pi, phi_steps)

    flavin = rp.simulation.Molecule.fromdb("flavin_anion", ["N5", "N10"])
    Z = rp.simulation.Molecule("zorro", [])
    sim = rp.simulation.HilbertSimulation([flavin, Z])

    time = np.arange(0, tmax, dt)
    B0 = 0.05
    k = 1e6

    results = anisotropy(
        sim,
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        theta=theta,
        phi=phi,
        B0=B0,
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
    # plt.show()
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path)

    return 0


if __name__ == "__main__":
    if is_fast_run():
        main(theta_steps=7, phi_steps=6, tmax=10e-6, dt=1e-6)
    else:
        main()
