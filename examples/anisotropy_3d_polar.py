#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def spherical_average_subtraction(lst, n_theta, theta_step, n_phi, phi_step):
    # Subtracting the spherical average from the singlet yield
    # Simpson's rule integration over theta (0, pi) and phi (0, 2pi)
    # n_theta = odd, n_phi = even

    w_theta = np.ones(n_theta) * 4
    w_theta[2:-2:2] = 2
    w_theta[0] = 1
    w_theta[-1] = 1

    w_phi = np.ones(n_phi) * 4
    w_phi[0:-1:2] = 2
    sintheta = np.sin(np.linspace(0, np.pi, n_theta))

    spherical_average = (
        sum(
            lst[i, j] * sintheta[i] * w_theta[i] * w_phi[j]
            for i in range(n_theta)
            for j in range(n_phi)
        )
        * theta_step
        * phi_step
        / (4 * np.pi)
        / 9
    )

    return lst - spherical_average


def main():

    theta = np.linspace(0, np.pi, 9)
    phi = np.linspace(0, 2 * np.pi, 18)

    # flavin = rp.simulation.Molecule("flavin_anion", ["H25", "N5"])
    # trp = rp.simulation.Molecule("tryptophan_cation", ["N1"])
    flavin = rp.simulation.Molecule("flavin_anion", ["N5", "N10"])
    trp = rp.simulation.Molecule("tryptophan_cation", [])
    sim = rp.simulation.HilbertSimulation([flavin, trp])

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
    print(results.keys())
    for key, val in results.items():
        try:
            print(f"{key} {val.shape}")
        except:
            pass
    Y = results["product_yields"][:, :, 1]
    # Y = results["product_yield_sums"]
    Y = np.sum(results["product_yields"], axis=2) * time[1] * k
    print(Y)
    Y = Y - rp.utils.spherical_average(Y, theta, phi)

    print(f"{Y.shape=}")
    rp.plot.anisotropy_surface(theta, phi, Y)
    plt.show()
    return 0


if __name__ == "__main__":
    main()
