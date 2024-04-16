#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State
from radicalpy import relaxation, kinetics


def main():

    fad_n5_hfc = np.array(
        [
            [0.280, -0.138, 0.678],
            [-0.138, 0.043, -0.331],
            [0.678, -0.331, 1.412]
    ]
    )

    dipolar = rp.estimations.dipolar_interaction_anisotropic(r=22e-10)

    theta = np.linspace(0, np.pi, 35)
    phi = np.linspace(0, 2 * np.pi, 58)

    flavin = rp.simulation.Molecule.fromisotopes(isotopes=["14N"], hfcs=[fad_n5_hfc])
    Z = rp.simulation.Molecule("zorro", [])
    sim = rp.simulation.HilbertSimulation([flavin, Z])

    time = np.arange(0, 15e-6, 5e-9)
    B0 = 0.05
    k = 1e6

    results = sim.anisotropy(
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        theta=theta,
        phi=phi,
        B0=B0,
        D=dipolar,
        J=0,
        kinetics=[
            kinetics.Exponential(k)
        ],
    )

    Y = results["product_yield_sums"]
    delta_phi_s, gamma_s = rp.utils.yield_anisotropy(Y, theta, phi)
    Y_av = rp.utils.spherical_average(Y, theta, phi)
    Y = Y - Y_av

    np.save("Y", Y)
    np.save("theta", theta)
    np.save("phi", phi)
    rp.plot.anisotropy_surface(theta, phi, Y)

    print(f"{Y_av=}")
    print(f"{delta_phi_s=}")
    print(f"{gamma_s=}")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
