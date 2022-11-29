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

    theta = np.linspace(0, np.pi, 2)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    print(f"{len(phi) - n_phi=}")
    print(f"{phi[1] - phi_step=}")
    print(f"{phi[1]=}")

    flavin = rp.simulation.Molecule("flavin_anion", ["H25", "N5"])
    trp = rp.simulation.Molecule("tryptophan_cation", ["N1"])
    sim = rp.simulation.HilbertSimulation([flavin, trp])

    time = np.arange(0, 5e-6, 5e-9)
    B0 = 0.05
    k = 3e6

    results = sim.anisotropy(
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        B=B0,
        D=0,
        J=0,
        kinetics=[rp.kinetics.Exponential(k)],
    )
    print(results)

    # for phi in np.linspace(0, 2 * np.pi - phi_step, n_phi)
    # for theta in np.linspace(0, np.pi, n_theta)

    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi - phi_step, n_phi)

    # Create a 2-D meshgrid of (theta, phi) angles
    phi, theta = np.meshgrid(phi, theta)

    # Calculate the Cartesian coordinates of each point in the mesh
    xyz = rp.utils.spherical_to_cartesian(theta, phi)

    Yx, Yy, Yz = Y.real * xyz

    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("Accent_r"))
    cmap.set_clim(-0.1, 0.1)
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("none")
    ax.plot_surface(Yx, Yy, Yz, facecolors=cmap.to_rgba(Y.real), rstride=2, cstride=2)


if __name__ == "__main__":
    main()
