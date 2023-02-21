#! /usr/bin/env python

# Simulation for Figure 4a in J. Chem. Phys. 150, 094105 (2019); https://doi.org/10.1063/1.5077078
# This example demonstrates how to make custom simulations using the RadicalPy framework

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy import kinetics, relaxation
from radicalpy.shared import constants as C
from radicalpy.simulation import LiouvilleSimulation as Liouv
from radicalpy.simulation import State


def main():
    # Create a radical pair with a 13C nucleus coupled to radical 1
    r1 = rp.simulation.Molecule.fromisotopes(isotopes=["13C"], hfcs=[0.0])
    r2 = rp.simulation.Molecule("radical 2")
    sim = rp.simulation.LiouvilleSimulation([r1, r2])

    # Modify the g-factors for both radicals
    ge = -C.g_e
    g1 = 2.0031
    g2 = 2.0031
    r1.radical.magnetogyric_ratio = r1.radical.magnetogyric_ratio / ge * g1
    r2.radical.magnetogyric_ratio = r2.radical.magnetogyric_ratio / ge * g2

    # Parameters for the simulation --> see Fig. 4 caption
    a = -0.05
    d = -2
    b = 0.15
    ks = 0.015e9
    kt = 0.4e9
    ksc = 0.0
    xi = 0.98

    init_state = State.SINGLET
    time = np.arange(0, 5e-6, 5e-9)
    B = np.arange(0, 20000, 100)
    kinetic = [
        kinetics.Haberkorn(ks, State.SINGLET),
        kinetics.Haberkorn(kt, State.TRIPLET),
        kinetics.HaberkornFree(ksc),
    ]
    relaxations = []

    # Create the CIDNP observables --> see Eq. 6
    singlet = sim.projection_operator(State.SINGLET)
    triplet = sim.projection_operator(State.TRIPLET)

    Iz = sim.spin_operator(2, "z")
    obs_state1 = np.reshape(singlet.dot(Iz), (-1, 1)).T
    obs_state2 = np.reshape(triplet.dot(Iz), (-1, 1)).T

    # Create the spin operators for the electron-electron coupling Hamiltonian --> see Eq. 5
    S1p = sim.spin_operator(0, "p")
    S1m = sim.spin_operator(0, "m")
    S2p = sim.spin_operator(1, "p")
    S2m = sim.spin_operator(1, "m")

    # Create the spin operators for the hyperfine Hamiltonian --> see Eq. 5
    S1z = sim.spin_operator(0, "z")
    Ix = sim.spin_operator(2, "x")

    # Construct the Hamiltonian and convert to Liouville space --> see Eq. 5
    H = d * sim.particles[0].gamma_mT * (S1p.dot(S2m) + S1m.dot(S2p))
    H += a * sim.particles[2].gamma_mT * S1z.dot(Iz)
    H += b * sim.particles[2].gamma_mT * S1z.dot(Ix)
    HL = Liouv.convert(H)

    # Run the magnetic field loop
    sim.apply_liouville_hamiltonian_modifiers(HL, kinetic + relaxations)
    rhos = sim.mary_loop(init_state, time, B, HL, theta=None, phi=None)

    # Calculate CIDNP of both singlet and triplet yields --> see Eqs. 6 and 7
    product_probabilities1 = np.real(np.trace(obs_state1 @ rhos, axis1=-2, axis2=-1))
    product_probabilities2 = np.real(np.trace(obs_state2 @ rhos, axis1=-2, axis2=-1))
    product_probabilities = product_probabilities1 + xi * product_probabilities2

    sim.apply_hilbert_kinetics(time, product_probabilities, kinetic)
    k = kinetic[0].rate_constant if kinetic else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    CIDNP, LFE, HFE = sim.mary_lfe_hfe(init_state, B, product_probabilities, dt, k)

    # Normalise the CIDNP intensity and plot and save figure
    norm_CIDNP = (CIDNP - CIDNP.max()) / (CIDNP.max() - CIDNP.min())
    plt.xscale("log")
    plt.plot(B[10:-1] / 1e3, norm_CIDNP[10:-1], color="red", linewidth=2)
    plt.xlabel("$log(B_0) (T)$")
    plt.ylabel("$^{13}$C CIDNP")

    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
