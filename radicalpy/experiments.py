#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .simulation import (HilbertIncoherentProcessBase, HilbertSimulation,
                         LiouvilleSimulation, State)


def steady_state_mary(
    sim: HilbertSimulation,
    obs_state: State,
    Bs: np.ndarray,
    D: float,
    E: float,
    J: float,
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
) -> np.ndarray:
    theta = np.pi / 4
    phi = 0
    HZFS = sim.zero_field_splitting_hamiltonian(D, E)
    HJ = sim.exchange_hamiltonian(J)
    Phi_s = np.zeros((len(Bs)), dtype=complex)
    Ps = sim.projection_operator(State.TP_SINGLET)
    for i, B in enumerate(tqdm(Bs)):
        HZ = sim.zeeman_hamiltonian_3d(B, theta, phi)
        print(f"{HZ.shape=}")
        print(f"{HZFS.shape=}")
        print(f"{HJ.shape=}")
        H = HZ + HZFS + HJ
        sim.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
        # HL = ry.Hilbert2Liouville(H)
        # L = 1j * H + sum(kinetics)
        rho = np.linalg.solve(H, Ps)  # Density operator

        print(f"{type(rho)=} {rho.shape=}")
        print(f"{type(Ps)=} {Ps.shape=}")

        Phi_s[i] = np.matmul(Ps.T, rho)

    return rho, Phi_s
