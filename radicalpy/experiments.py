#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .simulation import (HilbertIncoherentProcessBase, HilbertSimulation,
                         LiouvilleSimulation, State)


def steady_state_mary(
    sim: LiouvilleSimulation,
    obs: State,
    Bs: np.ndarray,
    D: float,
    E: float,
    J: float,
    theta: float,
    phi: float,
    kinetics: list[HilbertIncoherentProcessBase] = [],
    # relaxations: list[HilbertIncoherentProcessBase] = [],
) -> np.ndarray:
    HZFS = sim.zero_field_splitting_hamiltonian(-D, -E)
    HJ = sim.exchange_hamiltonian(-J, prod_coeff=1)
    rhos = np.zeros(shape=(len(Bs), sim.hamiltonian_size))
    Q = sim.projection_operator(obs)
    for i, B in enumerate(tqdm(Bs)):
        HZ = sim.zeeman_hamiltonian(B, theta=theta, phi=phi)
        H = HZ + HZFS + HJ
        H = sim.convert(H)
        sim.apply_liouville_hamiltonian_modifiers(H, kinetics)  # + relaxations)
        rhos[i] = np.linalg.solve(H, Q.flatten())

    # Phi_s = sim.product_probability(obs, rhos)
    print(f"{Q.shape=}")
    print(f"{rhos.shape=}")
    # Phi_s = np.sum(Q * rhos, axis=(-1, -2))
    print(f"{Q.shape=} {rhos.shape=}")
    Phi_s = rhos @ Q.flatten()
    # print(f"{(Q * rhos).shape=}")
    print(f"{Phi_s.shape=}")

    return rhos, Phi_s
