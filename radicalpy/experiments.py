#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .simulation import (HilbertIncoherentProcessBase, HilbertSimulation,
                         LiouvilleSimulation, State)


def steady_state_mary(
    sim: HilbertSimulation,
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
    HZFS = sim.zero_field_splitting_hamiltonian(D, E)
    HJ = sim.exchange_hamiltonian(J)
    rhos = np.zeros(shape=(len(Bs), *HJ.shape))
    Q = sim.projection_operator(obs)
    for i, B in enumerate(tqdm(Bs)):
        HZ = sim.zeeman_hamiltonian_3d(B, theta, phi)
        H = HZ + HZFS + HJ
        sim.apply_liouville_hamiltonian_modifiers(H, kinetics)  # + relaxations)
        rhos[i] = np.linalg.solve(H, Q)
    Phi_s = sim.product_probability(obs, rhos)

    return rhos, Phi_s
