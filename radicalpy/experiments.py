#!/usr/bin/env python

import numpy as np
from tqdm import tqdm

from .simulation import HilbertIncoherentProcessBase, LiouvilleSimulation, State
from .utils import mary_lorentzian, modulated_signal, reference_signal


def modulated_mary_brute_force(
    Bs: np.ndarray,
    modulation_depths: list,
    modulation_frequency: float,
    time_constant: float,
    harmonics: list,
    lfe_magnitude: float,
) -> np.ndarray:
    t = np.linspace(-3 * time_constant, 0, 100)  # Simulate over 10 time constants
    S = np.zeros([len(harmonics), len(modulation_depths), len(Bs)])
    sa = 0

    for i, h in enumerate(tqdm(harmonics)):
        for j, md in enumerate(modulation_depths):
            for k, B in enumerate(Bs):
                for l in range(0, 20):  # Randomise phase
                    theta = 2 * np.pi * np.random.rand()

                    # Calculate the modulated signal
                    ms = B + md * modulated_signal(t, theta, modulation_frequency)
                    s = mary_lorentzian(ms, lfe_magnitude)
                    s = s - np.mean(s)  # Signal (AC coupled)

                    # Calculate the reference signal
                    envelope = np.exp(t / time_constant) / time_constant  # Envelope
                    r = (
                        reference_signal(t, h, theta, modulation_frequency) * envelope
                    )  # Reference * envelope

                    # Calculate the MARY spectra
                    sa = sa + np.trapz(t, s * r)  # Integrate
                sa = sa
                S[i, j, k] = sa * np.sqrt(2)  # RMS signal
    return S


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
    HZFS = sim.zero_field_splitting_hamiltonian(D, E)
    HJ = sim.exchange_hamiltonian(-J, prod_coeff=1)
    rhos = np.zeros(shape=(len(Bs), sim.hamiltonian_size))
    Q = sim.projection_operator(obs)
    for i, B in enumerate(tqdm(Bs)):
        HZ = sim.zeeman_hamiltonian(B, theta=theta, phi=phi)
        H = HZ + HZFS + HJ
        H = sim.convert(H)
        sim.apply_liouville_hamiltonian_modifiers(H, kinetics)  # + relaxations)
        rhos[i] = np.linalg.solve(H, Q.flatten())

    Phi_s = rhos @ Q.flatten()
    return rhos, Phi_s
