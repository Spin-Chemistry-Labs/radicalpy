#!/usr/bin/env python

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from .simulation import (HilbertIncoherentProcessBase, LiouvilleSimulation,
                         SemiclassicalSimulation, State)
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
    Bs: NDArray[float],
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


def semiclassical_mary(
    sim: SemiclassicalSimulation,
    num_samples: int,
    init_state: State,
    time: NDArray[float],
    B0: ArrayLike,
    D: float,
    J: float,
    triplet_excited_state_quenching_rate: float,
    free_radical_escape_rate: float,
    kinetics,  ##########################################
    relaxations,  #######################################
    I_max: list[float],
    fI_max: list[float],
):
    gen = sim.semiclassical_gen(5, I_max, fI_max)
    for g in gen:
        print(f"{g=}")
    exit()
    for i, B0 in enumerate(tqdm(B)):
        gen = sim.semiclassical_gen(num_samples)
        for HZL in gen:
            sim.apply_liouville_hamiltonian_modifiers(HZL, kinetics + relaxation)
            L = HZL

            propagator = sp.sparse.linalg.expm(L * dt)

            FR_initial_population = 0  # free radical
            triplet_initial_population = 1  # triplet excited state

            initial_temp = np.reshape(initial / 3, (M, 1))
            density = np.reshape(np.zeros(16), (M, 1))

            for k in range(0, len(time)):
                FR_density = density
                population = np.trace(FR_density)
                rho = population + (FR_initial_population + population * kesc * dt)
                trace[j, k] = np.real(rho)
                density = (
                    propagator * density
                    + triplet_initial_population * (1 - np.exp(-kq * dt)) * initial_temp
                )
                triplet_initial_population = triplet_initial_population * np.exp(
                    -kq * dt
                )

        average = np.ones(num_samples) * 0.01
        decay = average @ trace
        if i == 0:
            decay0 = np.real(decay)

        mary_1[:, i] = np.real(decay)
        mary_2[:, i] = np.real(decay - decay0)
