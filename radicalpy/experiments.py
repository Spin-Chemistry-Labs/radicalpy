#!/usr/bin/env python

import numpy as np
import numpy.testing as npt
import scipy as sp
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from .simulation import (
    HilbertIncoherentProcessBase,
    HilbertSimulation,
    LiouvilleSimulation,
    SemiclassicalSimulation,
    State,
)
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
    ts: ArrayLike,
    Bs: ArrayLike,
    D: float,
    J: float,
    triplet_excited_state_quenching_rate: float,
    free_radical_escape_rate: float,
    kinetics,  ##########################################
    relaxations,  #######################################
    scale_factor: float,
):
    dt = ts[1] - ts[0]
    initial = sim.projection_operator(init_state)
    M = 16  # number of spin states
    trace = np.zeros((num_samples, len(ts)))
    mary = np.zeros((len(ts), len(Bs)))
    HHs = sim.semiclassical_HHs(num_samples)

    for i, B0 in enumerate(tqdm(Bs)):
        Hz = sim.zeeman_hamiltonian(B0)
        for j, HH in enumerate(HHs):
            Ht = Hz + HH
            L = sim.convert(Ht)
            sim.apply_liouville_hamiltonian_modifiers(L, kinetics + relaxations)
            propagator = sp.sparse.linalg.expm(L * dt)

            FR_initial_population = 0  # free radical
            triplet_initial_population = 1  # triplet excited state

            initial_temp = np.reshape(initial / 3, (M, 1))
            density = np.reshape(np.zeros(16), (M, 1))

            for k in range(0, len(ts)):
                FR_density = density
                population = np.trace(FR_density)
                rho = population + (
                    FR_initial_population + population * free_radical_escape_rate * dt
                )
                trace[j, k] = np.real(rho)
                density = (
                    propagator * density
                    + triplet_initial_population
                    * (1 - np.exp(-triplet_excited_state_quenching_rate * dt))
                    * initial_temp
                )
                triplet_initial_population = triplet_initial_population * np.exp(
                    -triplet_excited_state_quenching_rate * dt
                )

        average = np.ones(num_samples) * scale_factor
        decay = average @ trace
        if i == 0:
            decay0 = np.real(decay)

        mary[:, i] = np.real(decay - decay0)
    return {"ts": ts, "Bs": Bs, "MARY": mary}


def semiclassical_kinetics_mary(
    sim: SemiclassicalSimulation,
    num_samples: int,
    init_state: ArrayLike,
    ts: NDArray[float],
    Bs: ArrayLike,
    D: float,
    J: float,
    kinetics: ArrayLike,
    relaxations: list[ArrayLike],
):
    dt = ts[1] - ts[0]
    result_1 = np.zeros((len(ts), len(Bs)), dtype=complex)
    zero_field = np.zeros((len(ts), len(Bs)), dtype=complex)
    mary = np.zeros((len(ts), len(Bs)), dtype=complex)
    kinetic_model = kinetics
    kinetic_matrix = np.zeros((len(kinetics), len(kinetics)), dtype=complex)
    rho_radical_pair = np.zeros(len(ts), dtype=complex)
    rho_triplet = np.zeros(len(ts), dtype=complex)
    radical_pair_yield = np.zeros((1, len(ts)), dtype=complex)
    triplet_yield = np.zeros((1, len(ts)), dtype=complex)
    HHs = sim.semiclassical_HHs(num_samples)

    for i, B0 in enumerate(tqdm(Bs)):
        Hz = sim.zeeman_hamiltonian(B0)
        for j, Hnuc in enumerate(HHs):
            Ht = Hz + Hnuc
            L = sim.convert(Ht)
            kinetic_matrix[5:21, 5:21] -= L
            kinetics = kinetic_model + kinetic_matrix
            propagator = sp.sparse.linalg.expm(kinetics * dt)

            for k in range(0, len(ts)):
                rho_radical_pair[k] = (
                    init_state[5] + init_state[10] + init_state[15] + init_state[20]
                )
                rho_triplet[k] = init_state[2] + init_state[3] + init_state[4]

                init_state = propagator @ init_state

        radical_pair_yield = (radical_pair_yield + rho_radical_pair) / num_samples
        triplet_yield = (triplet_yield + rho_triplet) / num_samples

        total_yield = radical_pair_yield + triplet_yield
        result_1[:, i] = result_1[:, i] + total_yield

    yield_zero_field = result_1[:, 0]
    for i in range(0, len(Bs)):
        zero_field[:, i] = yield_zero_field

    mary = np.real(result_1 - zero_field)
    return {"ts": ts, "Bs": Bs, "MARY": mary}
