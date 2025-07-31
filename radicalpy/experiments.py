#!/usr/bin/env python

import itertools
from typing import Optional

import numpy as np
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
from .utils import anisotropy_check, mary_lorentzian, modulated_signal, reference_signal


def mary_lfe_hfe(
    init_state: State,
    B: np.ndarray,
    product_probability_seq: np.ndarray,
    dt: float,
    k: float,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Calculate MARY, LFE, HFE."""
    MARY = np.sum(product_probability_seq, axis=1) * dt * k
    idx = int(len(MARY) / 2) if B[0] != 0 else 0
    minmax = max if init_state is State.SINGLET else min
    HFE = (MARY[-1] - MARY[idx]) / MARY[idx] * 100
    LFE = (minmax(MARY) - MARY[idx]) / MARY[idx] * 100
    MARY = (MARY - MARY[idx]) / MARY[idx] * 100
    return MARY, LFE, HFE


def mary_loop(
    sim: HilbertSimulation,
    init_state: State,
    time: np.ndarray,
    B: np.ndarray,
    H_base: np.ndarray,
    theta: Optional[float] = None,
    phi: Optional[float] = None,
    hfc_anisotropy: bool = False,
) -> np.ndarray:
    """Generate density matrices (rhos) for MARY.

    Args:

        init_state (State): initial state.

    Returns:
        np.ndarray:

            Density matrices.

    .. todo:: Write proper docs.
    """
    H_zee = sim.convert(sim.zeeman_hamiltonian(1, theta, phi))
    shape = sim._get_rho_shape(H_zee.shape[0])
    rhos = np.zeros([len(B), len(time), *shape], dtype=complex)
    for i, B0 in enumerate(tqdm(B)):
        H = H_base + B0 * H_zee
        H_sparse = sp.sparse.csc_matrix(H)
        init_rho = sim.initial_density_matrix(init_state, H_sparse)
        rhos[i] = sim.time_evolution(init_rho, time, H_sparse)
    return rhos


def mary(
    sim: HilbertSimulation,
    init_state: State,
    obs_state: State,
    time: np.ndarray,
    B: np.ndarray,
    D: float,
    J: float,
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
    theta: Optional[float] = None,
    phi: Optional[float] = None,
    hfc_anisotropy: bool = False,
) -> dict:
    H = sim.total_hamiltonian(B0=0, D=D, J=J, hfc_anisotropy=hfc_anisotropy)

    sim.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
    rhos = mary_loop(sim, init_state, time, B, H, theta=theta, phi=phi)
    product_probabilities = sim.product_probability(obs_state, rhos)

    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    MARY, LFE, HFE = mary_lfe_hfe(init_state, B, product_probabilities, dt, k)
    rhos = sim._square_liouville_rhos(rhos)

    return dict(
        time=time,
        B=B,
        theta=theta,
        phi=phi,
        rhos=rhos,
        time_evolutions=product_probabilities,
        product_yields=product_yields,
        product_yield_sums=product_yield_sums,
        MARY=MARY,
        LFE=LFE,
        HFE=HFE,
    )


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
    kinetics: list[ArrayLike],
    relaxations: list[ArrayLike],
    scale_factor: float,
):
    dt = ts[1] - ts[0]
    initial = sim.projection_operator(init_state)
    M = 16  # number of spin states
    trace = np.zeros((num_samples, len(ts)))
    mary = np.zeros((len(ts), len(Bs)))
    HHs = sim.semiclassical_HHs(num_samples)
    HJ = sim.exchange_hamiltonian(J)
    HD = sim.dipolar_hamiltonian(D)

    for i, B0 in enumerate(tqdm(Bs)):
        Hz = sim.zeeman_hamiltonian(B0)
        for j, HH in enumerate(HHs):
            Ht = Hz + HH + HJ + HD
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


def kine_quantum_mary(
    sim: SemiclassicalSimulation,
    num_samples: int,
    init_state: ArrayLike,
    radical_pair: list,
    ts: NDArray[float],
    Bs: ArrayLike,
    D: float,
    J: float,
    kinetics: ArrayLike,
    relaxations: list[ArrayLike],
):
    dt = ts[1] - ts[0]
    total_yield = np.zeros((len(ts), len(init_state), len(Bs)), dtype=complex)
    kinetic_matrix = np.zeros((len(kinetics), len(kinetics)), dtype=complex)
    loop_rho = np.zeros((len(ts), len(init_state)), dtype=complex)
    loop_yield = np.zeros((len(ts), len(init_state)), dtype=complex)
    HHs = sim.semiclassical_HHs(num_samples)
    HJ = sim.exchange_hamiltonian(J)
    HD = sim.dipolar_hamiltonian(D)

    for i, B0 in enumerate(tqdm(Bs)):
        Hz = sim.zeeman_hamiltonian(B0)
        for HH in HHs:
            Ht = Hz + HH + HJ + HD
            L = sim.convert(Ht)
            sim.apply_liouville_hamiltonian_modifiers(L, relaxations)
            kinetic_matrix[
                radical_pair[0] : radical_pair[1], radical_pair[0] : radical_pair[1]
            ] = L
            kinetic = kinetics + kinetic_matrix
            rho0 = init_state
            propagator = sp.sparse.linalg.expm(kinetic * dt)

            for k in range(0, len(ts)):
                loop_rho[k, :] = rho0
                rho0 = propagator @ rho0

            loop_yield = loop_yield + loop_rho
        total_yield[:, :, i] = loop_yield / num_samples

    return {"ts": ts, "Bs": Bs, "yield": total_yield}


def anisotropy_loop(
    sim: HilbertSimulation,
    init_state: State,
    obs_state: State,
    time: np.ndarray,
    B0: float,
    H_base: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    r"""Inner loop of anisotropy experiment.

    Args:

        sim (LiouvilleSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        obs_state (State): Observable `State` of the density matrix.

        time (np.ndarray): An sequence of (uniform) time points,
            usually created using `np.arange` or `np.linspace`.

        B0 (float): External magnetic field intensity (milli
            Tesla) (see `zeeman_hamiltonian`).

        H_base (np.ndarray): A "base" Hamiltonian, i.e., the
            Zeeman Hamiltonian will be added to this base, usually
            obtained with `total_hamiltonian` and `B0=0`.

        theta (np.ndarray): rotation (polar) angle between the
            external magnetic field and the fixed molecule. See
            `zeeman_hamiltonian_3d`.

        phi (np.ndarray): rotation (azimuth) angle between the
            external magnetic field and the fixed molecule. See
            `zeeman_hamiltonian_3d`.

    Returns:
        np.ndarray:

        A tensor which has a series of density matrices for each
        angle `theta` and `phi` obtained by running
        `time_evolution` for each of them (with `time`
        time\-steps, `B0` magnetic intensity).

    """
    product_probabilities = np.zeros((len(theta), len(phi), len(time)), dtype=complex)

    iters = itertools.product(enumerate(theta), enumerate(phi))
    for (i, th), (j, ph) in tqdm(list(iters)):
        H_zee = sim.zeeman_hamiltonian(B0, th, ph)
        H = H_base + sim.convert(H_zee)
        init_rho = sim.initial_density_matrix(init_state, H)
        rho = sim.time_evolution(init_rho, time, H)
        product_probabilities[i, j] = sim.product_probability(obs_state, rho)
    return product_probabilities


def anisotropy(
    sim: HilbertSimulation,
    init_state: State,
    obs_state: State,
    time: np.ndarray,
    theta: ArrayLike,
    phi: ArrayLike,
    B0: float,
    D: np.ndarray,
    J: float,
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
) -> dict:
    """Anisotropy experiment.

    Args:

        init_state (State): Initial `State` of the density matrix.

        obs_state (State): Observable `State` of the density matrix.

        time (np.ndarray): An sequence of (uniform) time points,
            usually created using `np.arange` or `np.linspace`.

        theta (np.ndarray): rotation (polar) angle between the
            external magnetic field and the fixed molecule. See
            `zeeman_hamiltonian_3d`.

        phi (np.ndarray): rotation (azimuth) angle between the
            external magnetic field and the fixed molecule. See
            `zeeman_hamiltonian_3d`.

        B0 (float): External magnetic field intensity (milli
            Tesla) (see `zeeman_hamiltonian`).

        D (np.ndarray): Dipolar coupling constant (see
            `dipolar_hamiltonian`).

        J (float): Exchange coupling constant (see
            `exchange_hamiltonian`).

        kinetics (list): A list of kinetic (super)operators of
            type `radicalpy.kinetics.HilbertKineticsBase` or
            `radicalpy.kinetics.LiouvilleKineticsBase`.

        relaxations (list): A list of relaxation superoperators of
            type `radicalpy.relaxation.LiouvilleRelaxationBase`.

    Returns:
        dict:

        - time: the original `time` object
        - B0: `B0` parameter
        - theta: `theta` parameter
        - phi: `phi` parameter
        - rhos: tensor of sequences of time evolution of density
            matrices
        - time_evolutions: product probabilities
        - product_yields: product yields
        - product_yield_sums: product yield sums

    """
    H = sim.total_hamiltonian(B0=0, D=D, J=J, hfc_anisotropy=True)

    sim.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
    theta, phi = anisotropy_check(theta, phi)
    product_probabilities = anisotropy_loop(
        sim, init_state, obs_state, time, B0, H, theta=theta, phi=phi
    )
    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    return dict(
        time=time,
        B0=B0,
        theta=theta,
        phi=phi,
        time_evolutions=product_probabilities,
        product_yields=product_yields,
        product_yield_sums=product_yield_sums,
    )
