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
from .utils import (
    anisotropy_check,
    cidnp_polarisation_diffusion_model,
    cidnp_polarisation_exponential_model,
    cidnp_polarisation_truncated_diffusion_model,
    enumerate_spin_states_from_base,
    mary_lorentzian,
    modulated_signal,
    nmr_chemical_shift_imaginary_modulation,
    nmr_chemical_shift_real_modulation,
    nmr_scalar_coupling_modulation,
    nmr_t2_relaxation,
    reference_signal,
    s_t0_omega,
)


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
        H_zee = sim.zeeman_hamiltonian(B0, theta=th, phi=ph)
        H = H_base + sim.convert(H_zee)
        rho = sim.time_evolution(init_state, time, H)
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


def cidnp(
    B0: np.ndarray,
    deltag: float,
    cidnp_model: str,
    nucleus_of_interest: int,
    donor_hfc_spinhalf: float,
    acceptor_hfc_spinhalf: float,
    donor_hfc_spin1: float,
    acceptor_hfc_spin1: float,
    ks: float | None = None,
    alpha: float | None = None,
) -> (np.ndarray, np.ndarray):
    """
    CIDNP polarisation vs field for a radical pair with S-T0 mixing only.
    Args:

        B0 (np.ndarray): External magnetic field (T).

        deltag (float): Difference in g-value between the acceptor
            and donor.

        cidnp_model: Choose between CIDNP kinetic models. a) Exponential
            model. b) Truncated diffusion model. c) Full diffusion model.

        ks (float): Decay rate constant for the Exponential model (1/s).

        alpha (float): Parameter for the full diffusion model.

        nucleus_of_interest (int): The nucleus chosen for the simulation.

        donor_hfc_spinhalf (float): spin 1/2 HFCs (1H) for the donor (mT).

        acceptor_hfc_spinhalf (float): spin 1/2 HFCs (1H) for the acceptor (mT).

        donor_hfc_spin1 (float): spin 1 HFCs (14N) for the donor (mT).

        acceptor_hfc_spin1 (float): spin 1 HFCs (14N) for the acceptor (mT).

    Returns:
        B0 (T)
        polarisation (polarisation at each field point)
    """

    # Constants
    T_to_angular_frequency = (
        2.8e10 * 2.0 * np.pi
    )  # (T -> rad/s) for hyperfine conversion

    dnuc, anuc, dnuc1, anuc1 = (
        len(donor_hfc_spinhalf),
        len(acceptor_hfc_spinhalf),
        len(donor_hfc_spin1),
        len(acceptor_hfc_spin1),
    )
    nnuc = dnuc + anuc  # total spin-1/2
    nnuc1 = dnuc1 + anuc1  # total spin-1
    nnuct = nnuc + nnuc1

    # Spin-1/2 list, donor then acceptor, acceptor negated
    hfc_half = np.empty(nnuc, dtype=np.float64)
    if dnuc:
        hfc_half[:dnuc] = np.asarray(donor_hfc_spinhalf, dtype=np.float64) / 1e3
    if anuc:
        hfc_half[dnuc:] = -np.asarray(acceptor_hfc_spinhalf, dtype=np.float64) / 1e3

    # Spin-1 list
    hfc_one = np.empty(nnuc1, dtype=np.float64)
    if dnuc1:
        hfc_one[:dnuc1] = np.asarray(donor_hfc_spin1, dtype=np.float64) / 1e3
    if anuc1:
        hfc_one[dnuc1:] = -np.asarray(acceptor_hfc_spin1, dtype=np.float64) / 1e3

    # Convert to angular frequency
    if nnuc:
        hfc_half *= T_to_angular_frequency
    if nnuc1:
        hfc_one *= T_to_angular_frequency

    # Build hfcmod = all HFCs except the spin-1/2 nucleus of interest, then append spin-1
    assert (
        1 <= nucleus_of_interest <= max(nnuc, 1)
    ), "nucint out of range (1-based index into spin-1/2 list)"
    if nnuc <= 0:
        raise ValueError("At least one spin-1/2 nucleus is required.")

    idx0 = nucleus_of_interest - 1  # convert to 0-based
    hfcmod = np.empty(nnuct - 1, dtype=np.float64)
    # All spin-1/2 except the interest nucleus
    if idx0 > 0:
        hfcmod[:idx0] = hfc_half[:idx0]
    if idx0 < nnuc - 1:
        hfcmod[idx0 : nnuc - 1] = hfc_half[idx0 + 1 :]
    # append spin-1
    if nnuc1:
        hfcmod[nnuc - 1 :] = hfc_one

    # Base vector for mixed-radix enumeration: 2 for remaining spin-1/2, 3 for spin-1
    base = ([2] * (nnuc - 1)) + ([3] * nnuc1)  # length = nnuct - 1

    # Aall spin state patterns (total_states x (nnuct-1))
    if len(base) == 0:
        # Corner case: only one spin-1/2 nucleus (the one of interest), no others
        patterns = np.zeros((1, 0), dtype=np.float64)  # one "empty" pattern
    else:
        patterns = enumerate_spin_states_from_base(base)

    # nuc for every state: dot(pattern, hfcmod)
    nuc_all = patterns @ hfcmod  # shape: (total_states,)

    # precompute scale factor 2^(N-1) * 3^M
    scale = (2.0 ** max(nnuc - 1, 0)) * (3.0**nnuc1)

    # the HFC of the nucleus of interest (angular frequency)
    hfc_star = hfc_half[idx0]

    # Model checks
    if cidnp_model == "a" and ks is None:
        raise ValueError("Model 'a' requires ks.")
    if cidnp_model == "c" and alpha is None:
        raise ValueError("Model 'c' requires alpha.")

    # Compute polarisation vs field (vectorised over states)
    polarisation = np.empty_like(B0)
    for k, B in enumerate(B0):
        omega_plus, omega_minus = s_t0_omega(deltag, B, hfc_star, nuc_all)

        if cidnp_model == "a":
            p = cidnp_polarisation_exponential_model(ks, omega_plus, omega_minus)

        elif cidnp_model == "b":
            p = cidnp_polarisation_truncated_diffusion_model(omega_plus, omega_minus)

        else:  # model == "c"
            p = cidnp_polarisation_diffusion_model(omega_plus, omega_minus, alpha)

        polarisation[k] = p / scale

    return B0, polarisation


def epr(
    sim: HilbertSimulation,
    init_state: State,
    obs_state: State,
    time: np.ndarray,
    D: float,
    J: float,
    B0: np.ndarray,
    B1: float,
    B1_freq: float,
    B0_axis: str = "z",
    B1_axis: str = "x",
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
    hfc_anisotropy: bool = False,
) -> dict:
    H = sim.zeeman_hamiltonian(B0=-B1_freq, B_axis=B0_axis).astype(np.complex128)
    H += sim.zeeman_hamiltonian(B0=B1, B_axis=B1_axis).astype(np.complex128)
    H += sim.dipolar_hamiltonian(D=D)
    H += sim.exchange_hamiltonian(J=J)
    H += sim.hyperfine_hamiltonian(hfc_anisotropy)
    H = sim.convert(H)

    sim.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
    rhos = magnetic_field_loop(sim, init_state, time, H, B0, B_axis=B0_axis)
    product_probabilities = sim.product_probability(obs_state, rhos)

    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    MARY, LFE, HFE = mary_lfe_hfe(obs_state, B0, product_probabilities, dt, k)
    rhos = sim._square_liouville_rhos(rhos)

    return dict(
        time=time,
        B0=B0,
        B0_axis=B0_axis,
        B1=B1,
        B1_axis=B1_axis,
        B1_freq=B1_freq,
        B1_freq_axis=B0_axis,
        rhos=rhos,
        time_evolutions=product_probabilities,
        product_yields=product_yields,
        product_yield_sums=product_yield_sums,
        MARY=MARY,
        LFE=LFE,
        HFE=HFE,
    )


def magnetic_field_loop(
    sim: HilbertSimulation,
    init_state: State,
    time: np.ndarray,
    H_base: np.ndarray,
    B: np.ndarray,
    B_axis: str,
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
    H_zee = sim.convert(sim.zeeman_hamiltonian(1.0, B_axis, theta, phi))
    shape = sim._get_rho_shape(H_zee.shape[0])
    rhos = np.zeros([len(B), len(time), *shape], dtype=complex)
    for i, B0 in enumerate(tqdm(B)):
        H = H_base + B0 * H_zee
        H_sparse = sp.sparse.csc_matrix(H)
        rhos[i] = sim.time_evolution(init_state, time, H_sparse)
    return rhos


def mary_lfe_hfe(
    obs_state: State,
    B: np.ndarray,
    product_probability_seq: np.ndarray,
    dt: float,
    k: float,
    c: float = 0.0,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Calculate MARY, LFE, HFE."""
    MARY = np.sum(product_probability_seq, axis=1) * dt * k
    idx = int(len(MARY) / 2) if B[0] != 0 else 0
    minmax = max if obs_state == State.SINGLET else min
    HFE = (MARY[-1] - MARY[idx]) / (MARY[idx] + c) * 100
    LFE = (minmax(MARY) - MARY[idx]) / (MARY[idx] + c) * 100
    MARY = (MARY - MARY[idx]) / (MARY[idx] + c) * 100
    return MARY, LFE, HFE


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
    rhos = magnetic_field_loop(
        sim, init_state, time, H, B, B_axis="z", theta=theta, phi=phi
    )
    product_probabilities = sim.product_probability(obs_state, rhos)

    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    MARY, LFE, HFE = mary_lfe_hfe(obs_state, B, product_probabilities, dt, k)
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


def nmr(
    multiplets: list,
    spectral_width: float,
    number_of_points: float,
    fft_number: float,
    transmitter_frequency: float,
    carrier_position: float,
    linewidth: float,
    scale: float = 1.0,
) -> (np.ndarray, np.ndarray):
    
    # Derived quantities
    spectralwidth_inv = 1.0 / spectral_width
    acquisition_time = number_of_points * spectralwidth_inv
    # digital_resolution  = spectral_width / fft_number
    t2_relaxation_time = 1.0 / (np.pi * linewidth) if linewidth > 0 else 1e99
    reference_frequency = transmitter_frequency / (1.0 + carrier_position * 1.0e-6)

    # Time array 
    time = np.linspace(0.0, acquisition_time, number_of_points, endpoint=True)

    # Multiplet arrays
    if len(multiplets) > 0:
        arr = np.array(multiplets, dtype=float)
        nnuc = arr[:, 0] 
        f_hz = arr[:, 1] 
        mult = arr[:, 2].astype(int) 
        j_hz = arr[:, 3]  # J
    else:
        nnuc = np.zeros(0)
        f_hz = np.zeros(0)
        mult = np.zeros(0, dtype=int)
        j_hz = np.zeros(0)

    # Build FID (vectorised)
    if len(multiplets) > 0:
        cs_re = nmr_chemical_shift_real_modulation(f_hz, time)
        cs_im = nmr_chemical_shift_imaginary_modulation(f_hz, time)
        jpow = nmr_scalar_coupling_modulation(j_hz, time, mult - 1)
        decay = nmr_t2_relaxation(time, t2_relaxation_time)

        rfid = np.sum(nnuc[:, None] * cs_re * jpow, axis=0) * decay
        ifid = np.sum(nnuc[:, None] * cs_im * jpow, axis=0) * decay
    else:
        rfid = np.zeros(number_of_points)
        ifid = np.zeros(number_of_points)

    # Scale the first point
    rfid[0] *= 0.5
    ifid[0] *= 0.5

    # Zero-fill
    if fft_number > number_of_points:
        pad_len = fft_number - number_of_points
        rfid = np.concatenate([rfid, np.zeros(pad_len)])
        ifid = np.concatenate([ifid, np.zeros(pad_len)])

    # FFT
    fid = rfid + 1j * ifid
    spectrum = np.fft.fft(fid, n=fft_number) * scale

    # Frequency (MHz)
    i = np.arange(1, fft_number + 1, dtype=float)
    freq_mhz = (
        (transmitter_frequency * 1.0e6) 
        + (spectral_width / 2.0) 
        - ((i - 1.0) * spectral_width) / (fft_number - 1.0)
    ) / 1.0e6
    ppm = ((freq_mhz - reference_frequency) / reference_frequency) * 1.0e6
    return ppm, spectrum


def odmr(
    sim: HilbertSimulation,
    init_state: State,
    obs_state: State,
    time: np.ndarray,
    D: float,
    J: float,
    B0: float,
    B1: float,
    B1_freq: np.ndarray,
    B0_axis: str = "z",
    B1_axis: str = "x",
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
    hfc_anisotropy: bool = False,
) -> dict:
    H = sim.zeeman_hamiltonian(B0=B0, B_axis=B0_axis).astype(np.complex128)
    H += sim.zeeman_hamiltonian(B0=B1, B_axis=B1_axis).astype(np.complex128)
    H += sim.dipolar_hamiltonian(D=D)
    H += sim.exchange_hamiltonian(J=J)
    H += sim.hyperfine_hamiltonian(hfc_anisotropy)
    H = sim.convert(H)

    sim.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
    rhos = magnetic_field_loop(sim, init_state, time, H, -B1_freq, B_axis=B0_axis)
    product_probabilities = sim.product_probability(obs_state, rhos)

    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    MARY, LFE, HFE = mary_lfe_hfe(obs_state, B1_freq, product_probabilities, dt, k)
    rhos = sim._square_liouville_rhos(rhos)

    return dict(
        time=time,
        B0=B0,
        B0_axis=B0_axis,
        B1=B1,
        B1_axis=B1_axis,
        B1_freq=B1_freq,
        B1_freq_axis=B0_axis,
        rhos=rhos,
        time_evolutions=product_probabilities,
        product_yields=product_yields,
        product_yield_sums=product_yield_sums,
        MARY=MARY,
        LFE=LFE,
        HFE=HFE,
    )


def omfe(
    sim: HilbertSimulation,
    init_state: State,
    obs_state: State,
    time: np.ndarray,
    D: float,
    J: float,
    B1: float,
    B1_freq: np.ndarray,
    B1_axis: str = "x",
    B1_freq_axis: str = "z",
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
    hfc_anisotropy: bool = False,
) -> dict:
    H = sim.zeeman_hamiltonian(B0=B1, B_axis=B1_axis).astype(np.complex128)
    H += sim.dipolar_hamiltonian(D=D)
    H += sim.exchange_hamiltonian(J=J)
    H += sim.hyperfine_hamiltonian(hfc_anisotropy)
    H = sim.convert(H)

    sim.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
    rhos = magnetic_field_loop(sim, init_state, time, H, -B1_freq, B_axis=B1_freq_axis)
    product_probabilities = sim.product_probability(obs_state, rhos)

    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    MARY, LFE, HFE = mary_lfe_hfe(obs_state, B1_freq, product_probabilities, dt, k)
    rhos = sim._square_liouville_rhos(rhos)

    return dict(
        time=time,
        B1=B1,
        B1_axis=B1_axis,
        B1_freq=B1_freq,
        B1_freq_axis=B1_freq_axis,
        rhos=rhos,
        time_evolutions=product_probabilities,
        product_yields=product_yields,
        product_yield_sums=product_yield_sums,
        MARY=MARY,
        LFE=LFE,
        HFE=HFE,
    )


def oop_eseem(
    tau: float | np.ndarray, J: float, D: float, T1: float = np.inf, n_quad: int = 200
) -> np.ndarray:
    """Out-of-phase-electron-spin echo envelope modulation (OOP-ESEEM) simulation.
    Computes S(tau) ∝ exp(-tau/T1) * ∫_0^π sin( 2 [ J - D (cos^2 θ - 1/3) ] tau ) sinθ dθ
    using Gauss–Legendre quadrature on u = cosθ ∈ [-1, 1].

    Args:

        tau (float or np.ndarray): Time (s).

        J (float): Exchange interaction (rad/s).

        D (float): Dipolar coupling constant (rad/s).

        T1 (float): Longitudinal relaxation time constant (s). Use np.inf to disable the exponential decay.

        n_quad (int): Number of Gauss–Legendre nodes (accuracy increases with n).

    Returns:
        S (np.ndarray): OOP-ESEEM spectrum.
    """
    tau = np.atleast_1d(tau).astype(float)

    # Gauss–Legendre nodes/weights on [-1, 1]
    u, w = np.polynomial.legendre.leggauss(n_quad)  # u in [-1,1], weights w

    # Phase inside the sine: 2 * [ J - D (u^2 - 1/3) ] * tau
    # Broadcast over tau and u
    phi = 2.0 * (J - D * (u**2 - 1.0 / 3.0))  # shape (n_quad,)
    # integrand integrated over u: sin(phi * tau)
    # result over u for each tau: sum_w sin(phi*tau)
    S_int = np.sin(np.outer(tau, phi)) @ w  # shape (len(tau),)

    # exponential decay
    decay = np.exp(-tau / T1)

    S = decay * S_int
    return S if S.size > 1 else S.item()


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
    HHs = sim.semiclassical_HHs(num_samples)
    HJ = sim.exchange_hamiltonian(J)
    HD = sim.dipolar_hamiltonian(D)

    for i, B0 in enumerate(tqdm(Bs)):
        loop_yield = np.zeros((len(ts), len(init_state)), dtype=complex)
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


def rydmr(
    sim: HilbertSimulation,
    init_state: State,
    obs_state: State,
    time: np.ndarray,
    D: float,
    J: float,
    B0: np.ndarray,
    B1: float,
    B1_freq: float,
    B0_axis: str = "z",
    B1_axis: str = "x",
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
    hfc_anisotropy: bool = False,
) -> dict:
    H = sim.zeeman_hamiltonian(B0=-B1_freq, B_axis=B0_axis).astype(np.complex128)
    H += sim.zeeman_hamiltonian(B0=B1, B_axis=B1_axis).astype(np.complex128)
    H += sim.dipolar_hamiltonian(D=D)
    H += sim.exchange_hamiltonian(J=J)
    H += sim.hyperfine_hamiltonian(hfc_anisotropy)
    H = sim.convert(H)

    sim.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
    rhos = magnetic_field_loop(sim, init_state, time, H, B0, B_axis=B0_axis)
    product_probabilities = sim.product_probability(obs_state, rhos)

    sim.apply_hilbert_kinetics(time, product_probabilities, kinetics)
    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    MARY, LFE, HFE = mary_lfe_hfe(obs_state, B0, product_probabilities, dt, k)
    rhos = sim._square_liouville_rhos(rhos)

    return dict(
        time=time,
        B0=B0,
        B0_axis=B0_axis,
        B1=B1,
        B1_axis=B1_axis,
        B1_freq=B1_freq,
        B1_freq_axis=B0_axis,
        rhos=rhos,
        time_evolutions=product_probabilities,
        product_yields=product_yields,
        product_yield_sums=product_yield_sums,
        MARY=MARY,
        LFE=LFE,
        HFE=HFE,
    )


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
