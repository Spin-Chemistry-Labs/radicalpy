#!/usr/bin/env python
"""Experimental simulation routines for spin chemistry and magnetic resonance.

This module groups high-level experiment drivers and inner loops used to
simulate anisotropy scans, magnetically affected reaction yields (MARY),
EPR/ODMR/OMFE time-domain responses, RYDMR, steady-state signals, simple
NMR lineshapes, OOP-ESEEM envelopes, and hybrid semiclassical/quantum
MARY variants. Most functions assemble Hamiltonian/Liouvillian terms,
apply kinetics/relaxation superoperators, propagate density matrices, and
return structured results (dicts, arrays) ready for plotting and analysis.

Functions:
        - `anisotropy`: Full anisotropy experiment wrapper; returns time evolutions and yields.
        - `anisotropy_loop`: Inner loop over (θ, φ) orientations; propagates and returns product probabilities.
        - `cidnp`: CIDNP polarisation vs field for S–T0 mixing.
        - `coherent_control`: Coherent control of radical pair spin dynamics.
        - `epr`: CW/AC time-domain EPR vs B0 with B1 drive and frequency offset.
        - `field_switching`: Nanosecond field switching experiment (SEMF).
        - `kine_quantum_mary`: Hybrid kinetic+quantum MARY with stochastic hyperfine sampling.
        - `magnetic_field_loop`: Inner loop over swept field B; returns time-resolved density matrices.
        - `magnetic_field_loop_semiclassical`: Semiclassical inner loop over swept field B; returns time-resolved density matrices.
        - `mary_lfe_hfe`: Post-process product probabilities → MARY, low-/high-field effects (%).
        - `mary`: MARY vs B (time-domain propagation + yields + normalised response).
        - `mary_semiclassical`: Semiclassical MARY vs B (time-domain propagation + yields + normalised response).
        - `modulated_mary_brute_force`: Lock-in MARY via phase randomisation and numerical integration.
        - `nmr`: Simple 1D NMR synthesiser (FID → FFT) with multiplets and T2 decay.
        - `odmr`: ODMR vs RF frequency (B1_freq) at fixed B0.
        - `omfe`: Oscillating magnetic field effect vs RF frequency in transverse field.
        - `oop_eseem`: Out-of-phase ESEEM envelope via Gauss–Legendre quadrature.
        - `rydmr`: Reaction yield–detected magnetic resonance vs static field.
        - `semiclassical_mary`: Semiclassical MARY with explicit population channels.
        - `steady_state_mary`: Steady state via linear solve of Liouvillian (with ZFS/exchange/Zeeman).

Usage pattern:
        1) Build Hamiltonian components from your `Simulation` object
           (e.g., Zeeman, exchange J, dipolar D, hyperfine ± anisotropy).
        2) (Optional) Assemble kinetics/relaxation terms and apply to the
           Liouvillian via `apply_liouville_hamiltonian_modifiers(...)`.
        3) Propagate using an inner loop (`anisotropy_loop`, `magnetic_field_loop`)
           or a high-level routine (`mary`, `epr`, `odmr`, `omfe`, `rydmr`).
        4) Post-process to yields, MARY/LFE/HFE, or spectra; the helpers
           return a dict containing inputs, intermediates, and outputs.

Args conventions (selected):
        - `time` (np.ndarray): Uniform time grid (s).
        - `B`, `B0`, `B1`, `B1_freq` (array/float): Fields in mT unless noted.
        - `D`, `E`, `J` (float): Dipolar/ZFS/Exchange parameters (mT) for Zeeman-scaled forms,
          or (rad/s) where indicated (e.g., `oop_eseem`).
        - `theta`, `phi` (float or np.ndarray): Polar/azimuthal angles (rad) for anisotropy.
        - `kinetics`, `relaxations` (list): Collections of (super)operators compatible
          with the simulation API; applied in Liouville space.
        - `multiplets` (NMR): `[n_nuc, chem_shift_Hz, multiplicity, J_Hz]` per line.

Returns (typical):
        - Dictionaries with keys such as:
          `time`, `B`/`B0`/`B1_freq`, angles, `rhos`, `time_evolutions`
          (product probabilities), `product_yields`, `product_yield_sums`,
          and derived metrics (`MARY`, `LFE`, `HFE`).
        - Arrays for specialised routines (e.g., `(ppm, spectrum)` for `nmr`,
          MARY tensors for lock-in models, steady-state vectors for linear solves).

Notes:
        - **Spaces and shapes**: Propagation may occur in Liouville space; helpers such
          as `_square_liouville_rhos` are used to reshape back to Hilbert `(N, N)`
          for inspection. See each function’s docstring for shapes.
        - **Units**: Field parameters are mT unless otherwise specified; NMR chemical
          shifts are handled via Hz/MHz → ppm conversions; `oop_eseem` uses rad/s.
        - **Performance**: Inner loops precompute unit-field Zeeman operators and reuse
          sparse CSC matrices for repeated exponentials where possible.
        - **CIDNP models**: `cidnp_model` ∈ {"a","b","c"} selects exponential, truncated
          diffusion, or full diffusion variants and requires `ks` or `alpha` accordingly.
        - **Lock-in MARY**: `modulated_mary_brute_force` performs phase randomisation,
          envelope weighting, and numerical integration to estimate RMS harmonic responses.

References:
        - [Antill & Vatai, *J. Chem. Theory Comput.* **20**, 9488–9499 (2024)](https://doi.org/10.1021/acs.jctc.4c00887).
        - [Konowalczyk et al., *PCCP* **23**, 1273–1284 (2021)](https://doi.org/10.1039/D0CP04814C).
        - [Maeda et al., *Mol. Phys.* **104**, 1779–1788 (2006)](https://doi.org/10.1080/14767050600588106).
        - [Masuzawa et al., *J. Chem. Phys.* **152**, 014301 (2020)](https://doi.org/10.1063/1.5131557).

Requirements:
        - `numpy`, `scipy` (sparse algebra, matrix exponentials), and a simulation object
          implementing APIs such as `zeeman_hamiltonian`, `exchange_hamiltonian`,
          `dipolar_hamiltonian`, `hyperfine_hamiltonian`, `projection_operator`,
          `convert`, `time_evolution`, `product_probability`, `product_yield`,
          and modifier application hooks.

See also:
        - `plot.py` for visualization helpers.
        - `kinetics.py`, `relaxation.py` for incoherent processes added to Liouvillians.
        - Individual function docstrings within this module for precise signatures,
          units, and return structures.

"""

import itertools
from typing import Dict, Optional, Sequence

import numpy as np
import scipy as sp
import scipy.sparse as sps
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from tqdm import tqdm

from .shared import constants as C
from .simulation import (
    Basis,
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
    """CIDNP polarisation vs field for a radical pair with S-T0
    mixing only.


    Args:

        B0: External magnetic field (T).

        deltag: Difference in g-value between the acceptor and donor.

        cidnp_model: Choose between CIDNP kinetic models. a) Exponential
            model. b) Truncated diffusion model. c) Full diffusion model.

        ks: Decay rate constant for the Exponential model (1/s).

        alpha: Parameter for the full diffusion model.

        nucleus_of_interest: The nucleus chosen for the simulation.

        donor_hfc_spinhalf: spin 1/2 HFCs (1H) for the donor (mT).

        acceptor_hfc_spinhalf: spin 1/2 HFCs (1H) for the acceptor (mT).

        donor_hfc_spin1: spin 1 HFCs (14N) for the donor (mT).

        acceptor_hfc_spin1: spin 1 HFCs (14N) for the acceptor (mT).

    Returns:

        B0 (T) and polarisation (polarisation at each field point)

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


def coherent_control(
    sim,
    *,
    init_state,
    obs_state,
    time: np.ndarray,
    sticks_A_freq: Sequence[float],
    sticks_A_int: Sequence[float],
    sticks_B_freq: Sequence[float],
    sticks_B_int: Sequence[float],
    B1_G: float = 200.0,  # gauss
    g_e: float = 2.0023,
    k_s: float = 2e6,
    J: float = 0.0,
    dt_override: Optional[float] = None,
    u_max_factor: float = 1.0,
    u_smooth: float = 0.2,
) -> Dict[str, np.ndarray]:
    r"""
    Coherent microwave-feedback simulation.

    It treats the radical pair as an **ensemble** of frequency-configurations.
    Each configuration (A-stick × B-stick) is propagated in Hilbert space under

    .. math::

        H_{ij}(t) = \omega_{A,i} S_{zA} + \omega_{B,j} S_{zB} + H_J + u(t)(S_{xA}+S_{xB})

    where

    - the frequencies :math:`\omega_{A,i}`, :math:`\omega_{B,j}` come from the
      stick lists ``sticks_A_freq`` and ``sticks_B_freq`` (in rad/s),
    - the weights of each configuration are the products of the normalised
      stick intensities ``sticks_A_int`` and ``sticks_B_int``,
    - ``u(t)`` is a *global* feedback field, common to all configurations,
      built from the ensemble signal.

    The feedback field is computed at every time step as

    .. math::

        u_\text{raw}(t)
        = B_{1c}\, \sum_{ij} w_{ij} \, \Im \operatorname{Tr}
            \left[ P_\text{target}( V \rho_{ij} - \rho_{ij} V ) \right],

    then clipped to ``±u_max_factor * B1c`` and optionally smoothed in time by
    an exponential smoother. Here

    - ``V = S_xA + S_xB`` is the microwave operator,
    - ``P_target = |αβ⟩⟨αβ| + |βα⟩⟨βα|`` is the “target” projector,
    - ``B1c = g_e μ_B B1_T / ħ`` is the on-resonance Rabi frequency, with
      ``B1_G`` given in gauss.

    Time propagation is done configuration-by-configuration with an RK4 step
    on the user-supplied time grid.

    Args:

        sim : HilbertSimulation

        init_state : State
            Spin state used both for the initial density.

        obs_state : State
            Spin state to monitor as an observable population during the run.
            A projector onto this state is evaluated on every configuration and
            then averaged over the ensemble.

        time : ndarray
            1D array of time points (s). The integration advances along this grid.
            If ``dt_override`` is not given, the step size is inferred from
            ``time[1] - time[0]``.

        sticks_A_freq, sticks_B_freq : sequence of float
            Stick frequencies for radical A and radical B, respectively, in
            **angular frequency** units (rad/s), e.g. ``2*pi*1e6*[ ... MHz ... ]``.

        sticks_A_int, sticks_B_int : sequence of float
            Corresponding intensities for the two stick sets. Each list is
            normalised so that its entries sum to 1. The weight of configuration
            (i, j) is then the product of the two normalised intensities.

        B1_G : float, optional
            Microwave amplitude in **gauss**. Default is ``200.0``.

        g_e : float, optional
            Electron g-value used in the Rabi-frequency conversion. Default 2.0023.

        k_s : float, optional
            Haberkorn recombination rate applied with respect to ``init_state``
            (s⁻¹). Default ``2e6``.

        J : float, optional
            Exchange coupling (mT).

        dt_override : float, optional
            If provided, use this as the integration time step instead of
            ``time[1] - time[0]``.

        u_max_factor : float, optional
            Hard clip on the feedback: ``|u(t)| ≤ u_max_factor * B1c``.
            Useful to suppress large initial spikes. Default 1.0.

        u_smooth : float, optional
            Exponential smoothing factor for the feedback in (0, 1). Larger values
            give smoother control fields. Default 0.2.

    Returns:

        dict of str -> ndarray
            A dictionary with the following entries:

            - ``"time"``: Time axis (s)
            - ``"u"``: Feedback field (rad/s)
            - ``"target"``: nsemble-average target population
            - ``"each_target"``: Target population per configuration
            - ``"weights"``: Configuration weights
            - ``"obs"``: Ensemble-average population of ``obs_state``
            - ``"each_obs"``: Per-configuration population of ``obs_state``
            - ``"population"``: Ensemble-average trace of ρ
            - ``"each_population"``: Per-configuration trace
    """
    mu_B = C.mu_B
    hbar = C.hbar

    if dt_override is not None:
        dt = float(dt_override)
    else:
        if len(time) < 2:
            raise ValueError("time needs >=2 points")
        dt = float(time[1] - time[0])

    # normalise stick intensities
    sticks_A_int = np.asarray(sticks_A_int, float)
    sticks_B_int = np.asarray(sticks_B_int, float)
    sticks_A_int /= sticks_A_int.sum()
    sticks_B_int /= sticks_B_int.sum()
    sticks_A_freq = np.asarray(sticks_A_freq, float)
    sticks_B_freq = np.asarray(sticks_B_freq, float)

    SnumA = len(sticks_A_freq)
    SnumB = len(sticks_B_freq)
    Snum = SnumA * SnumB

    SxA = sim.spin_operator(0, "x")
    SzA = sim.spin_operator(0, "z")
    SxB = sim.spin_operator(1, "x")
    SzB = sim.spin_operator(1, "z")

    # microwave operator
    V = SxA + SxB

    Target = np.zeros((4, 4), complex)
    Target[1, 1] = 1.0
    Target[2, 2] = 1.0

    init_proj = sim.projection_operator(init_state)
    obs_proj = sim.projection_operator(obs_state)

    def Lk(rho):
        return -0.5 * k_s * (init_proj @ rho + rho @ init_proj)

    B1_T = B1_G * 1e-4
    B1c = (g_e * mu_B * B1_T) / hbar  # rad/s
    u_cap = u_max_factor * B1c

    if abs(J) > 0.0:
        H_J = sim.exchange_hamiltonian(J)
    else:
        H_J = np.zeros_like(SxA)

    H0_list = []
    weights = []
    for i in range(SnumA):
        for j in range(SnumB):
            H0 = sticks_A_freq[i] * SzA + sticks_B_freq[j] * SzB + H_J
            H0_list.append(H0)
            weights.append(sticks_A_int[i] * sticks_B_int[j])
    weights = np.asarray(weights, float)

    lam = 0.001
    rho0 = (obs_proj + 3.0 * lam * V) / 3.0
    rho_list = [rho0.copy() for _ in range(Snum)]

    Tlen = len(time)
    u_arr = np.zeros(Tlen, float)

    target_arr = np.zeros(Tlen, float)
    each_target = np.zeros((Snum, Tlen), float)

    obs_arr = np.zeros(Tlen, float)
    each_obs = np.zeros((Snum, Tlen), float)

    population_arr = np.zeros(Tlen, float)
    each_population = np.zeros((Snum, Tlen), float)

    def drho_dt(H, u, rho):
        Ht = H + u * V
        comm = Ht @ rho - rho @ Ht
        return -1j * comm + Lk(rho)

    u_prev = 0.0
    for t_idx, t in enumerate(tqdm(time)):
        signal = 0.0
        for m, rho in enumerate(rho_list):
            comm = V @ rho - rho @ V
            s = np.imag(np.trace(Target @ comm))
            signal += weights[m] * s
        u_raw = B1c * signal

        if u_raw > u_cap:
            u_raw = u_cap
        elif u_raw < -u_cap:
            u_raw = -u_cap

        if t_idx == 0:
            u_t = u_raw
        else:
            u_t = u_smooth * u_prev + (1.0 - u_smooth) * u_raw
        u_prev = u_t
        u_arr[t_idx] = u_t

        pop_t = 0.0
        for m, rho in enumerate(rho_list):
            each_target[m, t_idx] = np.real(np.trace(Target @ rho))
            each_obs[m, t_idx] = np.real(np.trace(obs_proj @ rho))
            tr_m = np.real(np.trace(rho))
            each_population[m, t_idx] = tr_m
            pop_t += weights[m] * tr_m

        target_arr[t_idx] = float(np.sum(weights * each_target[:, t_idx]))
        obs_arr[t_idx] = float(np.sum(weights * each_obs[:, t_idx]))
        population_arr[t_idx] = pop_t

        if t_idx == Tlen - 1:
            break

        new_rho_list = []
        for m, rho in enumerate(rho_list):
            H0 = H0_list[m]
            # RK4
            k1 = drho_dt(H0, u_t, rho)
            k2 = drho_dt(H0, u_t, rho + 0.5 * dt * k1)
            k3 = drho_dt(H0, u_t, rho + 0.5 * dt * k2)
            k4 = drho_dt(H0, u_t, rho + dt * k3)
            rho_next = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # symmetrise
            rho_next = 0.5 * (rho_next + rho_next.conj().T)

            new_rho_list.append(rho_next)
        rho_list = new_rho_list

    return {
        "time": time,
        "u": u_arr,
        "target": target_arr,
        "each_target": each_target,
        "weights": weights,
        "obs": obs_arr,
        "each_obs": each_obs,
        "population": population_arr,
        "each_population": each_population,
    }


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
    """Electron paramagnetic resonance (EPR) time‐domain simulation with CW/AC fields.

    Args:

        sim (HilbertSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        obs_state (State): Observable `State` of the density matrix.

        time (np.ndarray): Sequence of (uniform) time points.

        D (float): Dipolar coupling constant (mT).

        J (float): Exchange coupling constant (mT).

        B0 (np.ndarray): Static magnetic field sweep (mT) applied along `B0_axis`.

        B1 (float): RF/AC field amplitude (mT) applied along `B1_axis`.

        B1_freq (float): RF/AC angular frequency offset (mT). Implemented as
            an effective Zeeman term with negative sign on `B0_axis`.

        B0_axis (str): Axis for `B0` Zeeman term (`'x'|'y'|'z'`).

        B1_axis (str): Axis for `B1` Zeeman term (`'x'|'y'|'z'`).

        kinetics (list): List of kinetic superoperators.

        relaxations (list): List of relaxation superoperators.

        hfc_anisotropy (bool): Include anisotropic hyperfine Hamiltonian if True.

    Returns:
        dict:

        - time: original `time`
        - B0: sweep values
        - B0_axis: axis of `B0`
        - B1: RF amplitude
        - B1_axis: axis of `B1`
        - B1_freq: RF angular frequency offset
        - B1_freq_axis: axis used for the offset term
        - rhos: density matrices (squared to Hilbert shape)
        - time_evolutions: product probabilities vs time & field
        - product_yields: integrated product yields
        - product_yield_sums: scalar yield per field
        - MARY: normalised magnetoresponse
        - LFE: low field effect (%)
        - HFE: high field effect (%)
    """
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


def field_switching(
    sim: HilbertSimulation,
    *,
    B_on: float = 2.0,
    B_off: float = 0.0,
    init_state: State = State.TRIPLET,
    dt: float = 1e-9,
    n_offsets: int = 100,
    offset_step: int = 10,
    pulse_width_steps: int = 800,
    k_rec: float = 50e6,
    k_esc: float = 2e6,
    J: float = 0.0,
    D: float = 0.0,
) -> dict:
    """Nanosecond field-switching experiment. Switched External
    Magnetic Field (SEMF) simulation.

    Args:

        sim: Sim object.

        B_on, B_off: Magnetic fields (mT) for the “before switch” part and the “after switch”
            part, respectively.

        dt: Time step (s).

        n_offsets: Number of different switch times to scan.

        offset_step: Number of time steps to increase the switch time by each iteration.

        pulse_width_steps: Length (in steps) of the second part of the evolution (the “pulse”).

        k_rec, k_esc: Haberkorn recombination and escape rates (s⁻¹).

        J, D: Exchange and dipolar couplings.

    Returns:

            A `dict` with the following keys (dims):

            - `time`: (T,),

            - `switch_times`: (n_offsets,),

            - `TA_on`: (T, n_offsets),

            - `TA_off`: (T, n_offsets),

            - `TA_diff`: (T, n_offsets),


    """
    H_on = sim.total_hamiltonian(B0=B_on, J=J, D=D)
    H_off = sim.total_hamiltonian(B0=B_off, J=J, D=D)

    P_S = sim.projection_operator(State.SINGLET)
    dim = P_S.shape[0]
    I = np.eye(dim, dtype=complex)

    decay = -0.5j * (k_rec * P_S + k_esc * I)

    H_on_NH = H_on + decay
    H_off_NH = H_off + decay

    U_on = expm(-1j * H_on_NH * dt)
    U_off = expm(-1j * H_off_NH * dt)

    rho0 = sim.projection_operator(init_state)
    rho0 = rho0 / np.trace(rho0)

    max_steps = (n_offsets - 1) * offset_step + pulse_width_steps
    time = np.arange(max_steps, dtype=float) * dt

    TA_on = np.zeros((max_steps, n_offsets), dtype=float)
    TA_off = np.zeros((max_steps, n_offsets), dtype=float)

    for i in range(n_offsets):
        switch_steps = i * offset_step

        rho_on = rho0.copy()
        rho_off = rho0.copy()

        for k in range(max_steps):
            TA_on[k, i] = float(np.real(np.trace(rho_on)))
            TA_off[k, i] = float(np.real(np.trace(rho_off)))

            if k < switch_steps:
                rho_on = U_on @ rho_on @ U_on.conj().T
                rho_off = U_on @ rho_off @ U_on.conj().T
            else:
                rho_on = U_off @ rho_on @ U_off.conj().T
                rho_off = U_on @ rho_off @ U_on.conj().T

    TA_diff = TA_on - TA_off
    switch_times = np.arange(n_offsets, dtype=float) * offset_step * dt

    return {
        "time": time,
        "switch_times": switch_times,
        "TA_on": TA_on,
        "TA_off": TA_off,
        "TA_diff": TA_diff,
        "SEMF": TA_diff.sum(axis=0),
    }


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

        sim (HilbertSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        time (np.ndarray): A sequence of (uniform) time points,
            usually created using `np.arange` or `np.linspace` (s).

        H_base (np.ndarray): A "base" Hamiltonian or Liouvillian that
            does not include the swept Zeeman term (e.g. the result of
            `total_hamiltonian(...)` with `B0=0` and any static terms).

        B (np.ndarray): Magnetic-field sweep values (mT). Each value
            scales a precomputed unit-field Zeeman operator on `B_axis`.

        B_axis (str): Axis of the Zeeman term (`'x'|'y'|'z'`).

        theta (float, optional): Rotation (polar) angle between the external
            magnetic field and the fixed molecular frame. See
            `zeeman_hamiltonian_3d`.

        phi (float, optional): Rotation (azimuth) angle between the external
            magnetic field and the fixed molecular frame. See
            `zeeman_hamiltonian_3d`.

        hfc_anisotropy (bool): Reserved for API symmetry; anisotropic hyperfine
            terms should be included in `H_base` if required (not used here).

    Returns:
        np.ndarray:

        A tensor of density matrices with shape
        `(len(B), len(time), *rho_shape)`, where `rho_shape` is
        `(N, N)` for Hilbert-space propagation and `(N,)` for
        Liouville-space propagation. For each field value `B[i]`, the time
        evolution is obtained by propagating under
        `H = H_base + B[i] * H_Z(1.0)` with `H_Z(1.0)` the unit-field Zeeman
        operator on `B_axis`.

    Notes:
        - The Zeeman operator for a unit field is computed once and reused
          for all `B[i]`, then scaled and added to `H_base`.
        - Propagation uses a CSC sparse representation for efficiency.
    """
    H_zee = sim.convert(sim.zeeman_hamiltonian(1.0, B_axis, theta, phi))
    shape = sim._get_rho_shape(H_zee.shape[0])
    rhos = np.zeros([len(B), len(time), *shape], dtype=complex)
    for i, B0 in enumerate(tqdm(B)):
        H = H_base + B0 * H_zee
        H_sparse = sp.sparse.csc_matrix(H)
        rhos[i] = sim.time_evolution(init_state, time, H_sparse)
    return rhos


def magnetic_field_loop_semiclassical(
    sim: HilbertSimulation,
    init_state: State,
    time: np.ndarray,
    B: np.ndarray,
    H_base: np.ndarray,
    kinetics: list[HilbertIncoherentProcessBase] = [],
    relaxations: list[HilbertIncoherentProcessBase] = [],
    theta: Optional[float] = None,
    phi: Optional[float] = None,
    num_samples: Optional[int] = None,
) -> np.ndarray:
    """Generate density matrices (rhos) for MARY using a semiclassical
    ensemble of hyperfine realisations and incoherent processes.

    Args:

        sim (HilbertSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        time (np.ndarray): A sequence of (uniform) time points,
            usually created using `np.arange` or `np.linspace` (s).

        B (np.ndarray): Magnetic-field sweep values (mT). Each value
            scales a precomputed unit-field Zeeman operator summed over
            all radicals on the laboratory `'z'` axis (after applying the
            optional `theta/phi` rotation).

        H_base (np.ndarray): A "base" Hamiltonian that does not include
            the swept Zeeman term, e.g., the result of
            `total_hamiltonian(...)` with `B0=0` plus any static terms.

        kinetics (list[HilbertIncoherentProcessBase]): List of incoherent
            kinetic processes (e.g., recombination, intersystem crossing)
            to be added as Liouvillian modifiers. Defaults to `[]`.

        relaxations (list[HilbertIncoherentProcessBase]): List of
            relaxation/dephasing processes to be added as Liouvillian
            modifiers. Defaults to `[]`.

        theta (float, optional): Rotation (polar) angle between the
            external magnetic field and the fixed molecular frame. See
            `zeeman_hamiltonian_3d`.

        phi (float, optional): Rotation (azimuth) angle between the
            external magnetic field and the fixed molecular frame. See
            `zeeman_hamiltonian_3d`.

        num_samples (int, optional): Number of semiclassical hyperfine
            realizations used for ensemble averaging. If `None`, the
            simulator’s default is used by `sim.semiclassical_HHs(...)`.

    Returns:
        np.ndarray:

        A tensor of density matrices with shape
        `(len(B), len(time), *rho_shape)`, where `rho_shape` is
        `(N, N)` for Hilbert-space propagation and `(N,)` for
        Liouville-space propagation. For each field value `B[i]`,
        the result is the **sample-average** over `num_samples`
        semiclassical Hamiltonians `H_HH[k]` of the time evolution under

        `H_t = H_base + B[i] * H_Z(1.0) + H_HH[k]`,

        with incoherent `kinetics + relaxations` applied as Liouvillian
        modifiers.

    Notes:
        - The unit-field Zeeman operator is constructed by summing the
          per-radical contributions on the `'z'` axis (after rotation by
          `theta/phi`) and is then scaled by each `B[i]`.
        - An ensemble `HHs = sim.semiclassical_HHs(num_samples)` provides
          the semiclassical hyperfine Hamiltonians used for averaging.
        - `sim.convert(...)` maps the total Hamiltonian to the working
          representation (Hilbert→Liouville if required); incoherent
          processes are then added via
          `sim.apply_liouville_hamiltonian_modifiers(...)`.
        - Propagation uses a CSC sparse representation for efficiency, and
          the final array at index `i` is the mean across all samples for
          that field value.
    """

    H_zee = sim.zeeman_hamiltonian(1.0, "z", theta, phi)

    HHs = sim.semiclassical_HHs(num_samples)
    shape = sim._get_rho_shape(H_zee.shape[0] ** 2)
    average_rhos = np.zeros([len(B), len(time), *shape], dtype=complex)

    for i, B0 in enumerate(tqdm(B)):
        loop_rhos = np.zeros([len(HHs), len(time), *shape], dtype=complex)
        Hz = B0 * H_zee
        for k, HH in enumerate(HHs):
            Ht = Hz + HH + H_base
            L = sim.convert(Ht)

            sim.apply_liouville_hamiltonian_modifiers(L, kinetics + relaxations)
            L_sparse = sp.sparse.csc_matrix(L)
            rhos = sim.time_evolution(init_state, time, L_sparse)
            loop_rhos[k] = rhos
        average_rhos[i] = loop_rhos.mean(axis=0)

    return average_rhos


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
    minmax = min if obs_state == State.SINGLET else max
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
    """Magnetically affected reaction yield (MARY) simulation over a magnetic field sweep.

    Args:

        sim (HilbertSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        obs_state (State): Observable `State` of the density matrix.

        time (np.ndarray): Sequence of (uniform) time points (s).

        B (np.ndarray): Magnetic field sweep values (mT) along z-axis by default.

        D (float): Dipolar coupling constant (mT).

        J (float): Exchange coupling constant (mT).

        kinetics (list): List of kinetic superoperators.

        relaxations (list): List of relaxation superoperators.

        theta (float, optional): Polar angle for the Zeeman term (anisotropy).

        phi (float, optional): Azimuthal angle for the Zeeman term (anisotropy).

        hfc_anisotropy (bool): Include anisotropic hyperfine Hamiltonian if True.

    Returns:
        dict:

        - time: original `time`
        - B: sweep values
        - theta: `theta` parameter
        - phi: `phi` parameter
        - rhos: density matrices (squared to Hilbert shape)
        - time_evolutions: product probabilities
        - product_yields: product yields
        - product_yield_sums: product yield sums
        - MARY: normalised magnetoresponse
        - LFE: low field effect (%)
        - HFE: high field effect (%)
    """
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


def mary_semiclassical(
    sim: SemiclassicalSimulation,
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
    num_samples: Optional[int] = None,
    c: Optional[float] = None,
) -> dict:
    """
    Compute a MARY (Magnetically Affected Reaction Yield) curve using the
    semiclassical simulation pipeline.

    This routine constructs a field-independent base Hamiltonian
    :math:`H = H_J + H_D` from the exchange and dipolar interactions, then
    performs a semiclassical magnetic-field sweep, evolving the state from
    ``init_state`` over the time grid for each field value in ``B``. Product
    probabilities are converted to integrated yields and post-processed into
    the standard MARY, low-field effect (LFE), and high-field effect (HFE)
    metrics.

    Args:

        sim : SemiclassicalSimulation
            Semiclassical simulator providing Hamiltonian terms and evolution
            back-ends (including stochastic sampling where applicable).

        init_state : State
            Initial state for time evolution (e.g., ``State.SINGLET``).

        obs_state : State
            Observable (product) state whose probability/yield is reported.

        time : ndarray, shape (T,)
            Monotonic time grid for propagation (units consistent with Hamiltonians).

        B : ndarray, shape (NB,)
            Magnetic-field grid for the sweep (scalar magnitude in the chosen axis).

        D : float
            Dipolar interaction strength used to build ``H_D``.

        J : float
            Exchange interaction strength used to build ``H_J``.

        kinetics : list[HilbertIncoherentProcessBase], optional
            Kinetic processes to apply during evolution (e.g., Haberkorn).

        relaxations : list[HilbertIncoherentProcessBase], optional
            Relaxation processes to apply during evolution (semiclassical variants).

        theta : float, optional
            Polar angle (radians) of the magnetic-field axis relative to the lab frame.
            If ``None``, the simulator default is used.

        phi : float, optional
            Azimuthal angle (radians) of the magnetic-field axis.
            If ``None``, the simulator default is used.

        num_samples : int, optional
            Number of stochastic samples/trajectories for the semiclassical averaging.
            If ``None``, the simulator default is used.

        c : float, optional
            Normalisation/contrast parameter forwarded to ``mary_lfe_hfe`` for
            MARY post-processing.

    Returns:

        dict
            A results dictionary with keys:
            - ``time`` : ndarray, the input time grid.
            - ``B`` : ndarray, the field grid.
            - ``theta`` / ``phi`` : angles used for the sweep.
            - ``rhos`` : list/ndarray of density matrices (reshaped to square form).
            - ``time_evolutions`` : ndarray, product probabilities vs. ``time`` and ``B``.
            - ``product_yields`` : ndarray, integrated yields vs. ``B``.
            - ``product_yield_sums`` : float or ndarray, sum.
            - ``MARY`` : ndarray, normalised magnetoresponse vs. ``B``.
            - ``LFE`` : float, low-field effect (%).
            - ``HFE`` : float, high-field effect (%).
    """
    HJ = sim.exchange_hamiltonian(J)
    HD = sim.dipolar_hamiltonian(D)
    H = HJ + HD

    rhos = magnetic_field_loop_semiclassical(
        sim,
        init_state,
        time,
        B,
        H,
        kinetics,
        relaxations,
        theta=theta,
        phi=phi,
        num_samples=num_samples,
    )
    product_probabilities = sim.product_probability(obs_state, rhos)

    k = kinetics[0].rate_constant if kinetics else 1.0
    product_yields, product_yield_sums = sim.product_yield(
        product_probabilities, time, k
    )

    dt = time[1] - time[0]
    MARY, LFE, HFE = mary_lfe_hfe(obs_state, B, product_probabilities, dt, k, c)
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
    """Lock-in detected MARY via brute-force phase randomisation and numerical integration.

    Source: `Konowalczyk et al. Phys. Chem. Chem. Phys. 23, 1273-1284 (2021)`_.

    Args:

        Bs (np.ndarray): Array of bias fields at which to compute the MARY signal (G).

        modulation_depths (list): List of field modulation amplitudes (G).

        modulation_frequency (float): Field modulation angular frequency (Hz).

        time_constant (float): Lock-in amplifier time constant used for exponential
            weighting of the reference (s).

        harmonics (list): Harmonic indices (e.g., [1,2]) for which to compute responses.

        lfe_magnitude (float): Amplitude parameter controlling the Lorentzian LFE line shape.

    Returns:
        np.ndarray:

        Tensor of shape `(len(harmonics), len(modulation_depths), len(Bs))` containing
        RMS lock-in amplitudes at each harmonic, modulation depth, and bias field.

    .. _Konowalczyk et al. Phys. Chem. Chem. Phys. 23, 1273-1284 (2021):
       https://doi.org/10.1039/D0CP04814C
    """
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
    """Simple 1D NMR spectrum synthesiser (FID → FFT) with multiplets and T2 decay.

    Args:

        multiplets (list): Each entry `[number of nuclei, chemical shift, multiplicity, scalar coupling]`.

        spectral_width (float): Spectral width (Hz).

        number_of_points (float): Number of acquired points in the time domain.

        fft_number (float): Zero-filled length for the FFT (≥ `number_of_points`).

        transmitter_frequency (float): Transmitter frequency (MHz).

        carrier_position (float): Carrier position (ppm) used to define the reference.

        linewidth (float): Lorentzian linewidth (Hz) → T2 = 1/(π·linewidth).

        scale (float): Overall multiplicative scale on the final spectrum.

    Returns:
        (np.ndarray, np.ndarray):

        - ppm: Chemical shift axis (ppm).
        - spectrum: Complex spectrum after FFT (same length as `fft_number`).
    """
    spectralwidth_inv = 1.0 / spectral_width
    acquisition_time = number_of_points * spectralwidth_inv
    # digital_resolution  = spectral_width / fft_number
    t2_relaxation_time = 1.0 / (np.pi * linewidth) if linewidth > 0 else 1e99
    reference_frequency = transmitter_frequency / (1.0 + carrier_position * 1.0e-6)

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

    # Build FID
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

    if fft_number > number_of_points:
        pad_len = fft_number - number_of_points
        rfid = np.concatenate([rfid, np.zeros(pad_len)])
        ifid = np.concatenate([ifid, np.zeros(pad_len)])

    fid = rfid + 1j * ifid
    spectrum = np.fft.fft(fid, n=fft_number) * scale

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
    """Optically detected magnetic resonance (ODMR) simulation vs RF frequency.

    Args:

        sim (HilbertSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        obs_state (State): Observable `State` of the density matrix.

        time (np.ndarray): Sequence of (uniform) time points (s).

        D (float): Dipolar coupling constant (mT).

        J (float): Exchange coupling constant (mT).

        B0 (float): Static magnetic field (mT) along `B0_axis`.

        B1 (float): RF/AC field amplitude (mT) along `B1_axis`.

        B1_freq (np.ndarray): RF angular frequency sweep (mT).

        B0_axis (str): Axis for `B0` Zeeman term.

        B1_axis (str): Axis for `B1` Zeeman term.

        kinetics (list): List of kinetic superoperators.

        relaxations (list): List of relaxation superoperators.

        hfc_anisotropy (bool): Include anisotropic hyperfine Hamiltonian if True.

    Returns:
        dict:

        - time: original `time`
        - B0: static field value
        - B0_axis: axis of `B0`
        - B1: RF amplitude
        - B1_axis: axis of `B1`
        - B1_freq: RF sweep values
        - B1_freq_axis: axis used for the RF sweep
        - rhos: density matrices (squared to Hilbert shape)
        - time_evolutions: product probabilities vs time & field
        - product_yields: integrated product yields
        - product_yield_sums: scalar yield per field
        - MARY: normalised magnetoresponse
        - LFE: low field effect (%)
        - HFE: high field effect (%)
    """
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
    """Oscillating magnetic field effect (OMFE) simulation vs RF frequency in transverse field.

    Args:

        sim (HilbertSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        obs_state (State): Observable `State` of the density matrix.

        time (np.ndarray): Sequence of (uniform) time points (s).

        D (float): Dipolar coupling constant (mT).

        J (float): Exchange coupling constant (mT).

        B1 (float): Oscillating field amplitude (mT) along `B1_axis`.

        B1_freq (np.ndarray): Angular frequency sweep (mT) applied along `B1_freq_axis`.

        B1_axis (str): Axis for the static `B1` Zeeman term.

        B1_freq_axis (str): Axis used for the frequency-sweep effective term.

        kinetics (list): List of kinetic superoperators.

        relaxations (list): List of relaxation superoperators.

        hfc_anisotropy (bool): Include anisotropic hyperfine Hamiltonian if True.

    Returns:
        dict:

        - time: original `time`
        - B1: RF amplitude
        - B1_axis: axis of `B1`
        - B1_freq: RF sweep values
        - B1_freq_axis: axis used for the RF sweep
        - rhos: density matrices (squared to Hilbert shape)
        - time_evolutions: product probabilities vs time & field
        - product_yields: integrated product yields
        - product_yield_sums: scalar yield per field
        - MARY: normalised magnetoresponse
        - LFE: low field effect (%)
        - HFE: high field effect (%)
    """
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
    """Kinetic + quantum MARY (hybrid) with sampling over stochastic hyperfine fields.

    Source: `Antill and Vatai, J. Chem. Theory Comput. 20, 21, 9488–9499 (2024)`_.

    Args:

        sim (SemiclassicalSimulation): Simulation object.

        num_samples (int): Number of stochastic hyperfine field realisations.

        init_state (ArrayLike): Initial population vector/state.

        radical_pair (list): Slice indices `[i0, i1]` defining the radical-pair subspace
            into which the Liouvillian is inserted.

        ts (np.ndarray): Time grid (s).

        Bs (ArrayLike): Magnetic field sweep values (mT).

        D (float): Dipolar coupling constant (mT).

        J (float): Exchange coupling constant (mT).

        kinetics (ArrayLike): Kinetic matrix to be added to the Liouvillian.

        relaxations (list[ArrayLike]): List of relaxation superoperators.

    Returns:
        dict:

        - ts: time grid
        - Bs: field sweep
        - yield: complex yield tensor with shape `(len(ts), len(init_state), len(Bs))`

    .. _Antill and Vatai, J. Chem. Theory Comput. 20, 21, 9488–9499 (2024):
       https://doi.org/10.1021/acs.jctc.4c00887
    """
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
    """Reaction yield-detected magnetic resonance (RYDMR) vs static field.

    Args:

        sim (HilbertSimulation): Simulation object.

        init_state (State): Initial `State` of the density matrix.

        obs_state (State): Observable `State` of the density matrix.

        time (np.ndarray): Sequence of (uniform) time points (s).

        D (float): Dipolar coupling constant (mT).

        J (float): Exchange coupling constant (mT).

        B0 (np.ndarray): Static field sweep (mT) along `B0_axis`.

        B1 (float): RF/AC amplitude (mT) along `B1_axis`.

        B1_freq (float): RF angular frequency offset (mT).

        B0_axis (str): Axis for `B0` Zeeman term.

        B1_axis (str): Axis for `B1` Zeeman term.

        kinetics (list): List of kinetic superoperators.

        relaxations (list): List of relaxation superoperators.

        hfc_anisotropy (bool): Include anisotropic hyperfine Hamiltonian if True.

    Returns:
        dict:

        - time: original `time`
        - B0: sweep values
        - B0_axis: axis of `B0`
        - B1: RF amplitude
        - B1_axis: axis of `B1`
        - B1_freq: RF angular frequency offset
        - B1_freq_axis: axis used for the offset term
        - rhos: density matrices (squared to Hilbert shape)
        - time_evolutions: product probabilities vs time & field
        - product_yields: integrated product yields
        - product_yield_sums: scalar yield per field
        - MARY: normalised magnetoresponse
        - LFE: low field effect (%)
        - HFE: high field effect (%)
    """
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
    """Semiclassical MARY with stochastic averaging and explicit population channels.

    Source: `Maeda et al. Mol. Phys. 104, 1779–1788 (2006)`_.

    Args:

        sim (SemiclassicalSimulation): Simulation object.

        num_samples (int): Number of stochastic hyperfine field realisations.

        init_state (State): Initial projection operator state.

        ts (ArrayLike): Time grid (s).

        Bs (ArrayLike): Magnetic field sweep (mT).

        D (float): Dipolar coupling constant (mT).

        J (float): Exchange coupling constant (mT).

        triplet_excited_state_quenching_rate (float): Quenching rate (1/s) feeding RP.

        free_radical_escape_rate (float): Escape rate from free radical (1/s).

        kinetics (list[ArrayLike]): Kinetic Liouvillian/superoperators.

        relaxations (list[ArrayLike]): Relaxation superoperators.

        scale_factor (float): Averaging weights for decay construction.

    Returns:
        dict:

        - ts: time grid
        - Bs: field sweep
        - MARY: response matrix with shape `(len(ts), len(Bs))`

    .. _Maeda et al. Mol. Phys. 104, 1779–1788 (2006):
       https://doi.org/10.1080/14767050600588106
    """
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
    """Steady-state MARY via linear solve of Liouvillian with ZFS, exchange, and Zeeman terms.

    Args:

        sim (LiouvilleSimulation): Simulation object.

        obs (State): Observable `State` used to form the projection operator `Q`.

        Bs (np.ndarray): Magnetic field sweep (mT).

        D (float): Zero-field splitting parameter D (mT).

        E (float): Zero-field splitting parameter E (mT).

        J (float): Exchange coupling (mT).

        theta (float): Polar angle of the Zeeman field.

        phi (float): Azimuthal angle of the Zeeman field.

        kinetics (list): List of kinetic superoperators applied to the Liouvillian.

    Returns:
        np.ndarray:

        - rhos: Steady-state density vectors, one per field (shape `(len(Bs), N)`).
        - Phi_s: Projection `rhos @ Q.flatten()` giving steady-state signal per field.
    """
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
