"""Relaxation superoperators in Liouville space for radical-pair dynamics.

This module defines a family of incoherent (relaxational) processes as
**Liouville-space superoperators** that act on vectorised density matrices.
Each process contributes an additive term ``subH`` (a superoperator) that
can be combined with coherent Liouvillians to evolve spin systems subject
to stochastic environmental interactions.

The implementations follow standard models in spin chemistry and magnetic
resonance, including singlet–triplet dephasing, random local fields,
triplet–triplet relaxation/dephasing, T₁/T₂ Bloch-type relaxation, and
anisotropic g-tensor modulation under rotational diffusion.

Classes:
        - `LiouvilleRelaxationBase`: Common base for relaxation superoperators.
        - `DipolarModulation`: Dipolar modulation model (Kattnig et al., 2016).
        - `GTensorAnisotropy`: g-tensor anisotropy relaxation (Kivelson, 1960).
        - `RandomFields`: Isotropic random-field model (Kattnig et al., 2016).
        - `SingletTripletDephasing`: S–T dephasing channel (Shushin, 1991).
        - `T1Relaxation`: Longitudinal (spin–lattice) relaxation (Bloch, 1946).
        - `T2Relaxation`: Transverse (spin–spin) relaxation (Bloch, 1946).
        - `TripletTripletDephasing`: Triplet–triplet dephasing (Gorelik et al., 2001).
        - `TripletTripletRelaxation`: Triplet–triplet population relaxation (Miura et al., 2015).

Helper functions:
        - `_g_tensor_anisotropy_term`: Per-radical superoperator contribution for
          g-anisotropy with Kivelson spectral densities.

Key concepts:
        - **Liouville space / superoperators**: Operators acting on vectorized
          density matrices. Additive relaxation terms are represented by `subH`.
        - **Projection operators**: `Q_S`, `Q_T`, `Q_{T+}`, `Q_{T0}`, `Q_{T-}`
          select singlet/triplet subspaces and enter several models.
        - **Spin operators**: Single-spin `S_x, S_y, S_z` per radical index, used
          to build isotropic/anisotropic relaxation channels via Kronecker products.
        - **Spectral density**: Frequency- and correlation-time–dependent scaling
          (e.g., Kivelson form) for stochastic modulation processes.

Usage pattern:
        1) Instantiate a relaxation class with its parameters
           (e.g., `rate_constant`, `tau_c`, `g` principal values, `omega`).
        2) Call `.init(sim)` with a `LiouvilleSimulation` providing
           `projection_operator(...)` and `spin_operator(radical_idx, axis)`.
        3) Retrieve the superoperator via the instance’s `subH` attribute and
           add it to your total Liouvillian.

Args conventions (per-class):
        - `rate_constant` (float, s⁻¹): Overall relaxation rate used in most models.
        - `g1`, `g2` (list[float]): Principal components of the radicals’ g-tensors.
        - `omega1`, `omega2` (float, rad·s⁻¹·mT⁻¹ × B): Larmor prefactors; the actual
          frequency is typically `omega * B0`. See note below.
        - `tau_c1`, `tau_c2` (float, s): Rotational correlation times.

Notes:
        - **Magnetic-field dependence**: For `GTensorAnisotropy`, the instantaneous
          Larmor frequency is proportional to the applied field `B0`. If `B0` varies
          (e.g., in MARY scans), compute `omega` dynamically from `B0` rather than
          treating it as a fixed constant.
        - **Vectorisation size**: Superoperators are built with Kronecker products
          sized to the underlying Hilbert spaces of the radicals in `sim`.
        - **Normalisation and units**: Rate-like parameters are in s⁻¹ unless noted.
          Axis labels/units are not handled here (these are dynamical, not plotting
          utilities).

References:
        - [Bloch, *Phys. Rev.* **70**, 460–474 (1946)](https://doi.org/10.1103/PhysRev.70.460).
        - [Gorelik et al., *J. Phys. Chem. A* **105**, 8011–8017 (2001)](https://doi.org/10.1021/jp0109628).
        - [Kattnig et al., *New J. Phys.* **18**, 063007 (2016)](https://iopscience.iop.org/article/10.1088/1367-2630/18/6/063007).
        - [Kivelson, *J. Chem. Phys.* **33**, 1094 (1960)](https://doi.org/10.1063/1.1731340).
        - [Miura et al., *J. Phys. Chem. A* **119**, 5534–5544 (2015)](https://doi.org/10.1021/acs.jpca.5b02183).
        - [Shushin, *Chem. Phys. Lett.* **181** (2–3), 274–278 (1991)](https://doi.org/10.1016/0009-2614(91)90366-H).

Requirements:
        - `numpy` for array/Kronecker algebra.
        - A simulation object exposing:
          `projection_operator(State.*)` and `spin_operator(idx, 'x'|'y'|'z')`,
          as used in the `init` methods.

Raises:
        ValueError: May be raised upstream if the simulation object does not
            support the required operators or radical indexing.

See also:
        `radicalpy.simulation.LiouvilleSimulation`,
        `radicalpy.states.State`,
        and the corresponding coherent Liouvillian construction in your codebase.
"""
from typing import Callable, Iterable, List, Tuple

import numpy as np

from .simulation import LiouvilleIncoherentProcessBase, LiouvilleSimulation, State
from .utils import spectral_density

# Type alias for BR channels:
# Each channel is (A, S) where
#   A: (dim x dim) Hermitian coupling operator (same basis/units as H)
#   S: callable spectral density S(omega: float) -> float  (rate-like)
BRChannel = Tuple[np.ndarray, Callable[[float], float]]


class LiouvilleRelaxationBase(LiouvilleIncoherentProcessBase):
    """Base class for relaxation superoperators (Liouville space)."""

    def _name(self):
        """First line of `__repr__()`."""
        name = super()._name()
        return f"Relaxation: {name}"


class DipolarModulation(LiouvilleRelaxationBase):
    """Dipolar modulation relaxation superoperator.

    Source: `Kattnig et al. New J. Phys., 18, 063007 (2016)`_.

    >>> DipolarModulation(rate_constant=1e6)
    Relaxation: DipolarModulation
    Rate constant: 1000000.0
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QTp = sim.projection_operator(State.TRIPLET_PLUS)
        QTm = sim.projection_operator(State.TRIPLET_MINUS)
        QT0 = sim.projection_operator(State.TRIPLET_ZERO)
        QS = sim.projection_operator(State.SINGLET)
        self.subH = self.rate * (
            1 / 9 * np.kron(QS, QTp)
            + 1 / 9 * np.kron(QTp, QS)
            + 1 / 9 * np.kron(QS, QTm)
            + 1 / 9 * np.kron(QTm, QS)
            + 4 / 9 * np.kron(QS, QT0)
            + 4 / 9 * np.kron(QT0, QS)
            + np.kron(QTp, QT0)
            + np.kron(QT0, QTp)
            + np.kron(QTm, QT0)
            + np.kron(QT0, QTm)
        )


# !!!!!!!!!! omega depends on B, which changes in every step (MARY loop)
# See note below
# Instead of omega1 & omega2 use B and calculate omegas inside
class GTensorAnisotropy(LiouvilleRelaxationBase):
    """g-tensor anisotropy relaxation superoperator.

    Source: `Kivelson, J. Chem. Phys. 33, 1094 (1960)`_.

    Args:
        g1 (list): The principle components of g-tensor of the first radical.
        g2 (list): The principle components of g-tensor of the second radical.
        omega1 (float): The Larmor frequency of the first radical (rad/s/mT).
        omega2 (float): The Larmor frequency of the second radical (rad/s/mT).
        tau_c1 (float): The rotational correlation time of the first radical (s).
        tau_c2 (float): The rotational correlation time of the second radical (s).

    >>> GTensorAnisotropy(g1=[2.0032, 1.9975, 2.0014],
    ...                   g2=[2.00429, 2.00389, 2.00216],
    ...                   omega1=-158477366720.7,
    ...                   omega2=-158477366720.7,
    ...                   tau_c1=5e-12,
    ...                   tau_c2=100e-12)
    Relaxation: GTensorAnisotropy
    g1: [2.0032, 1.9975, 2.0014]
    g2: [2.00429, 2.00389, 2.00216]
    omega1: -158477366720.7
    omega2: -158477366720.7
    tau_c1: 5e-12
    tau_c2: 1e-10

    .. _Kivelson, J. Chem. Phys. 33, 1094 (1960):
       https://doi.org/10.1063/1.1731340
    """

    def __init__(
        self,
        g1: list,
        g2: list,
        omega1: float,
        omega2: float,
        tau_c1: float,
        tau_c2: float,
    ):
        self.g1 = g1
        self.g2 = g2
        self.omega1 = omega1
        self.omega2 = omega2
        self.tau_c1 = tau_c1
        self.tau_c2 = tau_c2

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        self.subH = _g_tensor_anisotropy_term(sim, 0, self.g1, self.omega1, self.tau_c1)
        self.subH += _g_tensor_anisotropy_term(
            sim, 1, self.g2, self.omega2, self.tau_c2
        )

    def __repr__(self):
        lines = [
            self._name(),
            f"g1: {self.g1}",
            f"g2: {self.g2}",
            f"omega1: {self.omega1}",
            f"omega2: {self.omega2}",
            f"tau_c1: {self.tau_c1}",
            f"tau_c2: {self.tau_c2}",
        ]
        return "\n".join(lines)


class RandomFields(LiouvilleRelaxationBase):
    """Random fields relaxation superoperator.

    Source: `Kattnig et al. New J. Phys., 18, 063007 (2016)`_.

    >>> RandomFields(rate_constant=1e6)
    Relaxation: RandomFields
    Rate constant: 1000000.0
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QS = sim.projection_operator(State.SINGLET)
        idxs = range(len(sim.radicals))
        self.SABxyz = [sim.spin_operator(e, a) for e in idxs for a in "xyz"]

        term0 = np.kron(np.eye(len(QS)), np.eye(len(QS)))
        term1 = sum([np.kron(S, S.T) for S in self.SABxyz])
        self.subH = self.rate * (1.5 * term0 - term1)


class SingletTripletDephasing(LiouvilleRelaxationBase):
    """Singlet-triplet dephasing relaxation superoperator.

    Source: `Shushin, Chem. Phys. Lett. 181, 2,3, 274-278 (1991)`_.

    >>> SingletTripletDephasing(rate_constant=1e6)
    Relaxation: SingletTripletDephasing
    Rate constant: 1000000.0

    .. _Shushin, Chem. Phys. Lett. 181, 2,3, 274-278 (1991):
       https://doi.org/10.1016/0009-2614(91)90366-H
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QS = sim.projection_operator(State.SINGLET)
        QT = sim.projection_operator(State.TRIPLET)
        self.subH = self.rate * (np.kron(QS, QT) + np.kron(QT, QS))


class T1Relaxation(LiouvilleRelaxationBase):
    """T1 (spin-lattice, longitudinal, thermal) relaxation superoperator.

    Source: `Bloch, Phys. Rev. 70, 460-474 (1946)`_.

    >>> T1Relaxation(rate_constant=1e6)
    Relaxation: T1Relaxation
    Rate constant: 1000000.0

    .. _Bloch, Phys. Rev. 70, 460-474 (1946):
       https://doi.org/10.1103/PhysRev.70.460
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        SAz = sim.spin_operator(0, "z")
        SBz = sim.spin_operator(1, "z")

        self.subH = self.rate * (
            np.eye(len(SAz) * len(SAz)) - np.kron(SAz, SAz.T) - np.kron(SBz, SBz.T)
        )


class T2Relaxation(LiouvilleRelaxationBase):
    """T2 (spin-spin, transverse) relaxation superoperator.

    Source: `Bloch, Phys. Rev. 70, 460-474 (1946)`_.

    >>> T2Relaxation(rate_constant=1e6)
    Relaxation: T2Relaxation
    Rate constant: 1000000.0

    .. _Bloch, Phys. Rev. 70, 460-474 (1946):
       https://doi.org/10.1103/PhysRev.70.460
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        SAx, SAy = sim.spin_operator(0, "x"), sim.spin_operator(0, "y")
        SBx, SBy = sim.spin_operator(1, "x"), sim.spin_operator(1, "y")

        self.subH = self.rate * (
            np.eye(len(SAx) * len(SAx))
            - np.kron(SAx, SAx.T)
            - np.kron(SBx, SBx.T)
            - np.kron(SAy, SAy.T)
            - np.kron(SBy, SBy.T)
        )


class TripletTripletDephasing(LiouvilleRelaxationBase):
    """Triplet-triplet dephasing relaxation superoperator.

    Source: `Gorelik et al. J. Phys. Chem. A 105, 8011-8017 (2001)`_.

    >>> TripletTripletDephasing(rate_constant=1e6)
    Relaxation: TripletTripletDephasing
    Rate constant: 1000000.0

    .. _Gorelik et al. J. Phys. Chem. A 105, 8011-8017 (2001):
       https://doi.org/10.1021/jp0109628
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QTp = sim.projection_operator(State.TRIPLET_PLUS)
        QTm = sim.projection_operator(State.TRIPLET_MINUS)
        QT0 = sim.projection_operator(State.TRIPLET_ZERO)
        self.subH = self.rate * (
            np.kron(QTp, QTm)
            + np.kron(QTm, QTp)
            + np.kron(QT0, QTm)
            + np.kron(QTm, QT0)
            + np.kron(QTp, QT0)
            + np.kron(QT0, QTp)
        )


class TripletTripletRelaxation(LiouvilleRelaxationBase):
    """Triplet-triplet relaxation superoperator.

    Source: `Miura et al. J. Phys. Chem. A 119, 5534-5544 (2015)`_.

    >>> TripletTripletRelaxation(rate_constant=1e6)
    Relaxation: TripletTripletRelaxation
    Rate constant: 1000000.0

    .. _Miura et al. J. Phys. Chem. A 119, 5534-5544 (2015):
       https://doi.org/10.1021/acs.jpca.5b02183
    """

    # restrict to
    # init_state=rpsim.State.TRIPLET_ZERO,
    # obs_state=rpsim.State.TRIPLET_ZERO,
    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        QTp = sim.projection_operator(State.TRIPLET_PLUS)
        QTm = sim.projection_operator(State.TRIPLET_MINUS)
        QT0 = sim.projection_operator(State.TRIPLET_ZERO)
        super().init(sim)
        term0 = np.kron(QT0, QT0)
        term1 = np.kron(QTp, QTp) + np.kron(QTm, QTm)
        term2 = (
            np.kron(QTp, QT0)
            + np.kron(QT0, QTp)
            + np.kron(QTm, QT0)
            + np.kron(QT0, QTm)
        )
        self.subH = self.rate * (2 / 3 * term0 + 1 / 3 * (term1 - term2))


def _g_tensor_anisotropy_term(
    sim: LiouvilleSimulation, idx: int, g: list, omega: float, tau_c: float
) -> np.ndarray:
    giso = np.mean(g)
    SAx, SAy, SAz = [sim.spin_operator(idx, ax) for ax in "xyz"]
    H = 0.5 * np.eye(len(SAx) * len(SAx), dtype=complex)
    H -= np.kron(SAx, SAx.T)
    H -= np.kron(SAy, SAy.T)
    H *= 3 * spectral_density(omega, tau_c)
    H += (
        2
        * spectral_density(0, tau_c)
        * (0.5 * np.eye(len(SAx) * len(SAx)) - 2 * np.kron(SAz, SAz.T))
    )
    H *= 1 / 15 * sum([((gj - giso) / giso) ** 2 for gj in g]) * omega**2
    return H


def bloch_redfield_tensor(
    H: np.ndarray,
    a_ops: Iterable[BRChannel],
    *,
    secular: bool = True,
    secular_cutoff: float = 0.01,
) -> np.ndarray:
    """
    Construct the Bloch–Redfield superoperator R in the lab basis.

    Args:
        H: Hermitian Hamiltonian (dim x dim), in rad/s.
        a_ops: iterable of (A, S) with A the coupling operator (dim x dim)
               and S(ω) the scalar spectral density / noise power (>= 0).
        secular: if True, apply a simple secular selection rule.
        secular_cutoff: relative cutoff multiplier for secular test.

    Returns:
        R_lab: (dim^2 x dim^2) complex Redfield superoperator acting on vec(rho)
               (column-stacking convention).
    """
    dim = H.shape[0]
    evals, V = np.linalg.eigh(H)  # energy basis
    a_ops_E = []
    for A, S in a_ops:
        A_E = V.conj().T @ A @ V
        a_ops_E.append((A_E, S))

    # (a,b) index pairs and Bohr frequencies ω_ab
    idx_pairs = [(a, b) for a in range(dim) for b in range(dim)]
    bohr = np.array([evals[a] - evals[b] for a, b in idx_pairs])

    # Global magnitude for secular test (very conservative)
    if secular:
        grid = np.unique(np.abs(bohr))
        grid_rates = []
        for _, S in a_ops_E:
            if grid.size:
                # include ±ω for robustness
                rates = [float(S(+w)) + float(S(-w)) for w in grid]
                grid_rates.append(max(rates) if rates else 0.0)
        gmax = max(grid_rates) if grid_rates else 0.0

    # Allocate Redfield generator in energy basis
    R = np.zeros((dim * dim, dim * dim), dtype=complex)

    # Unitary part: -i [H, ·] → -i (E_a - E_b) δ_ac δ_bd on Liouville indices
    for j, (a, b) in enumerate(idx_pairs):
        R[j, j] += -1j * (evals[a] - evals[b])

    # Dissipative part
    for j, (a, b) in enumerate(idx_pairs):
        for k, (c, d) in enumerate(idx_pairs):
            Δ = (evals[a] - evals[b]) - (evals[c] - evals[d])
            if secular and gmax > 0.0 and abs(Δ) > gmax * secular_cutoff:
                continue
            total = 0.0 + 0.0j
            for A, S in a_ops_E:
                term = 0.0 + 0.0j
                if b == d:
                    for n in range(dim):
                        term += A[a, n] * A[n, c] * S(evals[c] - evals[n])
                term -= A[a, c] * A[d, b] * S(evals[c] - evals[a])
                if a == c:
                    for n in range(dim):
                        term += A[d, n] * A[n, b] * S(evals[d] - evals[n])
                term -= A[a, c] * A[d, b] * S(evals[d] - evals[b])
                total += (-0.5) * term
            R[j, k] += total

    # Transform back to lab basis:
    # vec(ρ') = (V^T ⊗ V^†) vec(ρ)
    T = np.kron(V.T, V.conj())
    Tinv = np.linalg.inv(T)
    R_lab = Tinv @ R @ T
    return R_lab


class BlochRedfield(LiouvilleRelaxationBase):
    """
    Bloch–Redfield relaxation superoperator (Liouville space).

    Parameters
    ----------
    channels : list[tuple[np.ndarray, callable]]
        List of (A, S) pairs: coupling operator and spectral density.
    secular : bool, default True
        Apply simple secular selection based on a global cutoff.
    secular_cutoff : float, default 0.01
        Relative threshold for the secular filter.

    Notes
    -----
    - Call `rebuild(H)` whenever the coherent Hilbert-space Hamiltonian changes.
    - Exposes `self.subH` (Liouville superoperator) for the framework to add.
    """

    expects_hilbert = True  # opt-in flag used by experiments to pass Hilbert H

    def __init__(
        self,
        channels: List[BRChannel],
        *,
        secular: bool = True,
        secular_cutoff: float = 0.01,
    ):
        super().__init__(rate_constant=0.0)  # placeholder; not used directly
        self.channels = channels
        self.secular = secular
        self.secular_cutoff = secular_cutoff
        self.subH = None

    def init(self, sim):
        """Optional cache of the simulation reference."""
        self._sim = sim

    def rebuild(self, H: np.ndarray):
        """Recompute `subH` for the given Hilbert-space H (rad/s)."""
        self.subH = bloch_redfield_tensor(
            H, self.channels, secular=self.secular, secular_cutoff=self.secular_cutoff
        )
        return self.subH
