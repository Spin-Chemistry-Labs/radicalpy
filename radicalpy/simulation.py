#!/usr/bin/env python
"""
Simulation utilities for spin dynamics.

This module provides classes and helpers to build quantum mechanical models in
either **Hilbert** space (density matrices as square arrays) or **Liouville**
space (vectorised densities / superoperators). It focuses on electron–nuclear
spin systems typical of radical pairs and triplet pairs and supports common
interactions and observables used in magnetic resonance and spin chemistry.

Main classes
------------
- `State` :
    Enumerates common initial/observable spin states (singlet/triplet manifold,
    EPR observable, thermal equilibrium, etc.).
- `Basis` :
    Choice of electron-pair basis: Zeeman or singlet–triplet (S/T).
- `HilbertSimulation` :
    Core simulator that assembles Hamiltonians (Zeeman, hyperfine, exchange,
    dipolar, optional zero-field splitting), prepares initial density matrices,
    propagates them unitarily, and evaluates product probabilities/yields.
- `LiouvilleSimulation` :
    Extends `HilbertSimulation` with Liouville-space evolution (superoperators,
    vectorised densities) for convenient inclusion of incoherent processes.
- `HilbertIncoherentProcessBase` / `LiouvilleIncoherentProcessBase` :
    Base hooks to augment Hamiltonians or measured probabilities with
    phenomenological kinetics / relaxation.
- `SparseCholeskyHilbertSimulation` :
    Hilbert-space variant optimised for large systems via sparse algebra and a
    Cholesky-factor time-stepping scheme.
- `SemiclassicalSimulation` :
    Generates random semiclassical Hamiltonians for ensemble treatments.

Key features
------------
- Spin operators for arbitrary particles and bases.
- Zeeman (1D/3D), hyperfine (isotropic/tensor), exchange (J), dipolar (1D/3D),
  and ZFS Hamiltonians.
- Initial states: projection-based (S, T, etc.) and thermal equilibrium.
- Time evolution: Hilbert (ρ → U ρ U†) and Liouville (ρ → e^{Lt} ρ).
- Observables: projection operators and product probabilities/yields.
- Shape conventions:
    - Hilbert density: `(dim, dim)`
    - Liouville density (vectorised): `(dim**2, 1)`

Units & conventions
-------------------
- Magnetic fields in mT; gyromagnetic ratios provided as `gamma_mT`.
- Tensors follow x/y/z Cartesian ordering.
- S/T transform from the Zeeman basis.

See also
--------
- `utils.spherical_to_cartesian` for field orientation.
- Module docstrings of related packages/classes for data structures (`Molecule`,
  nuclei, radicals, hyperfine data).

"""

import enum
from math import prod
from typing import Optional

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from . import utils
from .data import Molecule
from .shared import constants as C


class State(enum.Enum):
    """Enumeration of common spin/population states used as observables or initial conditions."""

    EQUILIBRIUM = "Eq"
    EPR = "EPR"
    SINGLET = "S"
    TRIPLET = "T"
    TRIPLET_ZERO = "T_0"
    TRIPLET_PLUS = "T_+"
    TRIPLET_PLUS_MINUS = "T_\\pm"
    TRIPLET_MINUS = "T_-"
    TP_SINGLET = "TP_S"
    TP_TRIPLET = "TP_T"
    TP_TRIPLET_ZERO = "TP_T0"
    TP_TRIPLET_PLUS = "TP_T+"
    TP_TRIPLET_MINUS = "TP_T-"
    TP_QUINTET = "TP_Q"
    TP_QUINTET_ZERO = "TP_Q0"
    TP_QUINTET_PLUS_TWO = "TP_Q+2"
    TP_QUINTET_PLUS_ONE = "TP_Q+1"
    TP_QUINTET_MINUS_TWO = "TP_Q-2"
    TP_QUINTET_MINUS_ONE = "TP_Q-1"


class Basis(enum.Enum):
    """Spin basis choices for electron pair subspace (Zeeman vs. singlet–triplet)."""

    ZEEMAN = "Zeeman"
    ST = "ST"


class HilbertIncoherentProcessBase:
    def __init__(self, rate_constant: float):
        """Base holder for an incoherent (phenomenological) process.

        Args:
            rate_constant: Process rate constant (s⁻¹) used by subclasses to
                modify the Hamiltonian and/or product probabilities.
        """
        self.rate = rate_constant

    def init(self, sim):
        """Hook to (optionally) initialise process state with the simulation.

        Subclasses may cache shapes, indices, or precomputed operators derived
        from ``sim``. The default implementation does nothing.
        """
        pass

    def adjust_hamiltonian(self, *args, **kwargs):
        """Optionally modify the Hamiltonian in-place.

        Subclasses can implement e.g. Redfield/Lindblad additions. The default
        implementation is a no-op and returns ``None``.
        """
        return

    def adjust_product_probabilities(self, *args, **kwargs):
        """Optionally post-process simulated product probabilities.

        Can be used to fold in kinetic schemes in Hilbert simulations. The
        default implementation is a no-op and returns ``None``.
        """
        return

    @property
    def rate_constant(self) -> float:
        """Rate of the incoherent process."""
        return self.rate

    def _name(self) -> str:
        """Return a short human-readable name for the process (class name)."""
        return str(type(self).__name__)

    def __repr__(self) -> str:
        """Pretty, multi-line summary with the process name and rate."""
        lines = [
            self._name(),
            f"Rate constant: {self.rate}",
        ]
        return "\n".join(lines)


class HilbertSimulation:
    """Quantum simulation base class.

    Args:
        molecules (list[Molecule]): List of `Molecule` objects.

        custom_gfactor (bool): Flag to use g-factors instead of the
            default gyromagnetic ratio gamma.

    >>> HilbertSimulation([Molecule.fromdb("flavin_anion", ["N5"]),
    ...                    Molecule.fromdb("tryptophan_cation", ["Hbeta1", "H1"])])
    Number of electrons: 2
    Number of nuclei: 3
    Number of particles: 5
    Multiplicities: [2, 2, 3, 2, 2]
    Magnetogyric ratios (mT): [-176085963.023, -176085963.023, 19337.792, 267522.18744, 267522.18744]
    Nuclei: [14N(19337792.0, 3, 0.5141 <anisotropic available>), 1H(267522187.44, 2, 1.605 <anisotropic available>), 1H(267522187.44, 2, -0.5983 <anisotropic available>)]
    Couplings: [0, 1, 1]
    HFCs (mT): [0.5141 <anisotropic available>, 1.605 <anisotropic available>, -0.5983 <anisotropic available>]
    """

    def __init__(
        self,
        molecules: list[Molecule],
        custom_gfactors: bool = False,
        basis: Basis = Basis.ST,
    ):
        """Construct a Hilbert-space spin simulation.

        Args:
            molecules: List of radical‐containing molecules; each must provide a
                single electron (``.radical``) and zero or more nuclei (``.nuclei``).
            custom_gfactors: If ``True``, use per-particle g-factors rather than
                default magnetogyric ratios.
            basis: Electron subspace basis (``Basis.ST`` or ``Basis.ZEEMAN``).
        """
        self.molecules = molecules
        self.custom_gfactors = custom_gfactors
        self.basis = basis

    @property
    def coupling(self):
        """List mapping each nucleus to the index of its parent radical."""
        return sum([[i] * len(m.nuclei) for i, m in enumerate(self.molecules)], [])

    @property
    def radicals(self):
        """List of electron spins (one per molecule)."""
        return [m.radical for m in self.molecules]

    @property
    def nuclei(self):
        """Flattened list of all nuclei across molecules."""
        return sum([[n for n in m.nuclei] for m in self.molecules], [])

    @property
    def particles(self):
        """Concatenated list of electron spins followed by nuclei."""
        return self.radicals + self.nuclei

    @property
    def hamiltonian_size(self):
        """Dimension of the Hilbert space (product of particle multiplicities)."""
        return np.prod([p.multiplicity for p in self.particles])

    def __repr__(self) -> str:
        """Human-readable simulation summary (counts, multiplicities, HFCs, etc.)."""
        return "\n".join(
            [
                # "Simulation summary:",
                f"Number of electrons: {len(self.radicals)}",
                f"Number of nuclei: {len(self.nuclei)}",
                f"Number of particles: {len(self.particles)}",
                f"Multiplicities: {[p.multiplicity for p in self.particles]}",
                f"Magnetogyric ratios (mT): {[p.gamma_mT for p in self.particles]}",
                f"Nuclei: {sum([m.nuclei for m in self.molecules], [])}",
                f"Couplings: {self.coupling}",
                f"HFCs (mT): {[n.hfc for n in self.nuclei]}",
                # "",
                # f"Simulated molecules:\n{molecules}",
            ]
        )

    def ST_basis(self, M, kron_eye: bool = True):
        """Transform an operator from the Zeeman basis to the S/T basis.

        Args:
            M: Operator in the Zeeman basis.
            kron_eye: Whether to Kronecker-product the identity matrix is inserted.

        Returns:
            The operator expressed in the S/T basis.
        """
        # T+  T0  S  T-
        ST = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )
        if kron_eye:
            C = np.kron(ST, np.eye(prod([n.multiplicity for n in self.nuclei])))
        else:
            C = ST
        return C @ M @ C.T

    def spin_operator(self, idx: int, axis: str, kron_eye: bool = True) -> np.ndarray:
        """Construct the spin operator.

        Construct the spin operator for the particle with index `idx`
        in the `HilbertSimulation`.

        Args:

            idx (int): Index of the particle.

            axis (str): Axis, i.e. ``"x"``, ``"y"`` or ``"z"``.

            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray:

                Spin operator for a particle in the
                `HilbertSimulation` system with indexing `idx` and
                axis `axis`.
        """
        assert 0 <= idx and idx < len(self.particles)
        assert axis in "xyzpmu"

        sigma = self.particles[idx].pauli[axis]
        if kron_eye:
            before_size = prod(p.multiplicity for p in self.particles[:idx])
            after_size = prod(p.multiplicity for p in self.particles[idx + 1 :])
            spinop = np.kron(np.eye(before_size), sigma)
            spinop = np.kron(spinop, np.eye(after_size))
        else:
            # Only electronic states are kron-producted with the identity matrix
            if idx == 0:
                spinop = np.kron(sigma, np.eye(2))
            elif idx == 1:
                spinop = np.kron(np.eye(2), sigma)
            else:
                spinop = sigma
        if self.basis == Basis.ST:
            return self.ST_basis(spinop, kron_eye=kron_eye)
        else:
            return spinop

    def product_operator(
        self, idx1: int, idx2: int, h: float = 1.0, kron_eye: bool = True
    ) -> np.ndarray:
        """Construct the (1D) product operator.

        Construct the 1D (isotropic) product operator of two particles
        in the spin system.

        Args:

            idx1 (int): Index of the first particle.

            idx2 (int): Index of the second particle.

            h (float): Isotropic interaction constant.

            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray:

                Product operator for particles corresponding to `idx1`
                and `idx2` with isotropic interaction constant `h`.
        """
        return h * sum(
            [
                self.spin_operator(idx1, axis, kron_eye=kron_eye).dot(
                    self.spin_operator(idx2, axis, kron_eye=kron_eye)
                )
                for axis in "xyz"
            ]
        )

    def product_operator_3d(
        self, idx1: int, idx2: int, h: np.ndarray, kron_eye: bool = True
    ) -> np.ndarray:
        """Construct the 3D product operator.

        Construct the 3D (anisotropic) product operator of two
        particles in the spin system.

        Args:

            idx1 (int): Index of the first particle.

            idx2 (int): Index of the second particle.

            h (np.ndarray): Anisotropic interaction tensor.

            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray:

                Product operator for particles corresponding to `idx1`
                and `idx2` with anisotropic interaction tensor `h`.
        """
        return sum(
            (
                h[i, j]
                * self.spin_operator(idx1, ax1, kron_eye=kron_eye).dot(
                    self.spin_operator(idx2, ax2, kron_eye=kron_eye)
                )
                for i, ax1 in enumerate("xyz")
                for j, ax2 in enumerate("xyz")
            )
        )

    def get_eye(self, shape: int) -> np.ndarray:
        """Return an identity matrix of the requested dimension (dense)."""
        return np.eye(shape)

    def projection_operator(self, state: State, T: float = 298, kron_eye: bool = True):
        """Construct the projection operator corresponding to a `state`.

        Args:

            state (State): The target state which is projected out of
                the density matrix.
            T (float): Temperature for the EQUILIBRIUM projection operator (K).

            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray:

                Projection operator corresponding to the `State`
                `state`.
        """
        # Spin operators
        SAx, SAy, SAz = [self.spin_operator(0, ax, kron_eye=kron_eye) for ax in "xyz"]
        SBx, SBy, SBz = [self.spin_operator(1, ax, kron_eye=kron_eye) for ax in "xyz"]

        # Product operators
        SASB = self.product_operator(0, 1, kron_eye=kron_eye)
        eye = self.get_eye(SASB.shape[0])

        result = {
            State.SINGLET: (1 / 4) * eye - SASB,
            State.TRIPLET: (3 / 4) * eye + SASB,
            State.TRIPLET_PLUS: (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz),
            State.TRIPLET_MINUS: (2 * SAz**2 - SAz) * (2 * SBz**2 - SBz),
            State.TRIPLET_ZERO: (1 / 4) * eye + SAx @ SBx + SAy @ SBy - SAz @ SBz,
            State.TRIPLET_PLUS_MINUS: (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz)
            + (2 * SAz**2 - SAz) * (2 * SBz**2 - SBz),
            State.EQUILIBRIUM: C.hbar / (C.k_B * T),
            State.TP_SINGLET: self.tp_singlet_projop(SAx, SAy, SAz, SBx, SBy, SBz),
            State.TP_TRIPLET: self.tp_triplet_projop(SAx, SAy, SAz, SBx, SBy, SBz),
            State.TP_TRIPLET_ZERO: self.tp_triplet_zero_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.TP_TRIPLET_PLUS: self.tp_triplet_plus_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.TP_TRIPLET_MINUS: self.tp_triplet_minus_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.TP_QUINTET: self.tp_quintet_projop(SAx, SAy, SAz, SBx, SBy, SBz),
            State.TP_QUINTET_ZERO: self.tp_quintet_zero_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.TP_QUINTET_PLUS_TWO: self.tp_quintet_plus_two_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.TP_QUINTET_PLUS_ONE: self.tp_quintet_plus_one_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.TP_QUINTET_MINUS_TWO: self.tp_quintet_minus_two_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.TP_QUINTET_MINUS_ONE: self.tp_quintet_minus_one_projop(
                SAx, SAy, SAz, SBx, SBy, SBz
            ),
            State.EPR: -(SAy + SBy),
        }
        return result[state]

    def tp_singlet_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair singlet (TP-S) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        return (1 / 12) * (Ssquared - (6 * E)) @ (Ssquared - (2 * E))

    def tp_triplet_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair triplet (TP-T) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        return (1 / 8) * (-Ssquared @ Ssquared + 6 * Ssquared)

    def tp_triplet_zero_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair triplet zero (TP-T0) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PT = (1 / 8) * (-Ssquared @ Ssquared + 6 * Ssquared)
        return PT @ (-Sz @ Sz + E)

    def tp_triplet_plus_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair triplet plus (TP-T+) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PT = (1 / 8) * (-Ssquared @ Ssquared + 6 * Ssquared)
        return PT @ (Sz @ (Sz + E) / 2)

    def tp_triplet_minus_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair triplet minus (TP-T-) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PT = (1 / 8) * (-Ssquared @ Ssquared + 6 * Ssquared)
        return PT @ (Sz @ (Sz - E) / 2)

    def tp_quintet_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair quintet (TP-Q) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        return (1 / 24) * Ssquared @ (Ssquared - 2 * E)

    def tp_quintet_zero_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair quintet zero (TP-Q0) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PQ = (1 / 24) * Ssquared @ (Ssquared - 2 * E)
        return PQ @ (Sz**2 - E) @ (Sz**2 - 4 * E) / 4

    def tp_quintet_plus_two_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair quintet plus two (TP-Q+2) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PQ = (1 / 24) * Ssquared @ (Ssquared - 2 * E)
        return PQ @ (Sz - E) @ Sz @ (Sz + E) @ (Sz + 2 * E) / 24

    def tp_quintet_plus_one_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair quintet plus one (TP-Q+1) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PQ = (1 / 24) * Ssquared @ (Ssquared - 2 * E)
        return PQ @ (Sz + 2 * E) @ Sz @ (Sz + E) @ (Sz - 2 * E) / -6

    def tp_quintet_minus_two_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair quintet minus two (TP-Q-2) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PQ = (1 / 24) * Ssquared @ (Ssquared - 2 * E)
        return PQ @ (Sz + E) @ Sz @ (Sz - E) @ (Sz - 2 * E) / 24

    def tp_quintet_minus_one_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        """Projection operator onto the triplet-pair quintet minus one (TP-Q-1) subspace.

        Builds the projector from electron spin operators of radicals A and B.
        """
        # For radical triplet pair (RTP)
        E = self.get_eye(SAx.shape[0])
        Sz = SAz @ E + E @ SBz
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)
        PQ = (1 / 24) * Ssquared @ (Ssquared - 2 * E)
        return PQ @ (Sz + 2 * E) @ Sz @ (Sz + E) @ (Sz - 2 * E) / -6

    def zeeman_hamiltonian(
        self,
        B0: float,
        B_axis: str = "z",
        theta: Optional[float] = None,
        phi: Optional[float] = None,
    ) -> np.ndarray:
        """Construct the Zeeman Hamiltonian (1D or 3D).

        Construct the Zeeman Hamiltonian based on the external
        magnetic field `B0`.  If the angles `theta` and `phi` are also
        provided, the 3D Zeeman Hamiltonian is constructed (by
        invoking the `zeeman_hamiltonian_3d`), otherwise the 1D Zeeman
        Hamiltonian is constructed (by invoking the
        `zeeman_hamiltonian_1d`).

        Args:

            B0 (float): External magnetic field intensity (milli
                Tesla). See `zeeman_hamiltonian_1d` and
                `zeeman_hamiltonian_3d`.

            axis (str): Axis of the magnetic field.

            theta (Optional[float]): rotation (polar) angle between
                the external magnetic field and the fixed
                molecule. See `zeeman_hamiltonian_3d`.

            phi (Optional[float]): rotation (azimuth) angle between
                the external magnetic field and the fixed molecule.
                See `zeeman_hamiltonian_3d`.


        Returns:
            np.ndarray:

                The Zeeman Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and the
                external magnetic field intensity `B0` and angles
                `theta` and `phi`.
        """
        if theta is None and phi is None:
            return self.zeeman_hamiltonian_1d(B0, B_axis)
        return self.zeeman_hamiltonian_3d(B0, theta, phi)

    def zeeman_hamiltonian_1d(self, B0: float, axis: str) -> np.ndarray:
        """Construct the 1D Zeeman Hamiltonian.

        Construct the 1D Zeeman Hamiltonian based on the external
        magnetic field `B0`.

        Args:

            B0 (float): External magnetic field intensity (milli
                Tesla).

            axis (str): Axis of the magnetic field.

        Returns:
            np.ndarray:

                The Zeeman Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and the
                external magnetic field intensity `B0`.
        """
        assert axis in "xyz", "`axis` can only be `x`, `y` or `z`"
        gammas = enumerate(p.gamma_mT for p in self.particles)
        return -B0 * sum(g * self.spin_operator(i, axis) for i, g in gammas)

    def zeeman_hamiltonian_3d(
        self, B0: float, theta: Optional[float] = 0, phi: Optional[float] = 0
    ) -> np.ndarray:
        """Construct the 3D Zeeman Hamiltonian.

        Construct the 3D Zeeman Hamiltonian based on the external
        magnetic field `B0` and angles `theta` and `phi`.

        Args:

            B0 (float): External magnetic field intensity (milli
                Tesla).

            theta (Optional[float]): rotation (polar) angle between
                the external magnetic field and the fixed
                molecule.

            phi (Optional[float]): rotation (azimuth) angle between
                the external magnetic field and the fixed molecule.

        Returns:
            np.ndarray:

                The Zeeman Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and the
                external magnetic field intensity `B0` and angles
                `theta` and `phi`.
        """
        particles = [
            [p.gamma_mT * self.spin_operator(idx, axis) for axis in "xyz"]
            for idx, p in enumerate(self.particles)
        ]
        rotation = utils.spherical_to_cartesian(theta, phi)
        B = -B0 * rotation
        return sum(
            gamma_Sr * Br
            for gamma_Sxyz in particles
            for gamma_Sr, Br in zip(gamma_Sxyz, B, strict=True)
        )

    def hyperfine_hamiltonian(self, hfc_anisotropy: bool = False) -> np.ndarray:
        """Construct the Hyperfine Hamiltonian.

        Construct the Hyperfine Hamiltonian.  If `hfc_anisotropy` is
        `False`, then the isotropic hyperfine coupling constants are
        used. If `hfc_anisotropy` is `True` then the full hyperfine
        tensors are used (assuming they are available for all the
        nuclei of the molecule in the database, otherwise an exception
        is raised).

        Args:

            hfc_anisotropy (bool): Use isotropic hyperfine coupling
                constants if `False`, use full hyperfine tensors if
                `False`.

        Returns:
            np.ndarray:

                The Hyperfine Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object.
        """
        if hfc_anisotropy:
            for h in [n.hfc for n in self.nuclei]:
                # TODO(vatai) try except not is None
                if h.anisotropic is None:
                    raise ValueError(
                        "Not all molecules have anisotropic HFCs! "
                        "Please use `hfc_anisotropy=False`"
                    )

            prodop = self.product_operator_3d
            hfcs = [n.hfc.anisotropic for n in self.nuclei]
        else:
            prodop = self.product_operator
            hfcs = [n.hfc.isotropic for n in self.nuclei]
        return sum(
            (
                abs(self.particles[ei].gamma_mT)
                * prodop(ei, len(self.radicals) + ni, hfcs[ni])
                for ni, ei in enumerate(self.coupling)
            )
        )

    def exchange_hamiltonian(
        self, J: float, prod_coeff: float = 2, kron_eye: bool = True
    ) -> np.ndarray:
        """Construct the exchange Hamiltonian.

        Construct the exchange (J-coupling) Hamiltonian based on the
        coupling constant J between two electrons.

        The J-coupling constant can be obtained using:

        - `radicalpy.estimations.exchange_interaction_in_protein`
        - `radicalpy.estimations.exchange_interaction_in_solution`

        Args:

            J (float): Exchange coupling constant.

            prod_coeff (float): Coefficient of the product operator
                (default, radical-pair convention uses 2.0,
                spintronics convention uses 1.0).

            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray:

                The exchange (J-coupling) Hamiltonian corresponding to
                the system described by the `HilbertSimulation` object
                and the coupling constant `J`.
        """
        Jcoupling = -J * abs(self.radicals[0].gamma_mT)
        SASB = self.product_operator(0, 1, kron_eye=kron_eye)
        E = self.get_eye(SASB.shape[0])
        return Jcoupling * (prod_coeff * SASB + 0.5 * E)

    def dipolar_hamiltonian(
        self, D: float | np.ndarray, kron_eye: bool = True
    ) -> np.ndarray:
        """Construct the Dipolar Hamiltonian.

        Construct the Dipolar Hamiltonian based on dipolar coupling
        constant or dipolar interaction tensor `D` between two
        electrons.  Depending on the `type` of `D`, the 1D or the 3D
        version is invoked.

        See:

        - `dipolar_hamiltonian_1d`
        - `dipolar_hamiltonian_3d`

        Args:

            D (float | np.ndarray): dipolar coupling constant or
                dipolar interaction tensor.

            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray:

                The Dipolar Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and
                dipolar coupling constant or dipolar interaction
                tensor `D`.
        """
        if isinstance(D, np.ndarray):
            return self.dipolar_hamiltonian_3d(D, kron_eye=kron_eye)
        else:
            return self.dipolar_hamiltonian_1d(D, kron_eye=kron_eye)

    def dipolar_hamiltonian_1d(self, D: float, kron_eye: bool = True) -> np.ndarray:
        """Construct the 1D Dipolar Hamiltonian.

        Construct the Dipolar Hamiltonian based on dipolar coupling
        constant `D` between two electrons.

        The dipolar coupling constant can be obtained using
        `radicalpy.estimations.dipolar_interaction_isotropic`.

        Args:

            D (float): dipolar coupling constant in mT.

        Returns:
            np.ndarray:

                The 1D Dipolar Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and
                dipolar coupling constant `D`.
        """
        SASB = self.product_operator(0, 1, kron_eye=kron_eye)
        SAz = self.spin_operator(0, "z", kron_eye=kron_eye)
        SBz = self.spin_operator(1, "z", kron_eye=kron_eye)
        if D > 0.0:
            print(
                f"WARNING: D is {D} mT, which is positive. In point dipole approximation, D should be negative."
            )
        omega = (2 / 3) * abs(self.radicals[0].gamma_mT) * D
        return omega * (3 * SAz @ SBz - SASB)

    def dipolar_hamiltonian_3d(
        self, dipolar_tensor: np.ndarray, kron_eye: bool = True
    ) -> np.ndarray:
        """Construct the 3D Dipolar Hamiltonian.

        Construct the Dipolar Hamiltonian based on dipolar interaction
        tensor `D` between two electrons.

        The dipolar coupling tensor can be obtained using
        `radicalpy.estimations.dipolar_interaction_anisotropic`.

        Args:

            dipolar_tensor (np.ndarray): dipolar interaction tensor in mT.

            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray:

                The 3D Dipolar Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and
                dipolar interaction tensor `D`.
        """
        spinops = [
            [self.spin_operator(r, ax, kron_eye=kron_eye) for ax in "xyz"]
            for r, _ in enumerate(self.radicals)
        ]
        return sum(
            (
                dipolar_tensor[i, j] * (si @ sj) * abs(self.radicals[0].gamma_mT)
                for i, si in enumerate(spinops[0])
                for j, sj in enumerate(spinops[1])
            )
        )

    def zero_field_splitting_hamiltonian(
        self, D, E, kron_eye: bool = True
    ) -> np.ndarray:
        """Build the zero-field splitting (ZFS) Hamiltonian.

        Constructs the second-rank ZFS contribution for (typically triplet, S=1)
        spins using the conventional axial/rhombic parameters ``D`` and ``E``.
        For each particle index ``idx``, the operator
        :math:`D (S_z^2 - \\tfrac{1}{3} \\mathbf{S}^2) + E (S_x^2 - S_y^2)`
        is formed and summed. The parameters are scaled into frequency units via
        the first radical’s electron gyromagnetic ratio.

        Args:
            D: Axial ZFS parameter (in the same field/frequency units used
            elsewhere; internally scaled by ``-radicals[0].gamma_mT``).
            E: Rhombic ZFS parameter (scaled identically to ``D``).
            kron_eye: Whether to Kronecker-product of the identity matrix is inserted.

        Returns:
            np.ndarray: The ZFS Hamiltonian matrix in the current simulation basis.

        Notes:
            - This implementation multiplies ``D`` and ``E`` by ``-gamma_mT`` of
            the first radical to convert to angular frequency units consistent
            with the rest of the Hamiltonian.
            - The loop applies the operator to every particle index; in practice,
            ZFS is only meaningful for S≥1 spins (e.g., an excited triplet).
            Ensure your particle list reflects that physical situation.
        """
        Dmod = D * abs(self.radicals[0].gamma_mT)
        Emod = E * abs(self.radicals[0].gamma_mT)
        result = complex(0.0)
        for idx, p in enumerate(self.particles):
            Sx = self.spin_operator(idx, "x", kron_eye=kron_eye)
            Sy = self.spin_operator(idx, "y", kron_eye=kron_eye)
            Sz = self.spin_operator(idx, "z", kron_eye=kron_eye)
            Ssquared = Sx @ Sx + Sy @ Sy + Sz @ Sz
            result += Dmod * (Sz @ Sz - (1 / 3) * Ssquared)
            result += Emod * ((Sx @ Sx) - (Sy @ Sy))
        return result
    
    def linblad_hamiltonian(self, H: np.ndarray, Ls=[]):
        """
        Assemble the Liouville-space generator for coherent + Lindblad dynamics.

        This builds the superoperator
        :math:`\\mathcal{L} = -i\\,[H,\\cdot] + \\sum_k \\mathcal{D}[L_k]`
        using the column–stacking (vec) convention. The coherent part is
        implemented as

        .. math::
            -i[H,\\rho] \\;\\mapsto\\; -i\\,(I\\otimes H - H^{\\mathsf{T}}\\otimes I),

        and each dissipator is

        .. math::
            \\mathcal{D}[L](\\rho)
            = L\\,\\rho\\,L^{\\dagger}
            - \\tfrac{1}{2}\\{L^{\\dagger}L,\\rho\\}
            \\;\\mapsto\\;
            L^{\\ast}\\otimes L
            - \\tfrac{1}{2}\\Big( I\\otimes L^{\\dagger}L
                            + (L^{\\mathsf{T}}L^{\\ast})\\otimes I \\Big).

        Parameters
        ----------
        H : ndarray of shape (N, N), complex or float
            Hilbert-space Hamiltonian. For Hermitian problems, a complex dtype
            (e.g. ``np.complex128``) is recommended.
        Ls : Sequence[ndarray], optional
            Iterable of collapse operators ``L_k`` (each ``N×N``) defining the
            Lindblad dissipators. If you have physical rates ``γ_k``, scale your
            operators as ``L_k ← √γ_k · C_k`` before passing them.

        Returns
        -------
        superop : ndarray of shape (N*N, N*N), complex
            The full Liouville-space generator :math:`\\mathcal{L}` suitable for
            acting on ``vec(ρ)`` with the column-stacking convention.

        Notes
        -----
        - The mapping uses ``vec(O ρ) = (I ⊗ O^{\\mathsf{T}}) vec(ρ)`` and
        ``vec(ρ O) = (O^{\\mathsf{T}} ⊗ I) vec(ρ)``.
        - ``H`` enters only through the coherent commutator; the dissipator depends
        solely on ``Ls``.
        - All arrays are combined with NumPy ``kron``; performance-critical paths
        may prefer sparse representations.
        """
        dim = len(H)
        Hsuper = -1j * (np.kron(np.eye(dim), H) - np.kron(H.T, np.eye(dim)))  # Hamiltonian
        Lsuper = sum(
            [
                np.kron(L.conjugate(), L)
                - 0.5
                * (
                    np.kron(np.eye(dim), L.conjugate().T.dot(L))
                    + np.kron(L.T.dot(L.conjugate()), np.eye(dim))
                )
                for L in Ls
            ]
        )  # Lindblad
        return Hsuper + Lsuper

    def total_hamiltonian(
        self,
        B0: float,
        J: float,
        D: float | np.ndarray,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        hfc_anisotropy: bool = False,
    ) -> np.ndarray:
        """Construct the total Hamiltonian.

        The total Hamiltonian is the sum of Zeeman, Hyperfine,
        Exchange and Dipolar Hamiltonian.

        Args:

            B0 (float): See `zeeman_hamiltonian`.

            J (float): See `exchange_hamiltonian`.

            D (float): See `dipolar_hamiltonian`.

            theta (Optional[float]): See `zeeman_hamiltonian`.

            phi (Optional[float]): See `zeeman_hamiltonian`.

            hfc_anisotropy (bool): See `hyperfine_hamiltonian`.

        Returns:
            np.ndarray:

                The total Hamiltonian.
        """
        H = (
            self.zeeman_hamiltonian(B0, theta=theta, phi=phi)
            + self.hyperfine_hamiltonian(hfc_anisotropy)
            + self.exchange_hamiltonian(J)
            + self.dipolar_hamiltonian(D)
        )
        return self.convert(H)

    def time_evolution(
        self, init_state: State, time: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Evolve the system through time.

        See Also:
        - `HilbertSimulation.unitary_propagator`
        - `HilbertSimulation.propagate`
        - `LiouvilleSimulation.unitary_propagator`
        - `LiouvilleSimulation.propagate`

        Args:

            init_state (State): Initial `State` of the density matrix
                (see `projection_operator`).

            time (np.ndarray): An sequence of (uniform) time points,
                usually created using `np.arange` or `np.linspace`.

            H (np.ndarray): Hamiltonian operator.

        Returns:
            np.ndarray:

                Return a sequence of density matrices evolved through
                `time`, starting from a given initial `state` using
                the Hamiltonian `H`.

        Examples:
            >>> molecules = [Molecule.fromdb("flavin_anion", ["N5"]),
            ...              Molecule("Z")]
            >>> sim = HilbertSimulation(molecules)
            >>> H = sim.total_hamiltonian(B0=0, J=0, D=0)
            >>> time = np.arange(0, 2e-6, 5e-9)
            >>> time.shape
            (400,)
            >>> rhos = sim.time_evolution(State.SINGLET, time, H)
            >>> rhos.shape
            (400, 12, 12)
        """
        dt = time[1] - time[0]
        propagator = self.unitary_propagator(H, dt)
        rho0 = self.initial_density_matrix(init_state, H)
        rhos = np.zeros([len(time), *rho0.shape], dtype=complex)
        rhos[0] = rho0
        for t in range(1, len(time)):
            rhos[t] = self.propagate(propagator, rhos[t - 1])
        return rhos

    def product_probability(self, obs: State, rhos: np.ndarray) -> np.ndarray:
        """Calculate the probability of the observable from the densities."""
        if obs == State.EQUILIBRIUM:
            raise ValueError("Observable state should not be EQUILIBRIUM")
        Q = self.observable_projection_operator(obs)
        if isinstance(rhos, list):
            if isinstance(rhos[0], tuple):
                # Cholesky factorization
                Qrhos = [(Lt @ Q @ L).trace().real for (L, Lt) in rhos]
            else:
                Qrhos = [(Q @ rho).trace().real for rho in rhos]
            return np.array(Qrhos)
        else:
            return np.real(np.trace(Q @ rhos, axis1=-2, axis2=-1))

    @staticmethod
    def product_yield(product_probability, time, k):
        """Calculate the product yield and the product yield sum."""
        product_yield = k * sp.integrate.cumulative_trapezoid(
            product_probability, time, initial=0
        )
        product_yield_sum = k * np.trapezoid(product_probability, dx=time[1])
        return product_yield, product_yield_sum

    def apply_liouville_hamiltonian_modifiers(self, H, modifiers):
        """Apply (Hilbert) incoherent process modifiers to the Liouville Hamiltonian.

        Each modifier is initialised with ``self`` once, then given the chance to
        adjust ``H`` in place. This is a no-op for Hilbert simulations.
        """
        for K in modifiers:  # skip in hilbert
            K.init(self)
            K.adjust_hamiltonian(H)

    @staticmethod
    def apply_hilbert_kinetics(time, product_probabilities, kinetics):
        """Apply kinetic post-processing to product probabilities in Hilbert space.

        Each object in ``kinetics`` may rescale or filter the probabilities as a
        function of ``time`` (e.g., recombination channels).
        """
        for K in kinetics:  # skip in liouville
            K.adjust_product_probabilities(product_probabilities, time)

    @staticmethod
    def _square_liouville_rhos(rhos):
        """Hilbert helper: identity mapping (Liouville uses an override)."""
        return rhos

    @staticmethod
    def _get_rho_shape(dim):
        """Return the array shape for a density matrix in this simulation space.

        In Hilbert-space simulations the density is a square matrix, so the shape
        is ``(dim, dim)``. (Liouville simulations override this to ``(dim**2, 1)``.)

        Args:
            dim (int): Dimension of the Hilbert space.

        Returns:
            tuple[int, int]: ``(dim, dim)``.
        """
        return dim, dim

    @staticmethod
    def convert(H: np.ndarray) -> np.ndarray:
        """Convert a Hilbert-space Hamiltonian to the simulator's working space.

        In Hilbert simulations this is the identity; Liouville overrides it.
        """
        return H

    @staticmethod
    def _convert(Q: np.ndarray) -> np.ndarray:
        """Convert a Hilbert-space projector/observable to the working space.

        Identity for Hilbert; Liouville overrides to build the superoperator.
        """
        return Q

    def initial_density_matrix(self, state: State, H: np.ndarray) -> np.ndarray:
        """Construct the initial density matrix for time evolution.

        Builds a properly normalised density operator in Hilbert space
        corresponding to the chosen initial `state`.

        - For most states (e.g. singlet, triplet), this is simply the
        normalised projection operator.
        - For the special case `State.EQUILIBRIUM`, the density is taken
        as the thermal equilibrium state, approximated via
        :math:`\\rho_0 \\propto e^{-i H \\beta}` where
        :math:`\\beta = \\hbar / (k_B T)` is provided via
        `projection_operator`.

        Args:
            state (State): Target spin state (e.g., `SINGLET`, `TRIPLET`,
                `EQUILIBRIUM`).
            H (np.ndarray): Spin Hamiltonian in Hilbert space.

        Returns:
            np.ndarray: Normalised density matrix in Hilbert space.
        """
        Pi = self.projection_operator(state)

        if state == State.EQUILIBRIUM:
            rho0eq = sp.sparse.linalg.expm(-1j * sp.sparse.csc_matrix(H) * Pi).toarray()
            rho0 = rho0eq / rho0eq.trace()
        else:
            rho0 = Pi / Pi.trace()
        return rho0

    @staticmethod
    def unitary_propagator(H: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        r"""Create unitary propagator (Hilbert space).

        Create unitary propagator matrices **U** and **U*** for time
        evolution of the density matrix in Hilbert space (for the spin
        Hamiltonian `H`).

        .. math::
            \mathbf{U}   =& \exp( -i \hat{H} t ) \\
            \mathbf{U}^* =& \exp( +i \hat{H} t )

        See also: `propagate` and `time_evolution`.

        Args:

            H (np.ndarray): Spin Hamiltonian in Hilbert space.

            dt (float): Time evolution timestep.

        Returns:
            np.ndarray:

                Two matrices (as tensor) in Hilbert space which are
                used by the `propagate` method to perform a single
                step in the `time_evolution` method.

        Examples:

            >>> molecules = [Molecule.fromdb("flavin_anion", ["N5"]),
            ...              Molecule("Z")]
            >>> sim = HilbertSimulation(molecules)
            >>> H = sim.total_hamiltonian(B0=0, J=0, D=0)
            >>> Up, Um = sim.unitary_propagator(H, 3e-9)
            >>> Up.shape, Um.shape
            ((12, 12), (12, 12))
        """
        Up = sp.sparse.linalg.expm(1j * H * dt)
        Um = sp.sparse.linalg.expm(-1j * H * dt)
        return Up, Um

    def propagate(self, propagator: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Propagate the density matrix (Hilbert space).

        Propagates the density matrix using the propagator obtained
        using the `unitary_propagator` method.

        .. math::
            \\rho (t) = \\mathbf{U} \\rho_0 \\mathbf{U}^*

        See also: `unitary_propagator` and `time_evolution`.

        Args:

            propagator (np.ndarray): Unitary operator obtained via the
                `unitary_propagator` method.

            rho (np.ndarray): (Initial) density matrix.

        Returns:
            np.ndarray:

                The new density matrix after the unitary operator was
                applied to it.
        """
        Up, Um = propagator
        return Um @ rho @ Up

    def observable_projection_operator(self, state: State) -> np.ndarray:
        """Projection operator used to compute observable expectation values.

        Simply forwards to :meth:`projection_operator` in Hilbert space.
        """
        return self.projection_operator(state)
    

class LiouvilleSimulation(HilbertSimulation):
    @staticmethod
    def convert(H: np.ndarray) -> np.ndarray:
        """Convert the Hamiltonian from Hilbert to Liouville space."""
        eye = np.eye(H.shape[0])
        tmp = np.kron(H, eye) - np.kron(eye, H.T)
        return 1j * tmp

    @staticmethod
    def _convert(Q: np.ndarray) -> np.ndarray:
        """Lift a Hilbert-space observable to its Liouville-space superoperator.

        Builds the symmetrised Kronecker sum ``Q ⊗ I + I ⊗ Q`` that represents
        left- and right-multiplication by ``Q`` under this module's vectorisation
        convention. The result acts on vectorised density matrices in Liouville
        space to produce ``vec(Qρ + ρQ)``.

        Args:
            Q (np.ndarray): Square observable/projector in Hilbert space of size ``N×N``.

        Returns:
            np.ndarray: Liouville-space operator of size ``N²×N²`` implementing
            left/right action by ``Q``.
        """
        eye = np.eye(len(Q))
        return np.kron(Q, eye) + np.kron(eye, Q)

    @property
    def hamiltonian_size(self):
        """Dimension of the Liouville space (Hilbert dimension squared)."""
        return super().hamiltonian_size ** 2

    @staticmethod
    def _square_liouville_rhos(rhos):
        """Reshape vectorised Liouville densities back to square matrices.

        Treats the second-to-last axis of ``rhos`` as a flattened square
        dimension of size ``dim**2`` and reshapes the trailing ``(..., dim**2, 1)``
        (or ``(..., dim**2,)``) into ``(..., dim, dim)``. The Hilbert-space
        dimension ``dim`` is inferred as ``int(sqrt(rhos.shape[-2]))``.

        Args:
            rhos (np.ndarray): Array of vectorised density operators with shape
                ``(..., dim**2, 1)`` or ``(..., dim**2)``.

        Returns:
            np.ndarray: Array with the same leading dimensions but final shape
            ``(..., dim, dim)`` corresponding to square density matrices.

        Notes:
            This assumes ``rhos.shape[-2]`` is a perfect square; otherwise the
            inferred ``dim`` will be incorrect.
        """
        shape = rhos.shape
        dim = int(np.sqrt(shape[-2]))
        return rhos.reshape(*shape[:-2], dim, dim)

    @staticmethod
    def _get_rho_shape(dim):
        """Return the array shape for a density matrix in Liouville space.

        In Liouville-space simulations, the density operator is vectorised,
        so it is represented as a column vector of shape ``(dim**2, 1)``.
        This helper returns the correct shape for allocating such objects.

        Args:
            dim (int): Dimension of the underlying Hilbert space.

        Returns:
            tuple[int, int]: ``(dim, 1)``, the canonical Liouville density shape.
        """
        return (dim, 1)

    def liouville_projection_operator(self, state: State) -> np.ndarray:
        """Column-vectorised projector for Liouville space.

        Flattens the Hilbert-space projector associated with ``state`` to shape
        ``(N², 1)`` for use with Liouville evolution.
        """
        return np.reshape(self.projection_operator(state), (-1, 1))

    def observable_projection_operator(self, state: State) -> np.ndarray:
        """Row-vectorised observable (projectorᵀ) used for expectation values."""
        Q = self.liouville_projection_operator(state)
        return Q.T

    def initial_density_matrix(self, state: State, H: np.ndarray) -> np.ndarray:
        """Create an initial density matrix for time evolution of the spin
        Hamiltonian density matrix.

        Arguments:
            state (State): a string = spin state projection operator
            spins: an integer = sum of the number of electrons and nuclei
            H: a matrix = spin Hamiltonian in Hilbert space

        Returns:
            A matrix in Liouville space
        """
        Pi = self.liouville_projection_operator(state)
        if state == State.EQUILIBRIUM:
            rho0eq = sp.sparse.linalg.expm(-1j * H * Pi)
            rho0 = rho0eq / np.trace(rho0eq)
            rho0 = np.reshape(rho0, (len(H) ** 2, 1))
        else:
            rho0 = Pi / np.vdot(Pi, Pi)
        return rho0

    @staticmethod
    def unitary_propagator(H, dt):
        r"""Create unitary propagator (Liouville space).

        Create unitary propagator matrix **U** for the time evolution
        of the density matrix in Liouville space (for the spin
        Hamiltonian `H`).

        .. math::
            \mathbf{U} = \exp( \hat{\hat{L}} t )

        See also: `propagate` and `time_evolution`.

        Args:

            H (np.ndarray): Spin Hamiltonian in Liouville space.

            dt (float): Time evolution timestep.

        Returns:
            np.ndarray:

                The matrix in Liouville space which is used by the
                `propagate` method to perform a single step in the
                `time_evolution` method.

        Examples:

            >>> molecules = [Molecule.fromdb("flavin_anion", ["N5"]),
            ...              Molecule("Z")]
            >>> sim = LiouvilleSimulation(molecules)
            >>> H = sim.total_hamiltonian(B0=0, J=0, D=0)
            >>> sim.unitary_propagator(H, 3e-9).shape
            (144, 144)
        """
        return sp.sparse.linalg.expm(H * dt)

    def propagate(self, propagator: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Propagate the density matrix (Liouville space).

        Propagates the density matrix using the propagator obtained
        using the `unitary_propagator` method.

        .. math::
            \\rho (t) = \\mathbf{U} \\rho_0

        See also: `unitary_propagator` and `time_evolution`.

        Args:

            propagator (np.ndarray): Unitary operator obtained via the
                `unitary_propagator` method.

            rho (np.ndarray): (Initial) density matrix.

        Returns:
            np.ndarray:

                The new density matrix after the unitary operator was
                applied to it.
        """
        return propagator @ rho


class LiouvilleIncoherentProcessBase(HilbertIncoherentProcessBase):
    def adjust_hamiltonian(self, H: np.ndarray):
        """Subtract the prebuilt incoherent sub-Hamiltonian from ``H`` in Liouville space.

        Expects subclasses to define ``self.subH`` with the correct shape.
        """
        H -= self.subH

    # def adjust_hamiltonian(self, H: np.ndarray):
    #     sub = np.asarray(self.subH, dtype=H.dtype, order="C")   # <— fix dtype/object
    #     if sub.shape != H.shape:
    #         raise ValueError(
    #             f"Incoherent subH has shape {sub.shape} but H has shape {H.shape}."
    #             " Did you build BR in Hilbert and apply in Liouville with mismatched basis?"
    #         )
    #     H -= sub


class SemiclassicalSimulation(LiouvilleSimulation):
    # Expectations:
    #   - self.radicals: list/tuple of two spin-1/2 radicals
    #   - self.molecules: length-2 container aligned with radicals
    #   - self.spin_operator(ri, ax): returns D×D operator for radical ri and axis in {"x","y","z"}
    #   - External field accessor: self.external_field_mT() -> np.ndarray shape (3,) in mT (default zeros)

    def external_field_mT(self) -> np.ndarray:
        """Lab field B0 in mT as a 3-vector. Override if you have a field in the simulation."""
        return np.zeros(3, dtype=float)

    def semiclassical_HHs(
        self,
        num_samples: int,
        *,
        anisotropic: bool = False,
    ) -> np.ndarray:
        """
        Generate semiclassical Hamiltonians.

        For each radical r:
            draw a static random hyperfine field B_hf,r (3D Gaussian),
            add the external field B0, then convert to ω_r = γ_r * (B0 + B_hf,r),
            and construct H = Σ_r ω_r · S_r. Repeat num_samples times.

        Parameters
        ----------
        num_samples : int
            Number of random Hamiltonian realisations.
        anisotropic : bool, optional
            If True, draws from a 3D Gaussian with covariance Σ_r determined by anisotropic A tensors.
            If False (default), uses isotropic SW width.

        Returns
        -------
        np.ndarray
            Array (num_samples, D, D) of complex Hamiltonians.
        """
        # two S=1/2 radicals
        assert len(self.radicals) == 2
        assert self.radicals[0].multiplicity == 2
        assert self.radicals[1].multiplicity == 2

        R = len(self.radicals)

        # Spin operators Sx,Sy,Sz per radical -> (R,3,D,D)
        spinops = np.stack(
            [
                [np.asarray(self.spin_operator(ri, ax)) for ax in "xyz"]
                for ri in range(R)
            ],
            axis=0,
        ).astype(complex)
        _, _, D, _ = spinops.shape

        # Per-radical gyromagnetic ratio in rad s^-1 mT^-1
        gammas = np.array(
            [m.radical.gamma_mT for m in self.molecules], dtype=float
        )  # (2,)

        # External field (mT), broadcast to all samples and radicals
        B0 = np.asarray(self.external_field_mT(), dtype=float)  # (3,)
        if B0.shape != (3,):
            raise ValueError("external_field_mT() must return shape (3,) in mT.")

        # --- Draw hyperfine fields ---
        if not anisotropic:
            # Isotropic SW width: per-radical σ_B from σ_ω / γ
            stds_B = np.array(
                [m.semiclassical_std for m in self.molecules], dtype=float
            )  # (2,)

            # Random N(0, σ_B^2) for each component; shape (N,R,3)
            hf_fields = np.random.normal(
                loc=0.0,
                scale=stds_B[None, :, None],  # broadcast to (N,2,3)
                size=(num_samples, R, 3),
            )
        else:
            # --- Anisotropic version ---
            # Build per-radical 3x3 covariance matrices Σ_B,r in mT^2:
            #   Σ_ω,r = (1/3) Σ_k I_k(I_k+1) A_{rk} A_{rk}^T   (angular-frequency)
            #   Σ_B,r = Σ_ω,r / γ_r^2
            covs_B = []
            for r, m in enumerate(self.molecules):
                Sigma_omega = np.zeros((3, 3), dtype=float)
                for n in m.nuclei:
                    Ik = float(n.spin_quantum_number)
                    # n.hfc.anisotropic assumed to be a 3x3 Cartesian hyperfine tensor in angular-frequency units.
                    # If you only store an axial tensor, convert to full 3×3 first.
                    A = np.asarray(n.hfc.anisotropic, dtype=float)  # shape (3,3)
                    Sigma_omega += (Ik * (Ik + 1.0) / 3.0) * (A @ A.T)
                gamma = gammas[r]
                if gamma == 0.0:
                    covs_B.append(np.zeros((3, 3), dtype=float))
                else:
                    covs_B.append(Sigma_omega / (gamma**2))
            covs_B = np.stack(covs_B, axis=0)  # (2,3,3)

            # Draw from multivariate normal for each radical independently
            hf_fields = np.empty((num_samples, R, 3), dtype=float)
            for r in range(R):
                # Factorisation (np.linalg.cholesky requires SPD; use eigh)
                w, V = np.linalg.eigh(covs_B[r])
                w = np.clip(w, a_min=0.0, a_max=None)
                L = V @ np.diag(np.sqrt(w))
                z = np.random.normal(size=(num_samples, 3))
                hf_fields[:, r, :] = z @ L.T

        # Total field per sample/radical in mT
        B_tot = hf_fields + B0[None, None, :]  # (N,2,3)

        # Convert to angular frequency (ω = γ B) and contract with S
        comps = B_tot * gammas[None, :, None]  # (N,2,3) in angular frequency units
        HHs = np.einsum(
            "nra,raxy->nxy", comps, spinops, optimize=True
        )  # (N,D,D), complex

        return HHs

    @property
    def nuclei(self):
        """Semiclassical model replaces nuclei by random static fields; return empty list."""
        return []


class SparseCholeskyHilbertSimulation(HilbertSimulation):
    """
    A simulation class that exploits

    - Sparsity of the Hamiltonian
    - Cholesky decomposition of the density matrix

    Since density matrices are positive semi-definite, they can be decomposed into

    rho = X X^T

    where X is a lower triangular matrix by Cholesky decomposition.

    In particular, when rho is a diagonal matrix, X = sqrt(rho).

    This class accelerates the time evolution for the system with large Hilbert space (> 10^3).
    """

    def ST_basis(
        self, M: NDArray | sp.sparse.sparray, kron_eye: bool = True
    ) -> sp.sparse.sparray:
        """Sparse S/T-basis transform of an operator.

        Accepts dense or sparse input, converts to sparse as needed, and applies
        the electron-only change-of-basis while preserving sparsity.
        """
        assert kron_eye
        if not sp.sparse.issparse(M):
            M = sp.sparse.csc_matrix(M)
        # T+  T0  S  T-
        ST = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )
        ST = sp.sparse.csc_matrix(ST)
        C = sp.sparse.kron(
            ST, sp.sparse.eye(prod([n.multiplicity for n in self.nuclei]))
        )
        return C @ M @ C.T

    def spin_operator(
        self, idx: int, axis: str, kron_eye: bool = True
    ) -> sp.sparse.sparray:
        """Construct the spin operator.

        Construct the spin operator for the particle with index `idx`
        in the `HilbertSimulation`.

        Args:

            idx (int): Index of the particle.

            axis (str): Axis, i.e. ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            np.ndarray:

                Spin operator for a particle in the
                `HilbertSimulation` system with indexing `idx` and
                axis `axis`.
        """
        assert 0 <= idx and idx < len(self.particles)
        assert axis in "xyzpmu"
        assert kron_eye
        sigma = self.particles[idx].pauli[axis]
        before_size = prod(p.multiplicity for p in self.particles[:idx])
        after_size = prod(p.multiplicity for p in self.particles[idx + 1 :])
        spinop = sp.sparse.kron(sp.sparse.eye(before_size), sp.sparse.csr_matrix(sigma))
        spinop = sp.sparse.kron(spinop, sp.sparse.eye(after_size))
        if self.basis == Basis.ST:
            return self.ST_basis(spinop)
        else:
            return spinop

    def get_eye(self, shape: int) -> sp.sparse.sparray:
        """Return a sparse identity matrix of the requested dimension."""
        return sp.sparse.eye(shape)

    def time_evolution(
        self,
        init_state: State,
        time: np.ndarray,
        H: sp.sparse.sparray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evolve the system through time.

        See Also:
        - `HilbertSimulation.unitary_propagator`
        - `HilbertSimulation.propagate`

        Args:

            init_state (State): Initial `State` of the density matrix
                (see `projection_operator`).

            time (np.ndarray): An sequence of (uniform) time points,
                usually created using `np.arange` or `np.linspace`.

            H (np.ndarray): Hamiltonian operator.

        Returns:
            tuple[np.ndarray, np.ndarray]:

                Return a sequence of density matrices (X, X^T) evolved through
                `time`, starting from a given initial `state` using
                the Hamiltonian `H`.
                Density matrices are obtained by X X^T.

        Examples:
            >>> molecules = [Molecule.fromdb("flavin_anion", ["N5"]),
            ...              Molecule("Z")]
            >>> sim = SparseCholeskyHilbertSimulation(molecules)
            >>> H = sim.total_hamiltonian(B0=0, J=0, D=0)
            >>> time = np.arange(0, 2e-6, 5e-9)
            >>> time.shape
            (400,)
            >>> rhos = sim.time_evolution(State.SINGLET, time, H)
            >>> len(rhos)
            400
            >>> rhos[0][0].shape
            (12, 12)
            >>> rhos[0][1].shape
            (12, 12)
        """
        dt = time[1] - time[0]
        propagator = self.unitary_propagator(H, dt)
        rho0 = self.initial_density_matrix(init_state, H)
        rhos = [None for _ in range(len(time))]

        def is_sparse_diagonal(A, atol=1e-12) -> bool:
            """Return True iff A is a (square) diagonal matrix.
            Works for any SciPy sparse type without densifying."""
            m, n = A.shape
            if m != n:
                return False
            C = A.tocoo()  # just reindexes the nnz, still sparse
            C.sum_duplicates()  # combine duplicate entries
            C.data[np.abs(C.data) < atol] = 0.0
            C.eliminate_zeros()  # drop explicit zeros if any
            # all nonzeros must lie on the main diagonal
            return np.all(C.row == C.col)

        if sp.sparse.issparse(rho0) and is_sparse_diagonal(rho0):
            # rho0 is diagonal
            L = sp.sparse.diags_array(np.sqrt(rho0.diagonal()))
            L = L.tocsc()
        elif isinstance(rho0, np.ndarray) and np.all(np.diag(rho0) == rho0):
            L = np.diag(np.sqrt(rho0.diagonal()))
            L = L.tocsc()
        else:
            rho0 = np.array(rho0)
            L = np.linalg.cholesky(rho0)
        rhos[0] = (L, L.conj().T)
        for t in range(1, len(time)):
            rhos[t] = self.propagate(propagator, rhos[t - 1])
        return rhos

    def unitary_propagator(
        self, H: sp.sparse.sparray, dt: float
    ) -> sp.sparse.sparray | np.ndarray:
        """Sparse/dense unitary for one time step in Hilbert space.

        Builds ``U = exp(-i H dt)`` as a sparse matrix; if the result becomes
        too dense (fill > 50%), it is converted to a dense ndarray.

        Args:
            H: Sparse Hamiltonian (CSC preferred).
            dt: Time step (s).

        Returns:
            Sparse or dense unitary matrix depending on fill ratio.
        """
        if not isinstance(H, sp.sparse.csc_matrix):
            H = sp.sparse.csc_matrix(H)
        Um = sp.sparse.linalg.expm(-1j * H * dt)
        if Um.nnz / np.prod(Um.shape) > 0.5:
            Um = Um.toarray()
        return Um

    def propagate(
        self,
        propagator: sp.sparse.sparray,
        rho: tuple[sp.sparse.sparray | np.ndarray, sp.sparse.sparray | np.ndarray],
    ) -> tuple[sp.sparse.sparray | np.ndarray, sp.sparse.sparray | np.ndarray]:
        """Propagate a Cholesky factorisation ``(X, Xᵀ)`` one step.

        Uses ``Um @ X`` to advance the factor; the updated pair
        ``(UmX, (UmX)ᴴ)`` represents the new density without forming
        ``ρ = X Xᵀ`` explicitly. Automatically switches to dense math
        when sparsity falls below a threshold (~30% zeros in factors).
        """
        # if more than 30 % of the elements are non-zero, switch to dense
        X, Xt = rho
        if not sp.sparse.issparse(X) and not sp.sparse.issparse(propagator):
            UmX = sp.linalg.blas.dgemm(alpha=1.0, a=propagator, b=X)
        else:
            if sp.sparse.issparse(X) and X.nnz / np.prod(X.shape) > 0.3:
                X = X.toarray()
            UmX = propagator @ X
        return (UmX, UmX.conj().T)

    def product_probability(
        self,
        obs: State,
        rhos: list[
            tuple[sp.sparse.sparray | np.ndarray, sp.sparse.sparray | np.ndarray]
        ],
    ) -> np.ndarray:
        """Calculate the probability of the observable from the densities."""
        if obs == State.EQUILIBRIUM:
            raise ValueError("Observable state should not be EQUILIBRIUM")
        Q = self.observable_projection_operator(obs)
        assert isinstance(rhos, list)
        if isinstance(rhos[0], tuple) and len(rhos[0]) == 2:
            # Cholesky factorization
            # tr(Qrho) = tr(QXX^T) = tr(X^TQX)
            Qrhos = [(Xt @ Q @ X).trace().real for (X, Xt) in rhos]
        else:
            Qrhos = [(Q @ rho).trace().real for rho in rhos]
        return np.array(Qrhos)
