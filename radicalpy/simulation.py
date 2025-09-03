#!/usr/bin/env python

import enum
import itertools
from math import prod
from typing import Iterator, Optional

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from tqdm import tqdm

from . import utils
from .data import Molecule
from .shared import constants as C


class State(enum.Enum):
    EQUILIBRIUM = "Eq"
    EPR = "EPR"
    SINGLET = "S"
    TRIPLET = "T"
    TRIPLET_ZERO = "T_0"
    TRIPLET_PLUS = "T_+"
    TRIPLET_PLUS_MINUS = "T_\\pm"
    TRIPLET_MINUS = "T_-"
    TP_SINGLET = "TP_S"


class Basis(enum.Enum):
    ZEEMAN = "Zeeman"
    ST = "ST"


class HilbertIncoherentProcessBase:
    def __init__(self, rate_constant: float):
        self.rate = rate_constant

    def init(self, sim):
        pass

    def adjust_hamiltonian(self, *args, **kwargs):
        return

    def adjust_product_probabilities(self, *args, **kwargs):
        return

    @property
    def rate_constant(self) -> float:
        """Rate of the incoherent process."""
        return self.rate

    def _name(self) -> str:
        return str(type(self).__name__)

    def __repr__(self) -> str:
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
        self.molecules = molecules
        self.custom_gfactors = custom_gfactors
        self.basis = basis

    @property
    def coupling(self):
        return sum([[i] * len(m.nuclei) for i, m in enumerate(self.molecules)], [])

    @property
    def radicals(self):
        return [m.radical for m in self.molecules]

    @property
    def nuclei(self):
        return sum([[n for n in m.nuclei] for m in self.molecules], [])

    @property
    def particles(self):
        return self.radicals + self.nuclei

    @property
    def hamiltonian_size(self):
        return np.prod([p.multiplicity for p in self.particles])

    def __repr__(self) -> str:
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

    def ST_basis(self, M):
        # T+  T0  S  T-
        ST = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )

        C = np.kron(ST, np.eye(prod([n.multiplicity for n in self.nuclei])))
        return C @ M @ C.T

    def spin_operator(self, idx: int, axis: str) -> np.ndarray:
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

        sigma = self.particles[idx].pauli[axis]
        before_size = prod(p.multiplicity for p in self.particles[:idx])
        after_size = prod(p.multiplicity for p in self.particles[idx + 1 :])
        spinop = np.kron(np.eye(before_size), sigma)
        spinop = np.kron(spinop, np.eye(after_size))
        if self.basis == Basis.ST:
            return self.ST_basis(spinop)
        else:
            return spinop

    def product_operator(self, idx1: int, idx2: int, h: float = 1.0) -> np.ndarray:
        """Construct the (1D) product operator.

        Construct the 1D (isotropic) product operator of two particles
        in the spin system.

        Args:

            idx1 (int): Index of the first particle.

            idx2 (int): Index of the second particle.

            h (float): Isotropic interaction constant.

        Returns:
            np.ndarray:

                Product operator for particles corresponding to `idx1`
                and `idx2` with isotropic interaction constant `h`.

        """
        return h * sum(
            [
                self.spin_operator(idx1, axis).dot(self.spin_operator(idx2, axis))
                for axis in "xyz"
            ]
        )

    def product_operator_3d(self, idx1: int, idx2: int, h: np.ndarray) -> np.ndarray:
        """Construct the 3D product operator.

        Construct the 3D (anisotropic) product operator of two
        particles in the spin system.

        Args:

            idx1 (int): Index of the first particle.

            idx2 (int): Index of the second particle.

            h (np.ndarray): Anisotropic interaction tensor.

        Returns:
            np.ndarray:

                Product operator for particles corresponding to `idx1`
                and `idx2` with anisotropic interaction tensor `h`.

        """
        return sum(
            (
                h[i, j]
                * self.spin_operator(idx1, ax1).dot(self.spin_operator(idx2, ax2))
                for i, ax1 in enumerate("xyz")
                for j, ax2 in enumerate("xyz")
            )
        )

    def projection_operator(self, state: State, T: float = 298):
        """Construct the projection operator corresponding to a `state`.

        Args:

            state (State): The target state which is projected out of
                the density matrix.
            T (float): Temperature for the EQUILIBRIUM projection operator (K).

        Returns:
            np.ndarray:

                Projection operator corresponding to the `State`
                `state`.

        """
        # Spin operators
        SAx, SAy, SAz = [self.spin_operator(0, ax) for ax in "xyz"]
        SBx, SBy, SBz = [self.spin_operator(1, ax) for ax in "xyz"]

        # Product operators
        SASB = self.product_operator(0, 1)
        eye = np.eye(len(SASB))

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
            State.EPR: -(SAy + SBy),
        }

        return result[state]

    def tp_singlet_projop(self, SAx, SAy, SAz, SBx, SBy, SBz):
        # For radical triplet pair (RTP)
        SAsquared = SAx @ SAx + SAy @ SAy + SAz @ SAz
        SBsquared = SBx @ SBx + SBy @ SBy + SBz @ SBz
        Ssquared = SAsquared + SBsquared + 2 * (SAx @ SBx + SAy @ SBy + SAz @ SBz)  #
        return (
            (1 / 12)
            * (Ssquared - (6 * np.eye(len(SAx))))
            @ (Ssquared - (2 * np.eye(len(SAx))))
        )

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
        particles = np.array(
            [
                [p.gamma_mT * self.spin_operator(idx, axis) for axis in "xyz"]
                for idx, p in enumerate(self.particles)
            ]
        )
        rotation = utils.spherical_to_cartesian(theta, phi)
        return -B0 * np.einsum("j,ijkl->kl", rotation, particles)

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
                self.particles[ei].gamma_mT
                * prodop(ei, len(self.radicals) + ni, hfcs[ni])
                for ni, ei in enumerate(self.coupling)
            )
        )

    def exchange_hamiltonian(self, J: float, prod_coeff: float = 2) -> np.ndarray:
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

        Returns:
            np.ndarray:

                The exchange (J-coupling) Hamiltonian corresponding to
                the system described by the `HilbertSimulation` object
                and the coupling constant `J`.

        """
        Jcoupling = J * self.radicals[0].gamma_mT
        SASB = self.product_operator(0, 1)
        return Jcoupling * (prod_coeff * SASB + 0.5 * np.eye(SASB.shape[0]))

    def dipolar_hamiltonian(self, D: float | np.ndarray) -> np.ndarray:
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

        Returns:
            np.ndarray:

                The Dipolar Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and
                dipolar coupling constant or dipolar interaction
                tensor `D`.

        """
        if isinstance(D, np.ndarray):
            return self.dipolar_hamiltonian_3d(D)
        else:
            return self.dipolar_hamiltonian_1d(D)

    def dipolar_hamiltonian_1d(self, D: float) -> np.ndarray:
        """Construct the 1D Dipolar Hamiltonian.

        Construct the Dipolar Hamiltonian based on dipolar coupling
        constant `D` between two electrons.

        The dipolar coupling constant can be obtained using
        `radicalpy.estimations.dipolar_interaction_isotropic`.

        Args:

            D (float): dipolar coupling constant.

        Returns:
            np.ndarray:

                The 1D Dipolar Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and
                dipolar coupling constant `D`.

        """
        SASB = self.product_operator(0, 1)
        SAz = self.spin_operator(0, "z")
        SBz = self.spin_operator(1, "z")
        omega = (2 / 3) * self.radicals[0].gamma_mT * D
        return omega * (3 * SAz @ SBz - SASB)

    def dipolar_hamiltonian_3d(self, dipolar_tensor: np.ndarray) -> np.ndarray:
        """Construct the 3D Dipolar Hamiltonian.

        Construct the Dipolar Hamiltonian based on dipolar interaction
        tensor `D` between two electrons.

        The dipolar coupling tensor can be obtained using
        `radicalpy.estimations.dipolar_interaction_anisotropic`.

        Args:

            D (np.ndarray): dipolar interaction tensor.

        Returns:
            np.ndarray:

                The 3D Dipolar Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and
                dipolar interaction tensor `D`.

        """
        spinops = [
            [self.spin_operator(r, ax) for ax in "xyz"]
            for r, _ in enumerate(self.radicals)
        ]
        return sum(
            (
                dipolar_tensor[i, j] * (si @ sj)
                for i, si in enumerate(spinops[0])
                for j, sj in enumerate(spinops[1])
            )
        )

    def zero_field_splitting_hamiltonian(self, D, E) -> np.ndarray:
        """Construct the Zero Field Splitting (ZFS) Hamiltonian."""
        Dmod = D * -self.radicals[0].gamma_mT
        Emod = E * -self.radicals[0].gamma_mT
        result = complex(0.0)
        for idx, p in enumerate(self.particles):
            Sx = self.spin_operator(idx, "x")
            Sy = self.spin_operator(idx, "y")
            Sz = self.spin_operator(idx, "z")
            Ssquared = Sx @ Sx + Sy @ Sy + Sz @ Sz
            result += Dmod * (Sz @ Sz - (1 / 3) * Ssquared)
            result += Emod * ((Sx @ Sx) - (Sy @ Sy))
        return result

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
        for K in modifiers:  # skip in hilbert
            K.init(self)
            K.adjust_hamiltonian(H)

    @staticmethod
    def apply_hilbert_kinetics(time, product_probabilities, kinetics):
        for K in kinetics:  # skip in liouville
            K.adjust_product_probabilities(product_probabilities, time)

    @staticmethod
    def _square_liouville_rhos(rhos):
        return rhos

    @staticmethod
    def _get_rho_shape(dim):
        return dim, dim

    @staticmethod
    def convert(H: np.ndarray) -> np.ndarray:
        return H

    @staticmethod
    def _convert(Q: np.ndarray) -> np.ndarray:
        return Q

    def initial_density_matrix(self, state: State, H: np.ndarray) -> np.ndarray:
        """Create an initial desity matrix.

        Create an initial density matrix for time evolution of the
        spin Hamiltonian density matrix.

        Args:
            state (State): Spin state projection operator.

            H (np.ndarray): Spin Hamiltonian in Hilbert space.

        Returns:
            np.ndarray:

                A matrix in Hilbert space representing...

        """
        Pi = self.projection_operator(state)

        if state == State.EQUILIBRIUM:
            rho0eq = sp.sparse.linalg.expm(-1j * sp.sparse.csc_matrix(H) * Pi).toarray()
            rho0 = rho0eq / rho0eq.trace()
        else:
            rho0 = Pi / np.trace(Pi)
        return rho0

    @staticmethod
    def unitary_propagator(H: np.ndarray, dt: float) -> np.ndarray:
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
        return self.projection_operator(state)


class LiouvilleSimulation(HilbertSimulation):
    @staticmethod
    def convert(H: np.ndarray) -> np.ndarray:
        """Convert the Hamiltonian from Hilbert to Liouville space."""
        eye = np.eye(len(H))
        tmp = np.kron(H, eye) - np.kron(eye, H.T)
        return 1j * tmp

    @staticmethod
    def _convert(Q: np.ndarray) -> np.ndarray:
        eye = np.eye(len(Q))
        return np.kron(Q, eye) + np.kron(eye, Q)

    @property
    def hamiltonian_size(self):
        return super().hamiltonian_size ** 2

    @staticmethod
    def _square_liouville_rhos(rhos):
        shape = rhos.shape
        dim = int(np.sqrt(shape[-2]))
        return rhos.reshape(*shape[:-2], dim, dim)

    @staticmethod
    def _get_rho_shape(dim):
        return (dim, 1)

    def liouville_projection_operator(self, state: State) -> np.ndarray:
        return np.reshape(self.projection_operator(state), (-1, 1))

    def observable_projection_operator(self, state: State) -> np.ndarray:
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
        H -= self.subH


# class SemiclassicalSimulation(LiouvilleSimulation):
#    def semiclassical_gen(
#        self,
#        num_samples: int,
#        #B: float,
#    ) -> Iterator[NDArray[np.float_]]:
#        num_particles = len(self.radicals)
#        spinops = [
#            [self.spin_operator(ri, ax) for ax in "xyz"] for ri in range(num_particles)
#        ]
#        for i in range(num_samples):
#            result = complex(0)
#            for ri, m in enumerate(self.molecules):
#                std = m.semiclassical_std
#                Is = np.random.normal(0, std, size=1)
#                gamma = m.radical.gamma_mT
#                for ax in range(3):
#                    spinop = spinops[ri][ax]
#                    result += gamma * spinop * Is
#                #result += gamma * B * spinop
#            yield result


class SemiclassicalSimulation(LiouvilleSimulation):
    def semiclassical_HHs(
        self,
        num_samples: int,
    ) -> np.ndarray:
        assert len(self.radicals) == 2
        assert self.radicals[0].multiplicity == 2
        assert self.radicals[1].multiplicity == 2

        spinops = np.array([self.spin_operator(0, ax) for ax in "xyz"])
        cov = np.diag([m.semiclassical_std for m in self.molecules])
        samples = np.random.multivariate_normal(
            mean=[0, 0],
            cov=cov,
            size=(num_samples, 3),
        )
        result = np.einsum("nam,axy->nxy", samples, spinops) * 2
        return result * self.radicals[0].gamma_mT

    @property
    def nuclei(self):
        return []
