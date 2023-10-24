#!/usr/bin/env python

import enum
import itertools
from math import prod
from typing import Optional

import numpy as np
import scipy as sp
from tqdm import tqdm

from . import utils
from .data import Molecule


class State(enum.Enum):
    EQUILIBRIUM = "Eq"
    SINGLET = "S"
    TRIPLET = "T"
    TRIPLET_ZERO = "T_0"
    TRIPLET_PLUS = "T_+"
    TRIPLET_PLUS_MINUS = "T_\\pm"
    TRIPLET_MINUS = "T_-"


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

    @staticmethod
    def pauli(mult: int):
        """Generate Pauli matrices.

        Generates the Pauli matrices corresponding to a given multiplicity.

        Args:

            mult (int): The multiplicity of the element.

        Returns:
            dict:

                A dictionary containing 6 `np.array` matrices of
                shape `(mult, mult)`:

                - the unit operator `result["u"]`,
                - raising operator `result["p"]`,
                - lowering operator `result["m"]`,
                - Pauli matrix for x axis `result["x"]`,
                - Pauli matrix for y axis `result["y"]`,
                - Pauli matrix for z axis `result["z"]`.
        """
        assert mult > 1
        result = {}
        if mult == 2:
            result["u"] = np.array([[1, 0], [0, 1]])
            result["p"] = np.array([[0, 1], [0, 0]])
            result["m"] = np.array([[0, 0], [1, 0]])
            result["x"] = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
            result["y"] = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
            result["z"] = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
        else:
            spin = (mult - 1) / 2
            prjs = np.arange(mult - 1, -1, -1) - spin

            p_data = np.sqrt(spin * (spin + 1) - prjs * (prjs + 1))
            m_data = np.sqrt(spin * (spin + 1) - prjs * (prjs - 1))

            result["u"] = np.eye(mult)
            result["p"] = sp.sparse.spdiags(p_data, [1], mult, mult).toarray()
            result["m"] = sp.sparse.spdiags(m_data, [-1], mult, mult).toarray()
            result["x"] = 0.5 * (result["p"] + result["m"])
            result["y"] = -0.5 * 1j * (result["p"] - result["m"])
            result["z"] = sp.sparse.spdiags(prjs, 0, mult, mult).toarray()
        return result

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

        sigma = self.pauli(self.particles[idx].multiplicity)[axis]
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

            h (float): Isotopic interaction constant.

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

    def projection_operator(self, state: State):
        """Construct the projection operator corresponding to a `state`.

        Args:

            state (State): The target state which is projected out of
                the density matrix.

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
            State.EQUILIBRIUM: 1.05459e-34 / (1.38e-23 * 298),
        }
        return result[state]

    def zeeman_hamiltonian(
        self, B0: float, theta: Optional[float] = None, phi: Optional[float] = None
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
            return self.zeeman_hamiltonian_1d(B0)
        else:
            return self.zeeman_hamiltonian_3d(B0, theta, phi)

    def zeeman_hamiltonian_1d(self, B0: float) -> np.ndarray:
        """Construct the 1D Zeeman Hamiltonian.

        Construct the 1D Zeeman Hamiltonian based on the external
        magnetic field `B0`.

        Args:

            B0 (float): External magnetic field intensity (milli
                Tesla).

        Returns:
            np.ndarray:

                The Zeeman Hamiltonian corresponding to the system
                described by the `HilbertSimulation` object and the
                external magnetic field intensity `B0`.

        """
        axis = "z"
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
                [self.spin_operator(idx, axis) for axis in "xyz"]
                for idx in range(len(self.particles))
            ]
        )
        rotation = utils.spherical_to_cartesian(theta, phi)
        omega = B0 * self.radicals[0].gamma_mT
        return omega * np.einsum("j,ijkl->kl", rotation, particles)

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

    def exchange_hamiltonian(self, J: float) -> np.ndarray:
        """Construct the exchange Hamiltonian.

        Construct the exchange (J-coupling) Hamiltonian based on the
        coupling constant J between two electrons.

        The J-coupling constant can be obtained using:

        - `radicalpy.estimations.exchange_interaction_in_protein`
        - `radicalpy.estimations.exchange_interaction_in_solution`

        Args:

            J (float): Exchange coupling constant.

        Returns:
            np.ndarray:

                The exchange (J-coupling) Hamiltonian corresponding to
                the system described by the `HilbertSimulation` object
                and the coupling constant `J`.

        """
        Jcoupling = J * self.radicals[0].gamma_mT
        SASB = self.product_operator(0, 1)
        return Jcoupling * (2 * SASB + 0.5 * np.eye(SASB.shape[0]))

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
        return omega * (3 * SAz * SBz - SASB)

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
        ne = len(self.radicals)
        return -sum(
            (
                -self.radicals[0].gamma_mT
                * self.product_operator_3d(ei, ne + ni, dipolar_tensor)
                for ni, ei in enumerate(self.coupling)
            )
        )

    def zero_field_splitting_hamiltonian(self) -> np.ndarray:
        """Construct the Zero Field Splitting (ZFS) Hamiltoninan."""
        Sx, Sy, Sz = spinops(pos, 2, spin=3)
        Ssquared = Sx @ Sx + Sy @ Sy + Sz @ Sz
        return D * (Sz @ Sz - (1 / 3) * Ssquared) + E * ((Sx @ Sx) - (Sy @ Sy))

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
            self.zeeman_hamiltonian(B0, theta, phi)
            + self.hyperfine_hamiltonian(hfc_anisotropy)
            + self.exchange_hamiltonian(J)
            + self.dipolar_hamiltonian(D)
        )
        return self.convert(H)

    def time_evolution(
        self, init_state: State, time: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Evolve the system through time.

        See also:

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
        obs = self.observable_projection_operator(obs)
        return np.real(np.trace(obs @ rhos, axis1=-2, axis2=-1))

    @staticmethod
    def product_yield(product_probability, time, k):
        """Calculate the product yield and the product yield sum."""
        product_yield = k * sp.integrate.cumtrapz(product_probability, time, initial=0)
        product_yield_sum = k * np.trapz(product_probability, dx=time[1])
        return product_yield, product_yield_sum

    def apply_liouville_hamiltonian_modifiers(self, H, modifiers):
        for K in modifiers:  # skip in hilbert
            K.init(self)
            K.adjust_hamiltonian(H)

    @staticmethod
    def apply_hilbert_kinetics(time, product_probabilities, kinetics):
        for K in kinetics:  # skip in liouville
            K.adjust_product_probabilities(product_probabilities, time)

    def mary_loop(
        self,
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
        H_zee = self.convert(self.zeeman_hamiltonian(1, theta, phi))
        shape = self._get_rho_shape(H_zee.shape[0])
        rhos = np.zeros([len(B), len(time), *shape], dtype=complex)
        for i, B0 in enumerate(tqdm(B)):
            H = H_base + B0 * H_zee
            H_sparse = sp.sparse.csc_matrix(H)
            rhos[i] = self.time_evolution(init_state, time, H_sparse)
        return rhos

    @staticmethod
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
        minmax = max if init_state == State.SINGLET else min
        HFE = (MARY[-1] - MARY[idx]) / MARY[idx] * 100
        LFE = (minmax(MARY) - MARY[idx]) / MARY[idx] * 100
        MARY = (MARY - MARY[idx]) / MARY[idx] * 100
        return MARY, LFE, HFE

    @staticmethod
    def _square_liouville_rhos(rhos):
        return rhos

    @staticmethod
    def _get_rho_shape(dim):
        return dim, dim

    def MARY(
        self,
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
        H = self.total_hamiltonian(B0=0, D=D, J=J, hfc_anisotropy=hfc_anisotropy)

        self.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
        rhos = self.mary_loop(init_state, time, B, H, theta=theta, phi=phi)
        product_probabilities = self.product_probability(obs_state, rhos)

        self.apply_hilbert_kinetics(time, product_probabilities, kinetics)
        k = kinetics[0].rate_constant if kinetics else 1.0
        product_yields, product_yield_sums = self.product_yield(
            product_probabilities, time, k
        )

        dt = time[1] - time[0]
        MARY, LFE, HFE = self.mary_lfe_hfe(init_state, B, product_probabilities, dt, k)
        rhos = self._square_liouville_rhos(rhos)

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

    def anisotropy_loop(
        self,
        init_state: State,
        time: np.ndarray,
        B0: float,
        H_base: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """Inner loop of anisotropy experiment.

        Args:

            init_state (State): Initial `State` of the density matrix.

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
        shape = self._get_rho_shape(H_base.shape[0])
        rhos = np.zeros((len(theta), len(phi), len(time), *shape), dtype=complex)

        iters = itertools.product(enumerate(theta), enumerate(phi))
        for (i, th), (j, ph) in tqdm(list(iters)):
            H_zee = self.zeeman_hamiltonian(B0, th, ph)
            H = H_base + self.convert(H_zee)
            rhos[i, j] = self.time_evolution(init_state, time, H)
        return rhos

    def anisotropy(
        self,
        init_state: State,
        obs_state: State,
        time: np.ndarray,
        theta: np.ndarray | float,
        phi: np.ndarray | float,
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

            H_base (np.ndarray): A "base" Hamiltonian, i.e., the
                Zeeman Hamiltonian will be added to this base, usually
                obtained with `total_hamiltonian` and `B0=0`.

            theta (np.ndarray): rotation (polar) angle between the
                external magnetic field and the fixed molecule. See
                `zeeman_hamiltonian_3d`.

            B0 (float): External magnetic field intensity (milli
                Tesla) (see `zeeman_hamiltonian`).

            phi (np.ndarray): rotation (azimuth) angle between the
                external magnetic field and the fixed molecule. See
                `zeeman_hamiltonian_3d`.

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
        H = self.total_hamiltonian(B0=0, D=D, J=J, hfc_anisotropy=True)

        self.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
        theta, phi = utils.anisotropy_check(theta, phi)
        rhos = self.anisotropy_loop(init_state, time, B0, H, theta=theta, phi=phi)
        product_probabilities = self.product_probability(obs_state, rhos)

        self.apply_hilbert_kinetics(time, product_probabilities, kinetics)
        k = kinetics[0].rate_constant if kinetics else 1.0
        product_yields, product_yield_sums = self.product_yield(
            product_probabilities, time, k
        )
        rhos = self._square_liouville_rhos(rhos)

        return dict(
            time=time,
            B0=B0,
            theta=theta,
            phi=phi,
            rhos=rhos,
            time_evolutions=product_probabilities,
            product_yields=product_yields,
            product_yield_sums=product_yield_sums,
        )

    @staticmethod
    def convert(H: np.ndarray) -> np.ndarray:
        return H

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
        """Create unitary propagator (Hilbert space).

        Create unitary propagator matrices **U** and **U*** for time
        evolution of the density matrix in Hilbert space (for the spin
        Hamiltonian `H`).

        .. math::
            \mathbf{U}   =& \exp( -i \hat{H} t ) \\\\
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
        return 1j * (np.kron(H, eye) - np.kron(eye, H.T))

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
        """Create unitary propagator (Liouville space).

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
