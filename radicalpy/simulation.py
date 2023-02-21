#!/usr/bin/env python

import enum
from math import prod
from typing import Iterable, Optional

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

        Return:
            dict: A dictionary containing 6 `np.array` matrices of
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

        Args:

            idx (int): Index of the particle.

            axis (str): Axis, i.e. ``"x"``, ``"y"`` or ``"z"``.

        Returns:

            np.ndarray: Spin operator for a particle in the
            `HilbertSimulation` system with indexing `idx` and axis
            `axis`.

        Construct the spin operator for the particle with index
        `idx` in the `HilbertSimulation`.

        """
        assert 0 <= idx and idx < len(self.particles)
        assert axis in "xyzpmu"

        sigma = self.pauli(self.particles[idx].multiplicity)[axis]
        eye_before = np.eye(prod(p.multiplicity for p in self.particles[:idx]))
        eye_after = np.eye(prod(p.multiplicity for p in self.particles[idx + 1 :]))

        spinop = np.kron(np.kron(eye_before, sigma), eye_after)
        if self.basis == Basis.ST:
            return self.ST_basis(spinop)
        else:
            return spinop

    def product_operator(self, idx1: int, idx2: int, h: float = 1) -> np.ndarray:
        """Projection operator."""
        return h * sum(
            [
                self.spin_operator(idx1, axis).dot(self.spin_operator(idx2, axis))
                for axis in "xyz"
            ]
        )

    def product_operator_3d(self, idx1: int, idx2: int, h: np.ndarray) -> np.ndarray:
        """Projection operator."""
        return sum(
            [
                h[i, j]
                * self.spin_operator(idx1, ax1).dot(self.spin_operator(idx2, ax2))
                for i, ax1 in enumerate("xyz")
                for j, ax2 in enumerate("xyz")
            ]
        )

    def projection_operator(self, state: State):
        """Construct.

        .. todo::     Write proper docs.
        """
        # Spin operators
        SAx, SAy, SAz = [self.spin_operator(0, ax) for ax in "xyz"]
        SBx, SBy, SBz = [self.spin_operator(1, ax) for ax in "xyz"]

        # Product operators
        SASB = self.product_operator(0, 1)

        eye = np.eye(len(SASB))

        # Projection operators
        # todo change p/m to +/-
        match state:
            case State.SINGLET:
                return (1 / 4) * eye - SASB
            case State.TRIPLET:
                return (3 / 4) * eye + SASB
            case State.TRIPLET_PLUS:
                return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz)
            case State.TRIPLET_MINUS:
                return (2 * SAz**2 - SAz) * (2 * SBz**2 - SBz)
            case State.TRIPLET_ZERO:
                return (1 / 4) * eye + SAx @ SBx + SAy @ SBy - SAz @ SBz
            case State.TRIPLET_PLUS_MINUS:
                return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz) + (
                    2 * SAz**2 - SAz
                ) * (2 * SBz**2 - SBz)
            case State.EQUILIBRIUM:
                return 1.05459e-34 / (1.38e-23 * 298)

    def zeeman_hamiltonian(
        self, B0: float, theta: Optional[float] = None, phi: Optional[float] = None
    ) -> np.ndarray:
        """Construct the Zeeman Hamiltonian.

        Construct the Zeeman Hamiltonian based on the external
        magnetic field `B`.

        Args:

            B0 (float): External magnetic field intensity (milli
            Tesla).

        Returns:
            np.ndarray: The Zeeman Hamiltonian corresponding to the
            system described by the `Quantum` simulation object and
            the external magnetic field intensity `B`.
        """
        if theta is None and phi is None:
            return self.zeeman_hamiltonian_1d(B0)
        else:
            return self.zeeman_hamiltonian_3d(B0, theta, phi)

    def zeeman_hamiltonian_1d(self, B0: float) -> np.ndarray:
        axis = "z"
        gammas = enumerate(p.gamma_mT for p in self.particles)
        return -B0 * sum(g * self.spin_operator(i, axis) for i, g in gammas)

    def zeeman_hamiltonian_3d(
        self, B0: float, theta: float = 0, phi: float = 0
    ) -> np.ndarray:
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

        Construct the Hyperfine Hamiltonian based on the magnetic
        field.

        Returns:
            np.ndarray: The Hyperfine Hamiltonian corresponding to the
            system described by the `Quantum` simulation object.
        """
        if hfc_anisotropy:
            for h in [n.hfc for n in self.nuclei]:
                # TODO(vatai) try except not is None
                if h.anisotropic is None:
                    raise ValueError(
                        "Not all molecules have anisotropic HFCs! Please use `hfc_anisotropy=False`"
                    )

        if hfc_anisotropy:
            prodop = self.product_operator_3d
            hfcs = [n.hfc.anisotropic for n in self.nuclei]
        else:
            prodop = self.product_operator
            hfcs = [n.hfc.isotropic for n in self.nuclei]
        return sum(
            [
                self.particles[ei].gamma_mT
                * prodop(ei, len(self.radicals) + ni, hfcs[ni])
                for ni, ei in enumerate(self.coupling)
            ]
        )

    def exchange_hamiltonian(self, J: float) -> np.ndarray:
        """Construct the Exchange Hamiltonian.

        Construct the Exchange (J-coupling) Hamiltonian based on the
        coupling constant J between two electrons, which can be obtain
        from the radical pair separation `r` using `TODO` method.

        .. todo::
            Write proper docs.

        Returns:
            np.ndarray: The Exchange (J-coupling) Hamiltonian
            corresponding to the system described by the `Quantum`
            simulation object and the coupling constant `J`.
        """
        Jcoupling = self.radicals[0].gamma_mT * J
        SASB = self.product_operator(0, 1)
        return Jcoupling * (2 * SASB + 0.5 * np.eye(*SASB.shape))

    def dipolar_hamiltonian(self, D: float or np.ndarray) -> np.ndarray:
        """Construct the Dipolar Hamiltonian.

        Construct the Dipolar Hamiltonian based on dipolar coupling
        constant `D` between two electrons.

        .. todo::
            Write proper docs.

        Returns:
            np.ndarray: The Dipolar Hamiltonian corresponding to the
            system described by the `Quantum` simulation object and
            dipolar coupling constant `D`.
        """
        if isinstance(D, np.ndarray):
            return self.dipolar_hamiltonian_3d(D)
        else:
            return self.dipolar_hamiltonian_1d(D)

    def dipolar_hamiltonian_1d(self, D: float) -> np.ndarray:
        SASB = self.product_operator(0, 1)
        SAz = self.spin_operator(0, "z")
        SBz = self.spin_operator(1, "z")
        omega = (2 / 3) * self.radicals[0].gamma_mT * D
        return omega * (3 * SAz * SBz - SASB)

    def dipolar_hamiltonian_3d(self, dipolar_tensor: np.ndarray) -> np.ndarray:
        ne = len(self.radicals)
        return -sum(
            [
                -self.radicals[0].gamma_mT
                * self.product_operator_3d(ei, ne + ni, dipolar_tensor)
                for ni, ei in enumerate(self.coupling)
            ]
        )

    def total_hamiltonian(
        self,
        B: float,
        J: float,
        D: float,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        hfc_anisotropy: bool = False,
    ) -> np.ndarray:
        """Construct the final (total) Hamiltonian.

        Construct the final (total)

        .. todo::
            Write proper docs.
        """
        H = (
            self.zeeman_hamiltonian(B, theta, phi)
            + self.hyperfine_hamiltonian(hfc_anisotropy)
            + self.exchange_hamiltonian(J)
            + self.dipolar_hamiltonian(D)
        )
        return self.convert(H)

    def time_evolution(
        self, init_state: State, time: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Evolve the system through time.

        Args:
                init_state (State): blah blah

                time (np.ndarray): blah blah

                H (np.ndarray): blah blah

        Returns:
                np.ndarray: blah blah
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
        product_yield = sp.integrate.cumtrapz(product_probability, time, initial=0) * k
        product_yield_sum = np.trapz(product_probability, dx=time[1]) * k
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
            List generator.

        .. todo::
            Write proper docs.
        """
        H_zee = self.zeeman_hamiltonian(1, theta, phi)
        shape = H_zee.shape
        H_zee = self.convert(H_zee)
        if shape != H_zee.shape:
            shape = [shape[0] * shape[0], 1]
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
        H = self.total_hamiltonian(B=0, D=D, J=J, hfc_anisotropy=hfc_anisotropy)

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
        B: float,
        H_base: np.ndarray,
        theta: Iterable[float],
        phi: Iterable[float],
    ):
        shape = H_base.shape
        H_base = self.convert(H_base)
        if shape != H_base.shape:
            shape = [shape[0] * shape[0], 1]

        rhos = np.zeros([len(theta), len(phi), len(time), *shape], dtype=complex)

        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                H_zee = self.zeeman_hamiltonian(B, th, ph)
                H = H_base + H_zee
                rhos[i, j] = self.time_evolution(init_state, time, H)
        return rhos

    def anisotropy(
        self,
        init_state: State,
        obs_state: State,
        time: np.ndarray,
        theta: Iterable or float,
        phi: Iterable or float,
        B: float,
        D: np.ndarray,
        J: float,
        kinetics: list[HilbertIncoherentProcessBase] = [],
        relaxations: list[HilbertIncoherentProcessBase] = [],
    ) -> dict:
        H = self.total_hamiltonian(B=0, D=D, J=J, hfc_anisotropy=True)

        self.apply_liouville_hamiltonian_modifiers(H, kinetics + relaxations)
        theta, phi = utils._anisotropy_check(theta, phi)
        rhos = self.anisotropy_loop(init_state, time, B, H, theta=theta, phi=phi)
        product_probabilities = self.product_probability(obs_state, rhos)

        self.apply_hilbert_kinetics(time, product_probabilities, kinetics)
        k = kinetics[0].rate_constant if kinetics else 1.0
        product_yields, product_yield_sums = self.product_yield(
            product_probabilities, time, k
        )
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
            np.ndarray: A matrix in Hilbert space representing...
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

        Create unitary propagator matrices for time evolution of the
        spin Hamiltonian density matrix in Hilbert space.

        Arguments:

            H (np.ndarray): Spin Hamiltonian in Hilbert space.

            dt (float): Time evolution timestep.

        Returns:
            np.ndarray: Two matrices (a tensor) in either Hilbert.

        .. todo::
            https://docs.python.org/3/library/doctest.html

        Example:
            >> Up, Um = UnitaryPropagator(H, 3e-9, "Hilbert")
            >> UL = UnitaryPropagator(HL, 3e-9, "Liouville")
        """
        Up = sp.sparse.linalg.expm(1j * H * dt)
        Um = sp.sparse.linalg.expm(-1j * H * dt)
        return Up, Um

    def propagate(self, propagator: np.ndarray, rho: np.ndarray) -> np.ndarray:
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
        return rhos.reshape(shape[0], shape[1], dim, dim)

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
        """Create unitary propagator.

        Create unitary propagator matrices for time evolution of the
        spin Hamiltonian density matrix in both Hilbert and Liouville
        space.

        Arguments:
            H (np.ndarray): Spin Hamiltonian in Hilbert or Liouville space
            dt (float): Time evolution timestep.
            space (str): Select the spin space.
        """
        return sp.sparse.linalg.expm(H * dt)

    def propagate(self, propagator: np.ndarray, rho: np.ndarray) -> np.ndarray:
        return propagator @ rho


class LiouvilleIncoherentProcessBase(HilbertIncoherentProcessBase):
    def adjust_hamiltonian(self, H: np.ndarray):
        H -= self.subH
