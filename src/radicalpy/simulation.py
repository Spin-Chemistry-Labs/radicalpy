#!/usr/bin/env python

import enum
from math import prod
from typing import Iterable

import numpy as np
import scipy as sp

from .data import (MOLECULE_DATA, SPIN_DATA, gamma_mT, get_molecules,
                   multiplicity)
from .pauli_matrices import pauli


class State(enum.Enum):
    EQUILIBRIUM = "Eq"
    SINGLET = "S"
    TRIPLET = "T"
    TRIPLET_ZERO = "T0"
    TRIPLET_PLUS = "T+"
    TRIPLET_PLUS_MINUS = "T+/-"
    TRIPLET_MINUS = "T-"


class Molecule:
    # DOCS ALMOST DONE
    """Class representing a molecule in a simulation.

    A molecule is represented by hyperfine coupling constants, spin
    multiplicities and gyromagnetic ratios of its nuclei.  When using
    the database, one needs to specify the name of the molecule and
    the list of its nuclei.

    >>> Molecule("adenine_cation", ["N6-H1"])
    Molecule: adenine_cation
      HFCs: [-0.63]
      multiplicities: [3]
      gammas(mT): [19337.792]
      number of particles: 1


    If the wrong molecule name is given, the error helps you find the
    valid options.

    >>> Molecule("foobar", ["H1"])
    Traceback (most recent call last):
    ...
    ValueError: Available molecules below:
    adenine_cation
    flavin_anion
    flavin_neutral
    trp_cation
    tyrosine_neutral

    Similarly, giving a list of incorrect atom names will also result
    in a helpful error message listing the available atoms.

    >>> Molecule("adenine_cation", ["buz"])
    Traceback (most recent call last):
    ...
    ValueError: Available nuclei below.
    N6-H2 (hfc = -0.66)
    N6-H1 (hfc = -0.63)
    C8-H (hfc = -0.55)

    Use element/atom names (and HFCs):

    >>> Molecule(nuclei=["1H", "14N"], hfcs=[0.41, 1.82])
    Molecule: N/A
      HFCs: [0.41, 1.82]
      multiplicities: [2, 3]
      gammas(mT): [267522.18744, 19337.792]
      number of particles: 2

    Same as above, but with molecule name:

    >>> Molecule("isotopes", nuclei=["15N", "15N"], hfcs=[0.3, 1.7])
    Molecule: isotopes
      HFCs: [0.3, 1.7]
      multiplicities: [2, 2]
      gammas(mT): [-27126.180399999997, -27126.180399999997]
      number of particles: 2

    A molecule with no HFCs:

    >>> Molecule("kryptonite")
    Molecule: kryptonite
      HFCs: []
      multiplicities: []
      gammas(mT): []
      number of particles: 0

    Manual input for all relevant values (multiplicities, gammas, HFCs):

    >>> Molecule(multiplicities=[2, 2, 3], gammas_mT=[267522.18744, 267522.18744, 19337.792], hfcs=[0.42, 1.01, 1.33])
    Molecule: N/A
      HFCs: [0.42, 1.01, 1.33]
      multiplicities: [2, 2, 3]
      gammas(mT): [267522.18744, 267522.18744, 19337.792]
      number of particles: 3

    >>> Molecule("my_flavin", multiplicities=[2], gammas_mT=[267522.18744], hfcs=[0.5])
    Molecule: my_flavin
      HFCs: [0.5]
      multiplicities: [2]
      gammas(mT): [267522.18744]
      number of particles: 1

    """

    def __init__(
        self,
        radical: str = "",
        nuclei: list[str] = [],
        multiplicities: list[int] = [],
        gammas_mT: list[float] = [],
        hfcs: list[float] = [],
    ):
        """Construct a Molecule object.

        Args:
            radical (str): the name of the molecule, defaults to ""

            nuclei (list[str]): list of atoms from the molecule (or
                from the database), defaults to []

            multiplicities (list[int]): list of multiplicities of the
                atoms and their isotopes (when not using the
                database), defaults to []

            gammas_mT (list[float]): list of gyromagnetic ratios of
                the atoms and their isotopes (when not using the
                database), defaults to []

            hfcs (list[float]): list of hyperfine coupling constants
                of the atoms and their isotopes (when not using the
                database), defaults to []

        Returns: List generator.
        """
        self.radical = radical if radical else "N/A"
        if nuclei:
            if self._check_molecule_or_spin_db(radical, nuclei):
                self._init_from_molecule_db(radical, nuclei)
            else:
                self._init_from_spin_db(radical, nuclei, hfcs)
        else:
            self.multiplicities = multiplicities
            self.gammas_mT = gammas_mT
            self.hfcs = hfcs
        assert len(self.multiplicities) == self.num_particles
        assert len(self.gammas_mT) == self.num_particles
        assert len(self.hfcs) == self.num_particles

    def _check_molecule_or_spin_db(self, radical, nuclei):
        if radical in MOLECULE_DATA:
            self._check_nuclei(nuclei)
            return True
        else:
            if all(n in SPIN_DATA for n in nuclei):
                return False
            else:
                available = "\n".join(get_molecules().keys())
                raise ValueError(f"Available molecules below:\n{available}")

    def _check_nuclei(self, nuclei):
        molecule_data = MOLECULE_DATA[self.radical]["data"]
        for nucleus in nuclei:
            if nucleus not in molecule_data:
                keys = molecule_data.keys()
                hfcs = [molecule_data[k]["hfc"] for k in keys]
                pairs = sorted(
                    zip(keys, hfcs), key=lambda t: np.abs(t[1]), reverse=True
                )
                available = "\n".join([f"{k} (hfc = {h})" for k, h in pairs])
                raise ValueError(f"Available nuclei below.\n{available}")

    def _init_from_molecule_db(self, radical, nuclei):
        data = MOLECULE_DATA[radical]["data"]
        elem = [data[n]["element"] for n in nuclei]
        self.radical = radical
        self.gammas_mT = [gamma_mT(e) for e in elem]
        self.multiplicities = [multiplicity(e) for e in elem]
        self.hfcs = [data[n]["hfc"] for n in nuclei]

    def _init_from_spin_db(self, radical, nuclei, hfcs):
        self.multiplicities = [multiplicity(e) for e in nuclei]
        self.gammas_mT = [gamma_mT(e) for e in nuclei]
        self.hfcs = hfcs

    @property
    def num_particles(self):
        return len(self.multiplicities)

    def __repr__(self):
        return (
            f"Molecule: {self.radical}"
            # f"\n  Nuclei: {self.nuclei}"
            f"\n  HFCs: {self.hfcs}"
            f"\n  multiplicities: {self.multiplicities}"
            f"\n  gammas(mT): {self.gammas_mT}"
            f"\n  number of particles: {self.num_particles}"
            # f"\n  elements: {self.elements}"
        )


class KineticsRelaxationBase:
    def __init__(self, rate_constant: float):
        self.rate = rate_constant

    def adjust_hamiltonian(self, H: np.ndarray):
        return

    def adjust_product_probabilities(
        self,
        product_probabilities: np.ndarray,
        time: np.ndarray,
    ):
        return

    @property
    def rate_constant(self) -> float:
        return 1.0


class QuantumSimulation:
    """Quantum simulation class."""

    def __init__(self, molecules: list[Molecule], custom_gfactors=False):
        """Construct the object.

        Args:
            molecules (list[Molecule]): List of two `Molecule`
            objects.

        """
        assert len(molecules) == 2

        self.molecules = molecules
        self.coupling = [i for i, m in enumerate(molecules) for _ in m.gammas_mT]

        self.num_electrons = 2
        self.electrons = ["E"] * self.num_electrons
        self.hfcs = sum([m.hfcs for m in molecules], [])
        self.num_particles = self.num_electrons
        self.num_particles += sum([m.num_particles for m in molecules])
        self.multiplicities = list(map(multiplicity, self.electrons))
        self.multiplicities += sum([m.multiplicities for m in molecules], [])
        self.gammas_mT = self._get_electron_gammas_mT(custom_gfactors)
        self.gammas_mT += sum([m.gammas_mT for m in molecules], [])

    def _get_electron_gammas_mT(self, custom_gfactors):
        g = 2.0023  # free electron g-factor
        gfactor = [g, g]
        if custom_gfactors:
            # overwrite gfactor list TODO
            pass
        # muB = 9.274e-24
        # hbar = 1.05459e-34
        # return [gfactor[i] * muB / hbar / 1000 for i in range(self.num_electrons)]
        return [gamma_mT(e) * gfactor[i] / g for i, e in enumerate(self.electrons)]

    def spin_operator(self, idx: int, axis: str) -> np.ndarray:
        """Construct the spin operator for a particle.

        Construct the spin operator for the particle with index
        ``idx`` in the :class:`Quantum` system.

        Args:

            idx (int): Index of the particle.

            axis (str): Axis, i.e. ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            np.array: Spin operator for a particle in the
            :class:`Quantum` system simulated.

        """
        assert 0 <= idx and idx < len(self.multiplicities)
        assert axis in "xyz"

        sigma = pauli(self.multiplicities[idx])[axis]
        eye_before = np.eye(prod(m for m in self.multiplicities[:idx]))
        eye_after = np.eye(prod(m for m in self.multiplicities[idx + 1 :]))

        return np.kron(np.kron(eye_before, sigma), eye_after)

    def product_operator_axis(
        self, p1: int, p2: int, ax1: int, ax2: int = -1
    ) -> np.ndarray:
        if ax2 == -1:
            ax2 = ax1
        """Projection operator for a given axis."""
        return self.spin_operator(p1, ax1).dot(self.spin_operator(p2, ax2))

    def product_operator(self, idx1: int, idx2: int) -> np.ndarray:
        """Projection operator."""
        return sum([self.product_operator_axis(idx1, idx2, axis) for axis in "xyz"])

    def product_operator_3d(self, idx1: int, idx2: int, hfc: np.ndarray) -> np.ndarray:
        """Projection operator."""
        return sum(
            [
                hfc[i, j] * self.product_operator_axis(idx1, idx2, ax1, ax2)
                for i, ax1 in enumerate("xyz")
                for j, ax2 in enumerate("xyz")
            ]
        )

    def projection_operator(self, state: State):
        """Construct.

        .. todo::
            Write proper docs.
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

    def zeeman_hamiltonian(self, B0: float) -> np.ndarray:
        """Construct the Zeeman Hamiltonian.

        Construct the Zeeman Hamiltonian based on the external
        magnetic field `B`.

        Args:
            B0 (float): External magnetic field intensity (milli
            Tesla).

        Returns:
            np.array: The Zeeman Hamiltonian corresponding to the
            system described by the `Quantum` simulation object and
            the external magnetic field intensity `B`.

        """
        axis = "z"
        gammas = enumerate(self.gammas_mT)
        return -B0 * sum(g * self.spin_operator(i, axis) for i, g in gammas)

    def _HH_term(self, ei: int, ni: int) -> np.ndarray:
        """Construct a term of the Hyperfine Hamiltonian.

        .. todo::
            Write proper docs.
        """
        g = gamma_mT(self.electrons[ei])
        h = -g * self.hfcs[ni]
        effective_ni = self.num_electrons + ni
        if isinstance(h, np.ndarray):
            return self.product_operator_3d(ei, effective_ni, h)
        else:
            return h * self.product_operator(ei, effective_ni)

    def hyperfine_hamiltonian(self) -> np.ndarray:
        """Construct the Hyperfine Hamiltonian.

        Construct the Hyperfine Hamiltonian based on the magnetic
        field.

        Returns:
            np.array: The Hyperfine Hamiltonian corresponding to the
            system described by the `Quantum` simulation object.

        """
        return -sum([self._HH_term(ei, ni) for ni, ei in enumerate(self.coupling)])

    @staticmethod
    def exchange_interaction_solution(r: float) -> float:
        """Construct the exchange interaction constant in a solution.

        .. todo::
            Write proper docs.
        """
        J0rad = 1.7e17
        rj = 0.049e-9
        gamma = 1.76e8  # TODO
        J0 = J0rad / gamma / 10  # convert to mT?????????
        return J0 * np.exp(-r / rj)

    @staticmethod
    def exchange_interaction_protein(r: float) -> float:
        """Construct the exchange interaction constant in a protein.

        .. todo::
            Write proper docs.
        """
        beta = 1.4e10
        J0 = 8e13
        return J0 * np.exp(-beta * r)

    @staticmethod
    def exchange_interaction(r: float, model: str = "solution"):
        """Construct the exchange interaction constant in a solution.

        .. todo::
            Write proper docs.
        """
        methods = {
            "solution": __class__.exchange_int_solution,
            "protein": __class__.exchange_int_protein,
        }
        return methods[model](r)

    def exchange_hamiltonian(self, J: float) -> np.ndarray:
        """Construct the Exchange Hamiltonian.

        Construct the Exchange (J-coupling) Hamiltonian based on the
        coupling constant J between two electrons, which can be obtain
        from the radical pair separation `r` using `TODO` method.

        .. todo::
            Write proper docs.

        Returns:
            np.array: The Exchange (J-coupling) Hamiltonian
            corresponding to the system described by the `Quantum`
            simulation object and the coupling constant `J`.

        """
        Jcoupling = gamma_mT("E") * J
        SASB = self.product_operator(0, 1)
        return Jcoupling * (2 * SASB + 0.5 * np.eye(*SASB.shape))

    @staticmethod
    def dipolar_interaction(r: float) -> np.ndarray:
        """Construct the Dipolar interaction constant.

        Construct the Dipolar interaction based on the radius `r`.

        .. todo::
            Cite source.

        Returns:
            float: The dipolar coupling constant in milli Tesla (mT).

        """
        return -2.785 / r**3

    def dipolar_hamiltonian(self, D: float) -> np.ndarray:
        """Construct the Dipolar Hamiltonian.

        Construct the Dipolar Hamiltonian based on dipolar coupling
        constant `D` between two electrons.

        .. todo::
            Write proper docs.

        Returns:
            np.array: The Dipolar Hamiltonian corresponding to the
            system described by the `Quantum` simulation object and
            dipolar coupling constant `D`.

        """
        SASB = self.product_operator(0, 1)
        SAz = self.spin_operator(0, "z")
        SBz = self.spin_operator(1, "z")
        omega = (2 / 3) * gamma_mT("E") * D
        return omega * (3 * SAz * SBz - SASB)

    def total_hamiltonian(self, B: float, J: float, D: float) -> np.ndarray:
        """Construct the final (total) Hamiltonian.

        Construct the final (total)

        .. todo::
            Write proper docs.

        """
        return (
            self.zeeman_hamiltonian(B)
            + self.hyperfine_hamiltonian()
            + self.exchange_hamiltonian(J)
            + self.dipolar_hamiltonian(D)
        )

    def time_evolution(
        self, init_state: State, time: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Evolve the system through time."""
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
        obs = self.projection_operator(obs)
        return np.real(np.trace(obs @ rhos, axis1=-2, axis2=-1))

    @staticmethod
    def product_yield(probuct_probability, time, k):
        """Calculate the product yield and the product yield sum."""
        product_yield = sp.integrate.cumtrapz(probuct_probability, time, initial=0) * k
        product_yield_sum = np.max(product_yield, axis=-1)
        return product_yield, product_yield_sum

    @staticmethod
    def mary_lfe_hfe(
        init_state: State,
        B: np.ndarray,
        product_probability_seq: np.ndarray,
        dt: float,
        k: float,
    ) -> (np.array, np.array, np.array):
        """Calculate MARY, LFE, HFE."""
        MARY = np.sum(product_probability_seq, axis=1) * dt * k
        idx = int(len(MARY) / 2) if B[0] != 0 else 0
        minmax = max if init_state == State.SINGLET else min
        HFE = ((MARY[-1] - MARY[idx]) / MARY[idx]) * 100
        LFE = ((minmax(MARY) - MARY[idx]) / MARY[idx]) * 100
        MARY = ((MARY - MARY[idx]) / MARY[idx]) * 100
        return (MARY, LFE, HFE)

    def mary_loop(
        self,
        init_state: State,
        time: np.ndarray,
        B: np.ndarray,
        H_base: np.ndarray,
    ) -> np.ndarray:
        """Generate density matrices (rhos) for MARY.

        Args:
            init_state (State): initial state.
        Returns:
            List generator.

        .. todo::
            Write proper docs.
        """
        H_zee = self.zeeman_hamiltonian(1)
        rhos = np.zeros([len(B), len(time), *H_zee.shape], dtype=complex)
        for i, B0 in enumerate(B):
            H = H_base + B0 * H_zee
            rhos[i] = self.time_evolution(init_state, time, H)
        return rhos

    def MARY(
        self,
        init_state: State,
        obs_state: State,
        time: np.ndarray,
        B: np.ndarray,
        D: float,
        J: float,
        kinetics: list[KineticsRelaxationBase] = [],
        relaxations: list[KineticsRelaxationBase] = [],
    ) -> dict:
        dt = time[1] - time[0]
        H = self.total_hamiltonian(B=0, D=D, J=J)
        H = self.convert(H)
        for K in kinetics + relaxations:
            K.adjust_hamiltonian(H)
        rhos = self.mary_loop(init_state, time, B, H)
        product_probabilities = self.product_probability(obs_state, rhos)
        for K in kinetics:  # skip in liouville
            K.adjust_product_probabilities(product_probabilities, time)
        k = kinetics[0].rate_constant if kinetics else 1.0
        product_yields, product_yield_sums = self.product_yield(
            product_probabilities, time, k
        )
        MARY, LFE, HFE = self.mary_lfe_hfe(init_state, B, product_probabilities, dt, k)
        return dict(
            time=time,
            B=B,
            rhos=rhos,
            time_evolutions=product_probabilities,
            product_yields=product_yields,
            product_yield_sums=product_yield_sums,
            MARY=MARY,
            LFE=LFE,
            HFE=HFE,
        )


class HilbertSimulation(QuantumSimulation):
    @staticmethod
    def convert(H: np.ndarray) -> np.ndarray:
        return H

    def initial_density_matrix(self, state: State, H: np.ndarray) -> np.ndarray:
        """Create an initial desity matrix.

        Create an initial density matrix for time evolution of the
        spin Hamiltonian density matrix.

        Args:
            state (State): Spin state projection operator.
            H (np.array): Spin Hamiltonian in Hilbert space.

        Returns:
            np.array: A matrix in Hilbert space representing...

        """
        Pi = self.projection_operator(state)

        if np.array_equal(Pi, self.projection_operator(State.EQUILIBRIUM)):
            rho0eq = sp.linalg.expm(-1j * H * Pi)
            rho0 = rho0eq / np.trace(rho0eq)
        else:
            rho0 = Pi / np.trace(Pi)
        return rho0

    @staticmethod
    def unitary_propagator(H: np.ndarray, dt: float) -> np.ndarray:
        """Create unitary propagator (Hilbert space).

        Create unitary propagator matrices for time evolution of the
        spin Hamiltonian density matrix in Hilbert space.

        Arguments:
            H (np.array): Spin Hamiltonian in Hilbert space.
            dt (float): Time evolution timestep.

        Returns:
            np.array : Two matrices (a tensor) in either Hilbert.

        .. todo::
            https://docs.python.org/3/library/doctest.html

        Example:
            >> Up, Um = UnitaryPropagator(H, 3e-9, "Hilbert")
            >> UL = UnitaryPropagator(HL, 3e-9, "Liouville")

        """
        Up = sp.linalg.expm(1j * H * dt)
        Um = sp.linalg.expm(-1j * H * dt)
        return Up, Um

    def propagate(self, propagator: np.ndarray, rho: np.ndarray) -> np.ndarray:
        Up, Um = propagator
        return Um @ rho @ Up


class LiouvilleSimulation(QuantumSimulation):
    @staticmethod
    def convert(H: np.ndarray) -> np.ndarray:
        """Convert the Hamiltonian from Hilbert to Liouville space."""
        eye = np.eye(len(H))
        return 1j * (np.kron(H, eye) - np.kron(eye, H.T))

    def projection_operator(self, state: State) -> np.ndarray:
        return np.reshape(super().projection_operator(state), (-1, 1)).T

    def initial_density_matrix(self, state: State, H: np.ndarray) -> np.ndarray:
        """Create an initial density matrix for time evolution of the spin Hamiltonian density matrix.

        Arguments:
            state (State): a string = spin state projection operator
            spins: an integer = sum of the number of electrons and nuclei
            H: a matrix = spin Hamiltonian in Hilbert space

        Returns:
            A matrix in Liouville space

        """
        Pi = self.projection_operator(state).T
        if state == State.EQUILIBRIUM:
            rho0eq = sp.linalg.expm(-1j * H * Pi)
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
            H (np.array): Spin Hamiltonian in Hilbert or Liouville space
            dt (float): Time evolution timestep.
            space (str): Select the spin space.
        """
        return sp.linalg.expm(H * dt)

    def propagate(self, propagator: np.ndarray, rho: np.ndarray) -> np.ndarray:
        return propagator @ rho
