#!/usr/bin/env python

import enum
from math import prod
from typing import Iterable, Optional

import numpy as np
import scipy as sp

from . import utils
from .data import (MOLECULE_DATA, SPIN_DATA, gamma_mT, get_molecules,
                   multiplicity, pauli)


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


class Molecule:
    """Representation of a molecule for the simulation.

    Args:
        radical (str): the name of the `Molecule`, defaults to `""`

        nuclei (list[str]): list of atoms from the molecule (or from
            the database), defaults to `[]`

        multiplicities (list[int]): list of multiplicities of the
            atoms and their isotopes (when not using the database),
            defaults to `[]`

        gammas_mT (list[float]): list of gyromagnetic ratios of the
            atoms and their isotopes (when not using the database),
            defaults to `[]`

        hfcs (list[float]): list of hyperfine coupling constants of
            the atoms and their isotopes (when not using the
            database), defaults to `[]`

    A molecule is represented by hyperfine coupling constants, spin
    multiplicities and gyromagnetic ratios (gammas, specified in mT)
    of its nuclei.  When using the database, one needs to specify the
    name of the molecule and the list of its nuclei.

    >>> Molecule(radical="adenine_cation",
    ...          nuclei=["N6-H1", "N6-H2"])
    Molecule: adenine_cation
      HFCs: [-0.63, -0.66]
      Multiplicities: [3, 3]
      Magnetogyric ratios (mT): [19337.792, 19337.792]
      Number of particles: 2


    If the wrong molecule name is given, the error helps you find the
    valid options.

    >>> Molecule("foobar", ["H1"])
    Traceback (most recent call last):
    ...
    ValueError: Available molecules below:
    2_6_aqds
    adenine_cation
    flavin_anion
    flavin_neutral
    tryptophan_cation
    tyrosine_neutral

    Similarly, giving a list of incorrect atom names will also result
    in a helpful error message listing the available atoms.

    >>> Molecule("tryptophan_cation", ["buz"])
    Traceback (most recent call last):
    ...
    ValueError: Available nuclei below.
    Hbeta1 (hfc = 1.6045)
    H1 (hfc = -0.5983)
    H4 (hfc = -0.4879)
    H7 (hfc = -0.3634)
    N1 (hfc = 0.32156666666666667)
    H2 (hfc = -0.278)
    N* (hfc = 0.1465)
    Halpha (hfc = -0.09306666666666667)
    Hbeta2 (hfc = 0.04566666666666666)
    H5 (hfc = -0.04)
    H6 (hfc = -0.032133333333333326)

    >>> Molecule("adenine_cation", ["buz"])
    Traceback (most recent call last):
    ...
    ValueError: Available nuclei below.
    N6-H2 (hfc = -0.66)
    N6-H1 (hfc = -0.63)
    C8-H (hfc = -0.55)

    One can also specify a list of custom hyperfine coupling constants
    along with a list of their respective isotope names.

    >>> Molecule(nuclei=["1H", "14N"], hfcs=[0.41, 1.82])
    Molecule: N/A
      HFCs: [0.41, 1.82]
      Multiplicities: [2, 3]
      Magnetogyric ratios (mT): [267522.18744, 19337.792]
      Number of particles: 2

    Same as above, but with an informative molecule name (doesn't
    affect behaviour):

    >>> Molecule("isotopes", nuclei=["15N", "15N"], hfcs=[0.3, 1.7])
    Molecule: isotopes
      HFCs: [0.3, 1.7]
      Multiplicities: [2, 2]
      Magnetogyric ratios (mT): [-27126.180399999997, -27126.180399999997]
      Number of particles: 2

    A molecule with no HFCs, for one proton radical pair simulations
    (for simple simulations -- often with *fantastic* low-field
    effects):

    >>> Molecule("kryptonite")
    Molecule: kryptonite
      HFCs: []
      Multiplicities: []
      Magnetogyric ratios (mT): []
      Number of particles: 0

    Manual input for all relevant values (multiplicities, gammas,
    HFCs):

    >>> Molecule(multiplicities=[2, 2, 3],
    ...          gammas_mT=[267522.18744, 267522.18744, 19337.792],
    ...          hfcs=[0.42, 1.01, 1.33])
    Molecule: N/A
      HFCs: [0.42, 1.01, 1.33]
      Multiplicities: [2, 2, 3]
      Magnetogyric ratios (mT): [267522.18744, 267522.18744, 19337.792]
      Number of particles: 3

    Same as above with an informative molecule name:

    >>> Molecule("my_flavin", multiplicities=[2], gammas_mT=[267522.18744], hfcs=[0.5])
    Molecule: my_flavin
      HFCs: [0.5]
      Multiplicities: [2]
      Magnetogyric ratios (mT): [267522.18744]
      Number of particles: 1

    """

    def __init__(
        self,
        radical: str = "",
        nuclei: list[str] = [],
        multiplicities: list[int] = [],
        gammas_mT: list[float] = [],
        hfcs: list[float] = [],
    ):
        self.radical = radical if radical else "N/A"
        self.nuclei = nuclei
        self.custom_molecule = True
        if nuclei:
            if self._check_molecule_or_spin_db(radical, nuclei):
                self._init_from_molecule_db(radical, nuclei)
            else:
                self._init_from_spin_db(radical, nuclei, hfcs)
        else:
            if self._check_molecule_or_spin_db(radical, nuclei):
                self._init_from_molecule_db(radical, nuclei)
            else:
                self.multiplicities = multiplicities
                self.gammas_mT = gammas_mT
                self.hfcs = hfcs
        if self.hfcs and isinstance(self.hfcs[0], list):
            self.hfcs = [np.array(h) for h in self.hfcs]
        assert len(self.multiplicities) == self.num_particles
        assert len(self.gammas_mT) == self.num_particles
        assert len(self.hfcs) == self.num_particles

    def __repr__(self) -> str:
        """Pretty print the molecule.

        Returns:
            str: Representation of a molecule.
        """
        return (
            f"Molecule: {self.radical}"
            # f"\n  Nuclei: {self.nuclei}"
            f"\n  HFCs: {self.hfcs}"
            f"\n  Multiplicities: {self.multiplicities}"
            f"\n  Magnetogyric ratios (mT): {self.gammas_mT}"
            f"\n  Number of particles: {self.num_particles}"
            # f"\n  elements: {self.elements}"
        )

    def _check_molecule_or_spin_db(self, radical, nuclei):
        if radical in MOLECULE_DATA:
            self._check_nuclei(nuclei)
            return True
        else:
            # TODO: needs to fail with nuclei == [] + wrong molecule
            # name
            if all(n in SPIN_DATA for n in nuclei):
                return False
            else:
                available = "\n".join(get_molecules().keys())
                raise ValueError(f"Available molecules below:\n{available}")

    def _check_nuclei(self, nuclei: list[str]) -> None:
        molecule_data = MOLECULE_DATA[self.radical]["data"]
        for nucleus in nuclei:
            if nucleus not in molecule_data:
                keys = molecule_data.keys()
                hfcs = [molecule_data[k]["hfc"] for k in keys]
                hfcs = [
                    utils.isotropic(np.array(h)) if isinstance(h, list) else h
                    for h in hfcs
                ]
                pairs = sorted(
                    zip(keys, hfcs), key=lambda t: np.abs(t[1]), reverse=True
                )
                available = "\n".join([f"{k} (hfc = {h})" for k, h in pairs])
                raise ValueError(f"Available nuclei below.\n{available}")

    def _init_from_molecule_db(self, radical: str, nuclei: list[str]) -> None:
        data = MOLECULE_DATA[radical]["data"]
        elem = [data[n]["element"] for n in nuclei]
        self.radical = radical
        self.gammas_mT = [gamma_mT(e) for e in elem]
        self.multiplicities = [multiplicity(e) for e in elem]
        self.hfcs = [data[n]["hfc"] for n in nuclei]
        self.custom_molecule = False

    def _init_from_spin_db(
        self, radical: str, nuclei: list[str], hfcs: list[float]
    ) -> None:
        self.multiplicities = [multiplicity(e) for e in nuclei]
        self.gammas_mT = [gamma_mT(e) for e in nuclei]
        self.hfcs = hfcs

    @property
    def effective_hyperfine(self) -> float:
        if self.custom_molecule:
            multiplicities = self.multiplicities
            hfcs = self.hfcs
        else:
            # TODO: this can fail with wrong molecule name
            data = MOLECULE_DATA[self.radical]["data"]
            nuclei = list(data.keys())
            elem = [data[n]["element"] for n in nuclei]
            multiplicities = [multiplicity(e) for e in elem]
            hfcs = [data[n]["hfc"] for n in nuclei]

        # spin quantum number
        s = np.array(list(map(utils.spin_quantum_number, multiplicities)))
        hfcs = [utils.isotropic(h) if isinstance(h, list) else h for h in hfcs]
        hfcs = np.array(hfcs)
        return np.sqrt((4 / 3) * sum((hfcs**2 * s) * (s + 1)))

    @property
    def num_particles(self) -> int:
        """Return the number of isotopes in the molecule."""
        return len(self.multiplicities)


class KineticsRelaxationBase:
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
        return self.rate

    def _name(self):
        return f"Kinetics: {type(self).__name__}"

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

    >>> HilbertSimulation([Molecule("flavin_anion", ["N5"]),
    ...                    Molecule("tryptophan_cation", ["Hbeta1", "H1"])])
    Number of electrons: 2
    Number of nuclei: 3
    Number of particles: 5
    Multiplicities: [2, 2, 3, 2, 2]
    Magnetogyric ratios (mT): [-176085963.023, -176085963.023, 19337.792, 267522.18744, 267522.18744]
    Nuclei: ['N5', 'Hbeta1', 'H1']
    Couplings: [0, 1, 1]
    HFCs (mT): [array([[-0.06819637,  0.01570029,  0.08701531],
           [ 0.01570029, -0.03652102,  0.27142597],
           [ 0.08701531,  0.27142597,  1.64713923]]), array([[ 1.5808, -0.0453, -0.0506],
           [-0.0453,  1.5575,  0.0988],
           [-0.0506,  0.0988,  1.6752]]), array([[-0.992 , -0.2091, -0.2003],
           [-0.2091, -0.2631,  0.2803],
           [-0.2003,  0.2803, -0.5398]])]
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
        return sum([[i] * m.num_particles for i, m in enumerate(self.molecules)], [])

    @property
    def electrons(self):
        return ["E"] * self.num_electrons

    @property
    def hfcs(self):
        return sum([m.hfcs for m in self.molecules], [])

    @property
    def num_electrons(self):
        return len(self.molecules)

    @property
    def num_nuclei(self):
        return sum([m.num_particles for m in self.molecules])

    @property
    def num_particles(self):
        return self.num_electrons + self.num_nuclei

    @property
    def electron_multiplicities(self):
        return list(map(multiplicity, self.electrons))

    @property
    def nuclei_multiplicities(self):
        return sum([m.multiplicities for m in self.molecules], [])

    @property
    def multiplicities(self):
        return self.electron_multiplicities + self.nuclei_multiplicities

    @property
    def electron_gammas_mT(self):
        g = 2.0023  # free electron g-factor
        gfactor = [g, g]
        if self.custom_gfactors:
            # overwrite gfactor list TODO
            pass
        # muB = 9.274e-24
        # hbar = 1.05459e-34
        # return [gfactor[i] * muB / hbar / 1000 for i in range(self.num_electrons)]
        return [gamma_mT(e) * gfactor[i] / g for i, e in enumerate(self.electrons)]

    @property
    def nuclei_gammas_mT(self):
        return sum([m.gammas_mT for m in self.molecules], [])

    @property
    def gammas_mT(self):
        return self.electron_gammas_mT + self.nuclei_gammas_mT

    def __repr__(self) -> str:
        return "\n".join(
            [
                # "Simulation summary:",
                f"Number of electrons: {self.num_electrons}",
                f"Number of nuclei: {len(self.hfcs)}",
                f"Number of particles: {self.num_particles}",
                f"Multiplicities: {self.multiplicities}",
                f"Magnetogyric ratios (mT): {self.gammas_mT}",
                f"Nuclei: {sum([m.nuclei for m in self.molecules], [])}",
                f"Couplings: {self.coupling}",
                f"HFCs (mT): {self.hfcs}",
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

        C = np.kron(ST, np.eye(prod(self.nuclei_multiplicities)))
        return C @ M @ C.T

    def spin_operator(self, idx: int, axis: str) -> np.ndarray:
        """Construct the spin operator for a particle.

        Args:

            idx (int): Index of the particle.

            axis (str): Axis, i.e. ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            np.ndarray: Spin operator for a particle in the
            `HilbertSimulation` system simulated.

        Construct the spin operator for the particle with index
        `idx` in the `HilbertSimulation`.
        """
        assert 0 <= idx and idx < len(self.multiplicities)
        assert axis in "xyz"

        sigma = pauli(self.multiplicities[idx])[axis]
        eye_before = np.eye(prod(m for m in self.multiplicities[:idx]))
        eye_after = np.eye(prod(m for m in self.multiplicities[idx + 1 :]))

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
        gammas = enumerate(self.gammas_mT)
        return -B0 * sum(g * self.spin_operator(i, axis) for i, g in gammas)

    def zeeman_hamiltonian_3d(
        self, B0: float, theta: float = 0, phi: float = 0
    ) -> np.ndarray:
        particles = np.array(
            [
                [self.spin_operator(idx, axis) for axis in "xyz"]
                for idx in range(self.num_particles)
            ]
        )
        rotation = utils.spherical_to_cartesian(theta, phi)
        omega = B0 * self.gammas_mT[0]
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
            for h in self.hfcs:
                if not isinstance(h, np.ndarray) and h.shape == (3, 3):
                    raise ValueError(
                        "Not all molecules have 3x3 HFC tensors! Please use `hfc_anisotropy=False`"
                    )

        if hfc_anisotropy:
            prodop = self.product_operator_3d
            hfcs = self.hfcs
        else:
            prodop = self.product_operator
            hfcs = [
                utils.isotropic(h) if isinstance(h, np.ndarray) else h
                for h in self.hfcs
            ]
        return sum(
            [
                self.gammas_mT[ei] * prodop(ei, self.num_electrons + ni, hfcs[ni])
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
        Jcoupling = self.gammas_mT[0] * J
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
        omega = (2 / 3) * self.gammas_mT[0] * D
        return omega * (3 * SAz * SBz - SASB)

    def dipolar_hamiltonian_3d(self, dipolar_tensor: np.ndarray) -> np.ndarray:
        ne = self.num_electrons
        return -sum(
            [
                -self.gammas_mT[0]
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
        if obs == State.EQUILIBRIUM:
            raise ValueError("Observable state should not be EQUILIBRIUM")
        obs = self.observable_projection_operator(obs)
        return np.real(np.trace(obs @ rhos, axis1=-2, axis2=-1))

    @staticmethod
    def product_yield(probuct_probability, time, k):
        """Calculate the product yield and the product yield sum."""
        product_yield = sp.integrate.cumtrapz(probuct_probability, time, initial=0) * k
        product_yield_sum = np.trapz(probuct_probability, dx=time[1]) * k
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
        for i, B0 in enumerate(B):
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
        rhos = utils.square_vectors(rhos)

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
        kinetics: list[KineticsRelaxationBase] = [],
        relaxations: list[KineticsRelaxationBase] = [],
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
        rhos = utils.square_vectors(rhos)

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

    def liouville_projection_operator(self, state: State) -> np.ndarray:
        return np.reshape(self.projection_operator(state), (-1, 1))

    def observable_projection_operator(self, state: State) -> np.ndarray:
        Q = self.liouville_projection_operator(state)
        return Q.T

    def initial_density_matrix(self, state: State, H: np.ndarray) -> np.ndarray:
        """Create an initial density matrix for time evolution of the spin Hamiltonian density matrix.

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


class LiouvilleKineticsRelaxationBase(KineticsRelaxationBase):
    def adjust_hamiltonian(self, H: np.ndarray):
        H -= self.subH
