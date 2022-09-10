#!/usr/bin/env python

from math import prod
from typing import Iterable, Optional

import numpy as np
import scipy as sp

from .data import MOLECULE_DATA, gamma_mT, multiplicity
from .pauli_matrices import pauli


class Molecule:
    """Class representing a molecule in a simulation."""

    def __init__(
        self,
        radical: str = None,
        nuclei: list[str] = None,
        multiplicities: list[int] = None,
        gammas_mT: list[float] = None,
        hfcs: list[float] = None,
    ):
        """Construct a Molecule object."""
        self._set_radical_and_nuclei(radical, nuclei)
        self.multiplicities = self._cond_value(multiplicities, multiplicity)
        self.gammas_mT = self._cond_value(gammas_mT, gamma_mT)
        if nuclei is not None:
            self.num_particles = len(nuclei)
            self.elements = self._get_properties("element")
        else:
            self.num_particles = len(self.multiplicities)
            self.elements = self.num_particles * ["dummy"]
        self._set_hfcs(nuclei, hfcs)
        assert len(self.multiplicities) == self.num_particles
        assert len(self.gammas_mT) == self.num_particles
        assert len(self.hfcs) in {0, self.num_particles}

    def _set_hfcs(self, nuclei, hfcs):
        self.hfcs = []
        if hfcs is None:
            if nuclei is not None:
                self.hfcs = self._get_properties("hfc")
        else:
            self.hfcs = hfcs

    def _set_radical_and_nuclei(self, radical, nuclei):
        if radical is None:
            # Idea: nuclie = ["1H", "14N"] list of elements.
            assert nuclei is None
        else:
            assert radical in MOLECULE_DATA
            # todo cleanup
            self.data = MOLECULE_DATA[radical]["data"]
            for nucleus in nuclei:
                assert nucleus in self.data
        self.radical = radical
        self.nuclei = nuclei

    def _cond_value(self, value, func):
        if value is None:
            return list(map(func, self._get_properties("element")))
        return value

    def _get_properties(self, data: str) -> Iterable:
        """Construct a list for a given property.

        Args:
            data (str): the property.
        Returns:
            List generator.
        """
        return [] if self.nuclei is None else [self.data[n][data] for n in self.nuclei]

    def _get_property(self, idx: int, key: str):
        """Get data of a nucleus.

        Utility for used only for testing currently.

        .. todo::
            Make tests better and probably remove this functions.

        """
        return self.data[self.nuclei[idx]][key]


class Quantum:
    """Quantum simulation class."""

    def __init__(self, molecules: list[Molecule]):
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
        self.gammas_mT = list(map(gamma_mT, self.electrons))
        self.gammas_mT += sum([m.gammas_mT for m in molecules], [])

    def spinop(self, idx: int, axis: str) -> np.array:
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

    def prodop_axis(self, p1: int, p2: int, axis: int) -> np.array:
        """Projection operator for a given axis."""
        return self.spinop(p1, axis).dot(self.spinop(p2, axis))

    def prodop(self, particle1: int, particle2: int) -> np.array:
        """Projection operator."""
        return sum([self.prodop_axis(particle1, particle2, axis) for axis in "xyz"])

    def projop(self, state: str):
        """Construct.

        .. todo::
            Write proper docs.
        """
        # Spin operators
        SAx, SAy, SAz = [self.spinop(0, ax) for ax in "xyz"]
        SBx, SBy, SBz = [self.spinop(1, ax) for ax in "xyz"]

        # Product operators
        SASB = self.prodop(0, 1)

        eye = np.eye(len(SASB))

        # Projection operators
        # todo change p/m to +/-
        match state:
            case "S":
                return (1 / 4) * eye - SASB
            case "T":
                return (3 / 4) * eye + SASB
            case "Tp":
                return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz)
            case "Tm":
                return (2 * SAz**2 - SAz) * (2 * SBz**2 - SBz)
            case "T0":
                return (1 / 4) * eye + SAx @ SBx + SAy @ SBy - SAz @ SBz
            case "Tpm":
                return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz) + (
                    2 * SAz**2 - SAz
                ) * (2 * SBz**2 - SBz)
            case "Eq":
                return 1.05459e-34 / (1.38e-23 * 298)

    def zeeman_hamiltonian(self, B0: float) -> np.array:
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
        return -B0 * sum(g * self.spinop(i, axis) for i, g in gammas)

    def _HH_term(self, ei: int, ni: int) -> np.array:
        """Construct a term of the Hyperfine Hamiltonian.

        .. todo::
            Write proper docs.
        """
        g = gamma_mT(self.electrons[ei])
        h = self.hfcs[ni]
        return -g * h * self.prodop(ei, self.num_electrons + ni)

    def hyperfine_hamiltonian(self) -> np.array:
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
        gamma = 1.76e8
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

    def exchange_hamiltonian(self, J: float) -> np.array:
        """Construct the Exchange Hamiltonian.

        Construct the Exchange (J-coupling) Hamiltonian based on the
        coupling constant J between two electrons, which can be obtain
        from the radical pair separation `r` using :py:`TODO` method.

        .. todo::
            Write proper docs.

        Returns:
            np.array: The Exchange (J-coupling) Hamiltonian
            corresponding to the system described by the `Quantum`
            simulation object and the coupling constant `J`.

        """
        Jcoupling = gamma_mT("E") * J
        SASB = self.prodop(0, 1)
        return Jcoupling * (2 * SASB + 0.5 * np.eye(*SASB.shape))

    @staticmethod
    def dipolar_interaction(r: float) -> np.array:
        """Construct the Dipolar interaction constant.

        Construct the Dipolar interaction based on the radius `r`.

        .. todo::
            Cite source.

        Returns:
            float: The dipolar coupling constant in milli Tesla (mT).

        """
        return -2.785 / r**3

    def dipolar_hamiltonian(self, D: float) -> np.array:
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
        SASB = self.prodop(0, 1)
        SAz = self.spinop(0, "z")
        SBz = self.spinop(1, "z")
        omega = (2 / 3) * gamma_mT("E") * D
        return omega * (3 * SAz * SBz - SASB)

    def total_hamiltonian(self, B: float, J: float, D: float) -> np.array:
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

    def hilbert_initial(self, state: str, H: np.array) -> np.array:
        """Create an initial desity matrix.

        Create an initial density matrix for time evolution of the
        spin Hamiltonian density matrix.

        Args:
            state (str): Spin state projection operator.
            H (np.array): Spin Hamiltonian in Hilbert space.

        Returns:
            np.array: A matrix in Hilbert space representing...

        """
        Pi = self.projop(state)

        if np.array_equal(Pi, self.projop("Eq")):
            rho0eq = sp.linalg.expm(-1j * H * Pi)
            rho0 = rho0eq / np.trace(rho0eq)
        else:
            rho0 = Pi / np.trace(Pi)
        return rho0

    @staticmethod
    def hilbert_unitary_propagator(H: np.array, dt: float) -> np.array:
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
            >>> Up, Um = UnitaryPropagator(H, 3e-9, "Hilbert")
            >>> UL = UnitaryPropagator(HL, 3e-9, "Liouville")

        """
        Up = sp.linalg.expm(1j * H * dt)
        Um = sp.linalg.expm(-1j * H * dt)
        return Up, Um

    def hilbert_time_evolution(
        self, init_state: str, time: np.array, H: np.array
    ) -> np.array:
        """Evolve the system through time."""
        dt = time[1] - time[0]
        Up, Um = self.hilbert_unitary_propagator(H, dt)

        rhos = np.zeros([len(time), *H.shape], dtype=complex)
        rhos[0] = self.hilbert_initial(init_state, H)
        for t in range(1, len(time)):
            rhos[t] = Um @ rhos[t - 1] @ Up
        return rhos

    def product_probability(self, obs: np.array, rhos: np.array) -> np.array:
        """Calculate the probability of the observable from the densities."""
        return np.real(np.trace(obs @ rhos, axis1=-2, axis2=-1))

    @staticmethod
    def product_yield(prob, time, k):
        """Calculate the product yield and the product yield sum."""
        product_yield = sp.integrate.cumtrapz(prob, time, initial=0) * k
        product_yield_sum = np.max(product_yield)
        return product_yield, product_yield_sum

    def kinetics_exponential(self, k: float, time: np.array) -> np.array:
        """Return exponential kinetics."""
        return np.exp(-k * time)

    def liouville_projop(self, state: str) -> np.array:
        return np.reshape(self.projop(state), (-1, 1))

    def liouville_initial(self, state: str, H: np.array) -> np.array:
        """Create an initial density matrix for time evolution of the spin Hamiltonian density matrix.

        Arguments:
            state: a string = spin state projection operator
            spins: an integer = sum of the number of electrons and nuclei
            H: a matrix = spin Hamiltonian in Hilbert space

        Returns:
            A matrix in Liouville space

        Example:
            rho0 = Liouville_initial("S", 3, H)
        """
        Pi = self.liouville_projop(state)
        if state == "Eq":
            rho0eq = sp.linalg.expm(-1j * H * Pi)
            rho0 = rho0eq / np.trace(rho0eq)
            rho0 = np.reshape(rho0, (len(H) ** 2, 1))
        else:
            rho0 = Pi / np.vdot(Pi, Pi)
        return rho0

    @staticmethod
    def liouville_unitary_propagator(H, dt):
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

    @staticmethod
    def hilbert_to_liouville(H: np.array) -> np.array:
        """Convert the Hamiltonian from Hilbert to Liouville space."""
        eye = np.eye(len(H))
        return 1j * (np.kron(H, eye) - np.kron(eye, H.T))

    def liouville_time_evolution(
        self, init_state: str, time: np.array, H: np.array
    ) -> np.array:
        """Generate the density time evolution."""
        dt = time[1] - time[0]
        HL = self.hilbert_to_liouville(H)
        rho0 = self.liouville_initial(init_state, HL)
        rhos = np.zeros([len(time), *rho0.shape], dtype=complex)
        UL = self.liouville_unitary_propagator(HL, dt)
        rhos[0] = rho0
        for t in range(1, len(time)):
            rhos[t] = UL @ rhos[t - 1]
        return rhos

    def mary(self, init_state, obs_state, time, k, B, H_base, space="Hilbert"):
        dt = time[1] - time[0]
        MFE = np.zeros((len(B), len(time)))
        H_zee = self.zeeman_hamiltonian(1)
        obs = self.projop(obs_state)
        rhos = np.zeros([len(B), len(time), *H_zee.shape], dtype=complex)
        for i, B0 in enumerate(B):
            H = H_base + B0 * H_zee
            rhos[i] = self.hilbert_time_evolution(init_state, time, H)
            prob = self.product_probability(obs, rhos[i])
            Kexp = self.kinetics_exponential(k, time)
            MFE[i] = prob * Kexp

        MARY = np.sum(MFE, axis=1) * dt * k

        idx = int(len(MARY) / 2) if B[0] != 0 else 0
        minmax = max if init_state == "S" else min
        HFE = ((MARY[-1] - MARY[idx]) / MARY[idx]) * 100
        LFE = ((minmax(MARY) - MARY[idx]) / MARY[idx]) * 100
        MARY = ((MARY - MARY[idx]) / MARY[idx]) * 100

        return {
            "MFE": MFE,
            "HFE": HFE,
            "LFE": LFE,
            "MARY": MARY,
            "rhos": rhos,
        }

    # sim.mary(time=np.linspace(), magnetic_field=np.linspace())
    # sim.angle(time=np.linspace(), theta=np.linspace(), phi=np.linspace())

    # def MARY(spins, initial, observable, t_max, t_stepsize, k, B, Hplot, space="Hilbert"):

    #     timing = np.arange(0, t_max, t_stepsize)
    #     MFE = np.zeros((len(B), len(timing)))

    #     for i, B0 in enumerate(B):
    #         time, MFE[i, :], productyield, ProductYieldSum, rhot = TimeEvolution(
    #             spins, initial, observable, t_max, t_stepsize, k, B0, Hplot, space=space
    #         )

    #     raw = MFE
    #     dt = t_stepsize
    #     MARY = np.sum(raw, axis=1) * dt * k
    #     MARY = ((MARY - MARY[0]) / MARY[0]) * 100
    #     return [time, MFE, MARY, productyield, ProductYieldSum, rhot]
