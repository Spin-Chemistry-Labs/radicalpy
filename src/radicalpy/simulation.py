#!/usr/bin/env python

from math import prod
from typing import Iterable, Optional

import numpy as np

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
        """Make a spin operator."""
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

    def total_hamiltonian(self, B: float, J: float, D: float) -> np.array:
        """Calculate the final (total) Hamiltonian.

        .. todo::
            Write proper docs.

        """
        return self.HZ(B) + self.HH() + self.HE(J) + self.HD(D)

    def HZ(self, B: float) -> np.array:
        """Calculate the Zeeman Hamiltonian.

        Calculate the Zeeman Hamiltonian based on the magnetic field.

        Args:
            B (float): Magnetic field intensity (milli Tesla).

        Returns:
            np.array: The Zeeman Hamiltonian corresponding to the
            system described by the `Quantum` simulation object and
            the magnetic intensity `B`.

        """
        axis = "z"
        gammas = enumerate(self.gammas_mT)
        return -sum(B * g * self.spinop(i, axis) for i, g in gammas)

    def _HH_term(self, ei: int, ni: int) -> np.array:
        """Calculate a term of the Hyperfine Hamiltonian.

        .. todo::
            Write proper docs.
        """
        g = gamma_mT(self.electrons[ei])
        h = self.hfcs[ni]
        return -g * h * self.prodop(ei, self.num_electrons + ni)

    def HH(self) -> np.array:
        """Calculate the Hyperfine Hamiltonian.

        Calculate the Hyperfine Hamiltonian based on the magnetic
        field.

        Returns:
            np.array: The Hyperfine Hamiltonian corresponding to the
            system described by the `Quantum` simulation object and
            the magnetic intensity `B`.

        """
        return -sum([self._HH_term(ei, ni) for ni, ei in enumerate(self.coupling)])

    def HE(self, J: float) -> np.array:
        """Calculate the Exchange Hamiltonian.

        Calculate the Exchange (J-coupling) Hamiltonian based on the
        ...

        .. todo::
            Write proper docs.

        Returns:
            np.array: The Exchange (J-coupling) Hamiltonian
            corresponding to the system described by the `Quantum`
            simulation object and the magnetic intensity `B`.

        """
        Jcoupling = gamma_mT("E") * J
        SASB = self.prodop(0, 1)
        return Jcoupling * (2 * SASB + 0.5 * np.eye(*SASB.shape))

    def HD(self, D: float) -> np.array:
        """Calculate the Dipolar Hamiltonian.

        Calculate the Dipolar Hamiltonian based on ...

        .. todo::
            Write proper docs.

        Returns:
            np.array: The Dipolar Hamiltonian corresponding to the
            system described by the `Quantum` simulation object and
            the magnetic intensity `B`.

        """
        SASB = self.prodop(0, 1)
        SAz = self.spinop(0, "z")
        SBz = self.spinop(1, "z")
        omega = (2 / 3) * gamma_mT("E") * D
        return omega * (3 * SAz * SBz - SASB)

    def dipolar_interaction(self, r: float) -> np.array:
        """Calculate the Dipolar interaction constant.

        Calculate the Dipolar interaction based on the radius `r`.

        .. todo::
            Write proper docs.
        """
        return -2.785 / r**3

    @staticmethod
    def exchange_int_solution(r: float) -> float:
        """Calculate the exchange interaction constant in a solution.

        .. todo::
            Write proper docs.
        """
        J0rad = 1.7e17
        rj = 0.049e-9
        gamma = 1.76e8
        J0 = J0rad / gamma / 10  # convert to mT?????????
        return J0 * np.exp(-r / rj)

    @staticmethod
    def exchange_int_protein(r: float) -> float:
        """Calculate the exchange interaction constant in a protein.

        .. todo::
            Write proper docs.
        """
        beta = 1.4e10
        J0 = 8e13
        return J0 * np.exp(-beta * r)

    @staticmethod
    def exchange_interaction(r: float, model: str = "solution"):
        """Calculate the exchange interaction constant in a solution.

        .. todo::
            Write proper docs.
        """
        methods = {
            "solution": __class__.exchange_int_solution,
            "protein": __class__.exchange_int_protein,
        }
        return methods[model](r)
