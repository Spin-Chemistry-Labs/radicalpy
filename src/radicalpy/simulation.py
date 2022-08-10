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
        hfcs: list[float] = None,
        multis: list[int] = None,
        gammas_mT: list[float] = None,
    ):
        """Construct a Molecule object."""
        self._check_input(radical, nuclei)
        self.radical = radical
        self.nuclei = nuclei
        if nuclei is not None:
            self.num_particles = len(nuclei)
            self.elements = self._get_properties("element")
        else:
            self.num_particles = len(hfcs)
            self.elements = self.num_particles * ["dummy"]
        self.hfcs = self._cond_value(hfcs, "hfc")
        self.multis = self._cond_value(multis, multiplicity)
        self.gammas_mT = self._cond_value(gammas_mT, gamma_mT)
        assert len(self.hfcs) == self.num_particles
        assert len(self.multis) == self.num_particles
        assert len(self.gammas_mT) == self.num_particles

    def _check_input(self, radical, nuclei):
        if radical is None:
            assert nuclei is None
        else:
            assert radical in MOLECULE_DATA
            self.data = MOLECULE_DATA[radical]["data"]
            for nucleus in nuclei:
                assert nucleus in self.data

    def _cond_value(self, value, func):
        if value is None:
            if isinstance(func, str):
                return self._get_properties(func)
            return list(map(func, self.elements))
        return value

    def _get_properties(self, data: str) -> Iterable:
        """Construct a list for a given property.

        Args:
            data (str): the property.
        Returns:
            List generator.
        """
        return [self.data[n][data] for n in self.nuclei]

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
        self.coupling = [i for i, m in enumerate(molecules) for p in m.elements]

        self.num_electrons = 2
        self.electrons = ["E"] * self.num_electrons
        self.nuclei = sum([m.elements for m in molecules], [])
        self.hfcs = sum([m.hfcs for m in molecules], [])
        self.num_particles = self.num_electrons
        self.num_particles += sum([m.num_particles for m in molecules])
        self.multiplicities = list(map(multiplicity, self.electrons))
        self.multiplicities += sum([m.multis for m in molecules], [])
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
        op1 = self.spinop(p1, axis)
        op2 = self.spinop(p2, axis)
        return op1.dot(op2)

    def prodop(self, particle1: int, particle2: int) -> np.array:
        """Projection operator."""
        return sum([self.prodop_axis(particle1, particle2, axis) for axis in "xyz"])

    def total_hamiltonian(self, B: float) -> np.array:
        return self.HZ(B) + self.HH()

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

    def HH_term(self, ei, ni) -> np.array:
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
        return -sum([self.HH_term(ei, ni) for ni, ei in enumerate(self.coupling)])

    def HJ():
        pass

    def HD():
        pass
