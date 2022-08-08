#!/usr/bin/env python

from math import prod
from typing import Iterable

import numpy as np

from .data import MOLECULE_DATA, gamma_mT, multiplicity
from .pauli_matrices import pauli


class Molecule:
    """Class representing a molecule in a simulation."""

    def __init__(self, rad: str, nuclei: list[str]):
        """Construct a Molecule object."""
        assert rad in MOLECULE_DATA.keys()
        rad_data = MOLECULE_DATA[rad]["data"]
        for nucleus in nuclei:
            assert nucleus in rad_data.keys()
        self.rad = rad
        self.nuclei = nuclei
        self.data = MOLECULE_DATA[rad]["data"]

    def data_generator(self, data: str) -> Iterable:
        """Construct a generator for a given property.

        Args:
            data (str): the property.
        Returns:
            List generator.
        """
        for nucleus in self.nuclei:
            yield self.data[nucleus][data]

    def elements(self) -> Iterable[str]:
        """Construct a generator for the elements of different nuclei.

        Returns:
            Return a generator of element names.

        """
        return self.data_generator("element")

    def hfcs(self) -> Iterable[float]:
        """Construct a generator for the HFCs of different nuclei.

        Returns:
            Return a generator of HFC values.

        """
        return self.data_generator("hfc")

    def get_data(self, idx: int, key: str):
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
        self.coupling = [i for i, m in enumerate(molecules) for p in m.elements()]

        self.nelectrons = 2
        self.electrons = ["E"] * self.nelectrons
        self.nuclei = sum([list(m.elements()) for m in molecules], [])
        self.hfcs = sum([list(m.hfcs()) for m in molecules], [])
        self.particles = self.electrons + self.nuclei
        self.multiplicities = list(map(multiplicity, self.particles))
        self.gammas_mT = list(map(gamma_mT, self.particles))
        print(self.total_hamiltonian(0.5))

    @property
    def num_particles(self) -> int:
        """Return the number of paricles."""
        return len(self.particles)

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

    def HH(self) -> np.array:
        return sum(
            [
                gamma_mT(self.electrons[ei]) * self.hfcs[ni] * self.prodop(ei, ni)
                for ni, ei in enumerate(self.coupling)
            ]
        )
