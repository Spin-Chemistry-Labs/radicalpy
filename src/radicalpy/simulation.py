#!/usr/bin/env python

from math import prod
from typing import Iterable

import numpy as np

from .data import MOLECULE_DATA, gamma, multiplicity
from .pauli_matrices import pauli


class Molecule:
    """Class representing a molecule in a simulation."""

    def __init__(self, rad: str, hfc: list[str]):
        """Construct a Molecule object."""
        assert rad in MOLECULE_DATA.keys()
        rad_data = MOLECULE_DATA[rad]["data"]
        for c in hfc:
            assert c in rad_data.keys()
        self.rad, self.hfc = rad, hfc
        self.data = MOLECULE_DATA[rad]["data"]

    def data_generator(self, data: str) -> Iterable:
        """Construct a generator for a given property.

        Args:
            data (str): the property.
        Returns:
            List generator.
        """
        for hfc in self.hfc:
            yield self.data[hfc][data]

    def elements(self) -> Iterable:
        """Construct a generator for the elements of different nuclei.

        Returns:
            Return a generator of element names.

        """
        return self.data_generator("element")

    def get_data(self, idx: int, key: str):
        """Get data of a nucleus.

        Utility for used only for testing currently.

        .. todo::
        Make tests better and probably remove this functions.
        """
        return self.data[self.hfc[idx]][key]


class Quantum:
    """Quantum simulation class."""

    def __init__(self, molecules: list[Molecule]):
        """Construct the object.

        Args:
            molecules (list[Molecule]): List of two `Molecule`
            objects.

        """
        self.molecules = molecules
        self.particles = ["E", "E"] + sum([list(m.elements()) for m in molecules], [])
        self.multiplicities = list(map(multiplicity, self.particles))
        self.gamma = list(map(gamma, self.particles))

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
        gammas = enumerate(self.gamma)
        return -sum(B * g * 0.001 * self.spinop(i, axis) for i, g in gammas)

    def HH(self) -> np.array:
        pass
