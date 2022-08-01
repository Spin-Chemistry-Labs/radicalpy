#!/usr/bin/env python

from math import prod

import numpy as np

from .data import MOLECULE_DATA, SPIN_DATA
from .pauli_matrices import pauli

# This is just something based on some earlier scripts... nothing is
# set in stone yet.


class Molecule:
    def __init__(self, rad, hfc):
        self._check_rad_and_hfc(rad, hfc)
        self.rad, self.hfc = rad, hfc
        self.data = MOLECULE_DATA[rad]["data"]

    def _check_rad_and_hfc(self, rad, hfc):
        assert rad in MOLECULE_DATA.keys()
        rad_data = MOLECULE_DATA[rad]["data"]
        for c in hfc:
            assert c in rad_data.keys()

    def get_data(self, idx: int, data: str):
        return self.data[self.hfc[idx]][data]

    def data_generator(self, data: str):
        for hfc in self.hfc:
            yield self.data[hfc][data]

    # def get_elemprop(self, idx: int, property: str):
    #     return SPIN_DATA[self.get_data(idx, "element")][property]

    # def elemprop_generator(self, property: str):
    #     for hfc in self.data_generator("element"):
    #         yield SPIN_DATA[hfc][property]


def spinop(mult: list[int], idx: int, axis: str) -> np.array:
    """Spin operator."""
    assert 0 <= idx and idx < len(mult)

    sigma = pauli(mult[idx])[axis]
    eye_before = np.eye(prod(m for m in mult[:idx]))
    eye_after = np.eye(prod(m for m in mult[idx + 1 :]))

    return np.kron(np.kron(eye_before, sigma), eye_after)


class Sim:
    """Simulation class foo bar.

    .. todo::
       Move const to json."""

    def __init__(self, molecules: list[Molecule], kinetics=None):
        self.molecules = molecules
        self.particles = ["E", "E"] + sum(
            [list(m.data_generator("element")) for m in molecules], []
        )
        self.multiplicities = list(
            map(lambda t: SPIN_DATA[t]["multiplicity"], self.particles)
        )

        self.const = dict(
            ge=1.760859644e8,
            gn=267.513e3,
        )

        self.hamiltonians = {}
        self.hamiltonians["zeeman"] = self.Hzeeman(0.5)

        # if kinetics:
        #     self.hamiltonians["kinetics"] = self.kinetics(**kinetics)

    def Hzeeman(self, B0):
        """Calculate the Zeeman Hamiltonian.

        .. todo::
           Fix :code:`self.const`."""

        mult = self.multiplicities

        omega0 = self.const["ge"] * B0  # Electron Larmor freq.
        Hzee = omega0 * sum(spinop(mult, i, "z") for i in range(2))

        omega0n = -self.const["gn"] * B0  # Nuclear Larmor freq.
        Hzee += omega0n * sum(
            spinop(mult, i, "z") for i in range(2, len(self.particles))
        )

        # self.update_hamiltonian()
        return Hzee

    def hyperfine(self):
        pass

    def kinetics(self, model, rate):
        """Calculate the kinetic superoperator."""
        self.update_hamiltonian()

    def update_hamiltonian(self):
        self.H = sum(self.hamiltonians.values())

    def spin_operator_axis(self, partice_index: int, axis: int) -> np.array:
        """Spin operator for a single axis."""

        print("AXIS", self.sigmas[0].shape[0])
        assert partice_index < self.num_particles
        assert axis < self.sigmas[0].shape[0]
        kron_pre_identity_size = 0
        kron_post_identity_size = 0
        for i, mats in enumerate(self.sigmas):
            size = mats.shape[-1]
            if partice_index < i:
                kron_pre_identity_size += size
            if partice_index > i:
                kron_pre_identity_size += size
        result = self.sigmas[partice_index][axis]
        result = np.kron(np.eye(kron_pre_identity_size), result)
        result = np.kron(result, np.eye(kron_post_identity_size))
        return result
