#!/usr/bin/env python

from typing import Iterable

import numpy as np

from .pauli_matrices import SIGMA_SPIN_HALF, SIGMA_SPIN_ONE

# This is just something based on some earlier scripts... nothing is
# set in stone yet.


def spin_operator_axis(partice_index: int | Iterable[int], axis: int) -> np.array:
    """Spin operator."""
    result = 1.0
    for i in range(self.num_particles):
        m = SIGMA[axis]
        if partice_index != i:
            m = np.eye(*m.shape)
        result = np.kron(result, m)
    return result


class Sim:
    """Simulation class foo bar."""

    def __init__(self, rad1, hfc1, rad2, hfc2, kinetics=None):
        self.num_particles = len(hfc1) + len(hfc2)

        self.sigmas = spin_halves * [SIGMA_SPIN_HALF]
        self.sigmas += spin_ones * [SIGMA_SPIN_ONE]

        self.hamiltonians["zeeman"] = self.Hzeeman(rad1, hfc1, rad2, hfc2)
        if kinetics:
            self.hamiltonians["kinetics"] = self.kinetics(**kinetics)

    def Hzeeman(self):
        """Calculate the Zeeman Hamiltonian."""
        omega0 = self.const["ge"] * self.const["B0"]  # Electron Larmor freq.
        self.Hzee = omega0 * self.sum(spin_op_axis(electron_idxs, self.axis))
        omega0n = -self.const["gn"] * self.const["B0"]  # Nuclear Larmor freq.
        self.Hzee += omega0n * self.sum(spin_op_axis(nucleus_idxs, self.axis))
        # self.update_hamiltonian()

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
