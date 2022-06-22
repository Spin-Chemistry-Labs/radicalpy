#!/usr/bin/env python

import numpy as np

from .pauli_matrices import SIGMA_SPIN_HALF, SIGMA_SPIN_ONE

# This is just something based on some earlier scripts... nothing is
# set in stone yet.


class Sim:
    def __init__(self, spin_halves, spin_ones):
        self.spin_halves = spin_halves
        self.spin_ones = spin_ones
        self.num_particles = spin_halves + spin_ones

        self.sigmas = spin_halves * [SIGMA_SPIN_HALF]
        self.sigmas += spin_ones * [SIGMA_SPIN_ONE]

    def spin_operator_axis(self, partice_index: int, axis: int) -> np.array:
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
