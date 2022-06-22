#!/usr/bin/env python

import numpy as np

from .pauli_matrices import SIGMA_SPIN_HALF, SIGMA_SPIN_ONE

# This is just something based on some earlier scripts... nothing is
# set in stone yet.


class Sim:
    def __init__(self, spin_halves, spin_ones):
        self.sigmas = spin_halves * [SIGMA_SPIN_HALF] + spin_ones * [SIGMA_SPIN_ONE]
