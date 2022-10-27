#!/usr/bin/env python

import numpy as np


def spherical_to_cartesian(theta, phi):
    return np.array(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )


def spin_quantum_number(multiplicity: int) -> float:
    return float(multiplicity - 1) / 2.0
