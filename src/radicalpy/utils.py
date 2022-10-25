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
