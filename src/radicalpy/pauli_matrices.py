import numpy as np

PAULI_FACTOR = 0.5

PAULI = {
    "spin 1/2": PAULI_FACTOR
    * np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, -1.0j], [1.0j, 0.0]],
            [[1.0, 0.0], [0.0, -1.0]],
        ],
    ),
    "spin 1": PAULI_FACTOR
    * np.array(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]] / np.sqrt(2),
            [[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]] / np.sqrt(2),
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ],
    ),
}


PAULI_SPIN_HALF = PAULI["spin 1/2"]
PAULI_SPIN_ONE = PAULI["spin 1"]

SIGMA_SPIN_HALF = PAULI["spin 1/2"]
SIGMA_SPIN_ONE = PAULI["spin 1"]
