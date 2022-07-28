import numpy as np
import scipy.sparse as sp

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


def pauli(multiplicity):
    assert multiplicity > 1
    if multiplicity == 2:
        return PAULI_FACTOR * np.array(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, -1.0j], [1.0j, 0.0]],
                [[1.0, 0.0], [0.0, -1.0]],
            ],
        )
    else:
        spin = (multiplicity - 1) / 2
        prjs = np.arange(multiplicity - 1, -1, -1) - spin
        raising_op = sp.spdiags(
            np.sqrt(spin * (spin + 1) - prjs * (prjs + 1)),
            [1],
            multiplicity,
            multiplicity,
        ).toarray()
        lowering_op = sp.spdiags(
            np.sqrt(spin * (spin + 1) - prjs * (prjs - 1)),
            [-1],
            multiplicity,
            multiplicity,
        ).toarray()
        x = PAULI_FACTOR * (raising_op + lowering_op)
        y = -PAULI_FACTOR * 1j * (raising_op - lowering_op)
        z = sp.spdiags(prjs, 0, multiplicity, multiplicity).toarray()
        return 0.5 * np.array([x, y, z])


if __name__ == "__main__":
    # print(pauli(2))
    print(PAULI["spin 1"])
    print(pauli(3))
    print(PAULI["spin 1"] - pauli(3))
