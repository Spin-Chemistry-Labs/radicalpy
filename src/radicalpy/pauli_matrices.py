import numpy as np
import scipy.sparse as sp

# from .data import spin

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
    "spin 1": np.array(
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


def pauli(mult):
    assert mult > 1
    if mult == 2:
        return PAULI_FACTOR * np.array(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, -1.0j], [1.0j, 0.0]],
                [[1.0, 0.0], [0.0, -1.0]],
            ]
        )
    else:
        spin = (mult - 1) / 2
        prjs = np.arange(mult - 1, -1, -1) - spin
        raising_data = np.sqrt(spin * (spin + 1) - prjs * (prjs + 1))
        raising_op = sp.spdiags(raising_data, [1], mult, mult).toarray()
        lowering_data = np.sqrt(spin * (spin + 1) - prjs * (prjs - 1))
        lowering_op = sp.spdiags(lowering_data, [-1], mult, mult).toarray()
        x = PAULI_FACTOR * (raising_op + lowering_op)
        y = -PAULI_FACTOR * 1j * (raising_op - lowering_op)
        z = sp.spdiags(prjs, 0, mult, mult).toarray()
        # @todo(vatai): save raising/lowering
        return np.array([x, y, z])


def pauli_for_element(element):
    pass


if __name__ == "__main__":
    # print(pauli(2))
    # print(PAULI["spin 1"])
    print(pauli(3))
    # print(PAULI["spin 1"] - pauli(3))
    pass
