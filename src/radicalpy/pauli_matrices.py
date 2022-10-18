import numpy as np
import scipy.sparse as sp


def pauli(mult: int):
    """Generate Pauli matrices.

    Generates the Pauli matrices corresponding to a given multiplicity.

    Args:
        mult (int): The multiplicity of the element.

    Return:
        dict: A dictionary containing 6 `np.array` matrices of
        shape `(mult, mult)`:
            - the unit operator `result["u"]`,
            - raising operator `result["p"]`,
            - lowering operator `result["m"]`,
            - Pauli matrix for x axis `result["x"]`,
            - Pauli matrix for y axis `result["y"]`,
            - Pauli matrix for z axis `result["z"]`.

    """
    assert mult > 1
    result = {}
    if mult == 2:
        result["u"] = np.array([[1, 0], [0, 1]])
        result["p"] = np.array([[0, 1], [0, 0]])
        result["m"] = np.array([[0, 0], [1, 0]])
        result["x"] = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
        result["y"] = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
        result["z"] = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    else:
        spin = (mult - 1) / 2
        prjs = np.arange(mult - 1, -1, -1) - spin

        p_data = np.sqrt(spin * (spin + 1) - prjs * (prjs + 1))
        m_data = np.sqrt(spin * (spin + 1) - prjs * (prjs - 1))

        result["u"] = np.eye(mult)
        result["p"] = sp.spdiags(p_data, [1], mult, mult).toarray()
        result["m"] = sp.spdiags(m_data, [-1], mult, mult).toarray()
        result["x"] = 0.5 * (result["p"] + result["m"])
        result["y"] = -0.5 * 1j * (result["p"] - result["m"])
        result["z"] = sp.spdiags(prjs, 0, mult, mult).toarray()
    return result


if __name__ == "__main__":
    pass
