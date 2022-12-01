#! /usr/bin/env python

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_DIR = Path(__file__).parent
SPIN_DATA_JSON = DATA_DIR / "spin_data.json"
MOLECULES_DIR = DATA_DIR / "molecules"

with open(SPIN_DATA_JSON) as f:
    SPIN_DATA = json.load(f)
    """Dictionary containing spin data for elements.

    :meta hide-value:"""


def get_molecules(molecules_dir=MOLECULES_DIR):
    molecules = {}
    for json_path in sorted(molecules_dir.glob("*.json")):
        molecule_name = json_path.with_suffix("").name
        with open(json_path) as f:
            molecules[molecule_name] = json.load(f)
    return molecules


MOLECULE_DATA = get_molecules()
"""Dictionary containing data for each molecule.

:meta hide-value: """


def gamma_T(element: str):
    """Return the `gamma` value of an element in Tesla."""
    return SPIN_DATA[element]["gamma"]


def gamma_mT(element: str):
    """Return the `gamma` value of an element in milli-Tesla."""
    return SPIN_DATA[element]["gamma"] * 0.001


def multiplicity(element: str):
    return SPIN_DATA[element]["multiplicity"]


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
