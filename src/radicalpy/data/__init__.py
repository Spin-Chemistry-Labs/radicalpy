#! /usr/bin/env python

import json
from pathlib import Path

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
