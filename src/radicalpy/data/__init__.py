import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
SPIN_DATA_JSON = DATA_DIR / "spin_data.json"
MOLECULES_DIR = DATA_DIR / "molecules"

SPIN_DATA = json.load(open(SPIN_DATA_JSON))
"""Dictionary containing spin data for elements.

:meta hide-value:"""


def _get_molecules(molecules_dir=MOLECULES_DIR):
    molecules = {}
    for json_path in molecules_dir.glob("*.json"):
        molecule_name = json_path.with_suffix("").name
        molecules[molecule_name] = json.load(open(json_path))
    return molecules


MOLECULE_DATA = _get_molecules()
"""Dictionary containing data for each molecule.

:meta hide-value: """


def multiplicity(element: str):
    return SPIN_DATA[element]["multiplicity"]
