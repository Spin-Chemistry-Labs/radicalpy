import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SPIN_DATA_JSON = "spin_data.json"
MOLECULES_DIR = DATA_DIR / "molecules"


def _get_molecules(molecules_dir=MOLECULES_DIR):
    molecules = {}
    for json_path in molecules_dir.glob("*.json"):
        molecule_name = json_path.with_suffix("").name
        molecules[molecule_name] = json.load(open(json_path))
    return molecules


spin = json.load(open(DATA_DIR / SPIN_DATA_JSON))
molecules = _get_molecules()
