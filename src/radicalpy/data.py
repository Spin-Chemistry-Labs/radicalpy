import json
from pathlib import Path

datadir = Path(__file__).parent / "data"
print("THIS IS DATADIR", datadir)
spin_data = json.load(open(datadir / "spin_data.json"))
