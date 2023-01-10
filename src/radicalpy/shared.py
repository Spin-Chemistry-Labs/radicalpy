#! /usr/bin/env python
"""Shared objects."""
import json
from pathlib import Path
from types import SimpleNamespace


class Constant(float):
    """Constan class.

    Extends float with the `Constant.details` member.
    """

    details: SimpleNamespace
    """Details (e.g. units) of the constant."""

    def __new__(cls, details: dict):  # noqa D102
        obj = super().__new__(cls, details.pop("value"))
        obj.details = SimpleNamespace(**details)
        return obj

    @staticmethod
    def fromjson(json_file: Path) -> SimpleNamespace:
        """Read all constants from the JSON file.

        Args:
            json_file (str)

        Returns:
            SimpleNamespace: A namespace containing all constants.
        """
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        return SimpleNamespace(**{k: Constant(v) for k, v in data.items()})


DATA_DIR = Path(__file__).parent / "data"
constants = Constant.fromjson(DATA_DIR / "constants.json")
