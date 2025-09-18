#! /usr/bin/env python
"""
Shared utilities for physical constants and data paths.

This module provides a lightweight mechanism to load numerically usable
physical constants from a JSON file while preserving rich metadata
(e.g., units, symbols, references):

- ``Constant``: a subclass of ``float`` that carries a ``details`` attribute
  (a ``types.SimpleNamespace``) with auxiliary information about the constant.
  You can use a ``Constant`` anywhere a ``float`` is expected, and still access
  metadata via ``.details``, e.g. ``constants.mu_B.details.units``.

- ``Constant.fromjson(path)``: load a JSON mapping of name → {value, ...}
  into a ``SimpleNamespace`` of ``Constant`` objects, accessible by attribute.

- ``DATA_DIR``: path to bundled data files.

- ``constants``: the default namespace of constants loaded from
  ``DATA_DIR/constants.json``.
"""

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


DATA_DIR = Path(__file__).parent / "data_files"
constants = Constant.fromjson(DATA_DIR / "constants.json")
