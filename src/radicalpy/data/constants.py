#! /usr/bin/env python
import json

from .. import data

CONSTANTS_JSON = data.DATA_DIR / "constants.json"

with open(CONSTANTS_JSON) as f:
    CONSTANTS_DATA = json.load(f)
    """Dictionary containing constants.

    :meta hide-value:"""


def value(var_name: str) -> float:
    return CONSTANTS_DATA[var_name]["value"]


def details(var_name: str) -> dict:
    return CONSTANTS_DATA[var_name]
