#!/usr/bin/env python

import json
import sys
from pathlib import Path
from pprint import pprint


def process_line(line):
    # print(line)
    line = line.replace("\u03b1", "alpha")
    line = line.replace("\u03bc", "mu")
    line = line.replace("\u03b3", "gamma")
    line = line.replace("\u03c0", "pi")
    name, rest = line.split(",")
    alt_def = None
    if "(" in name:
        name, *alt_def = name.split("(")
        if isinstance(alt_def, list):
            alt_def = "(".join(alt_def)
        alt_def = alt_def[:-1]
    var, val = rest.split("=")
    var = var.strip()
    val = val.strip()
    val, *unit = val.split()
    val = float(val)
    unit = " ".join(unit)
    result = dict(name=name.strip(), val=val, unit=unit)
    if alt_def is not None:
        result["alt_def"] = alt_def
    return var, result


def main():
    if len(sys.argv) < 2:
        raise ValueError("File not specified!")
    path = Path(sys.argv[1])
    if not path.is_file():
        raise ValueError("File doesn't exists!")
    final = {}
    with open(path) as f:
        for line in f:
            line = line[:-1]
            if line != "":
                key, val = process_line(line)
                final[key] = val

    # pprint(final)
    print(final.keys())
    with open("out.json", "w") as j:
        json.dump(final, j, indent=2)


if __name__ == "__main__":
    main()
