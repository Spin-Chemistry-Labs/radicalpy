import os
import sys

import numpy as np

import utils

from . import data, utils
from .flavin_3x3 import flavin

# SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))


def isotropic(anisotropic: np.ndarray):
    return anisotropic.trace() / 3


if __name__ == "__main__":
    N = len(flavin)

    flavin_data = data.MOLECULE_DATA["flavin_anion"]["data"]
    rp = np.zeros(N)
    rp_keys = []
    for i, (k, v) in enumerate(flavin_data.items()):
        rp[i] = v["hfc"]
        rp_keys.append(k)

    orca = np.zeros(N)
    orca_keys = []
    for i, (k, v) in enumerate(flavin.items()):
        m = np.array(v)
        m = isotropic(m)
        m = utils.MHz_to_mT(m)
        orca[i] = m
        orca_keys.append(k)

    # print(f"{rp=}")
    # print(f"{rp_keys=}")
    # print(f"{orca=}")
    # print(f"{orca_keys=}")

    # for i in range(N):
    #     print(f"{i=}  {rp[i]:8} {rp_keys[i]:5} {orca[i]:10.5} {orca_keys[i]:5}")

    zrp = zip(rp, rp_keys)
    zor = zip(orca, orca_keys)

    srp = sorted(zrp, key=lambda t: t[0])
    sor = sorted(zor, key=lambda t: t[0])

    new = list(map(lambda t: t[1], srp))
    new[3], new[4] = new[4], new[3]
    new[9], new[10], new[11], new[12], new[15] = (
        new[8],
        new[9],
        new[10],
        new[11],
        new[15],
    )

    for i in range(N):
        print(
            f"{i=:3}  {srp[i][0]:8} {srp[i][1]:5} {sor[i][0]:10.5} {sor[i][1]:5} {new[i]} :{i}"
        )
