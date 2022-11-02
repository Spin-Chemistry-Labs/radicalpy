from collections import OrderedDict

import numpy as np

import utils

from . import data, utils
from .flavin_3x3 import flavin


def isotropic(anisotropic: np.ndarray):
    return anisotropic.trace() / 3


def get_srp():

    flavin_data = data.MOLECULE_DATA["flavin_anion"]["data"]
    rp = np.zeros(N)
    rp_keys = []
    for i, (k, v) in enumerate(flavin_data.items()):
        rp[i] = v["hfc"]
        rp_keys.append(k)

    zrp = zip(rp_keys, rp)
    srp = sorted(zrp, key=lambda t: t[1])
    return srp


def get_sor():
    orca = []
    orca_keys = []
    for i, (k, v) in enumerate(flavin.items()):
        m = np.array(v)
        # m = isotropic(m)
        m = utils.MHz_to_mT(m)
        orca.append(m)
        orca_keys.append(k)

    zor = zip(orca_keys, orca)
    sor = sorted(zor, key=lambda t: isotropic(t[1]))
    return sor


if __name__ == "__main__":
    N = len(flavin)

    srp = get_srp()
    sor = get_sor()
    new = OrderedDict()

    for i in range(N):
        print(
            f"{i=:2} {srp[i][1]:7} {srp[i][0]:5} {isotropic(sor[i][1]):10.5} {sor[i][0]:5} {i=:2}"
        )
    # dsor = OrderedDict(sor)
    # print(list(dsor.items()))
