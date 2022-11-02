import json
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
    dsor = OrderedDict(sor)
    new = OrderedDict()

    H_21_22_23 = (dsor["18H"] + dsor["19H"] + dsor["20H"]) / 3
    new["H21"] = H_21_22_23
    new["H22"] = H_21_22_23
    new["H23"] = H_21_22_23
    new["H24"] = dsor["24H"]
    new["N16"] = dsor["15N"]
    new["H31"] = dsor["26H"]
    new["N14"] = dsor["2N"]
    new["H28"] = dsor["28H"]
    new["H20"] = dsor["25H"]
    H_25_26_27 = (dsor["21H"] + dsor["22H"] + dsor["23H"]) / 3
    new["H25"] = H_25_26_27
    new["H26"] = H_25_26_27
    new["H27"] = H_25_26_27
    new["N6"] = dsor["4N"]
    new["H29"] = dsor["29H"]
    new["H30"] = dsor["30H"]
    new["N5"] = dsor["11N"]
    nlst = list(new.items())
    nlst = sorted(nlst, key=lambda t: isotropic(t[1]))

    for i in range(N):
        print(
            f"{i=:2} {srp[i][1]:7} {srp[i][0]:5} {nlst[i][0]:5} {isotropic(nlst[i][1]):10.5} {isotropic(sor[i][1]):10.5} {sor[i][0]:5} {i=:2}"
        )

    flavin_data = data.MOLECULE_DATA["flavin_anion"]
    new_flavin = dict(flavin_data)
    for k, v in new.items():
        new_flavin["data"][k]["hfc"] = list(map(lambda t: list(t), v))

    # print(new_flavin)
    with open("flavin_anion.json", "w") as f:
        json.dump(new_flavin, f, indent=2)
