import json
from collections import OrderedDict

import numpy as np

import utils

from . import data, utils
from .flavin_3x3 import flavin as molecule

# from .trp import TRP as molecule


# from .TYRrad import TYRrad as molecule


def isotropic(anisotropic: np.ndarray):
    return anisotropic.trace() / 3


MOLECULE = "flavin_anion"
# MOLECULE = "tryptophan_cation"
# MOLECULE = "tyrosine_neutral"


def get_srp():
    molecule_data = data.MOLECULE_DATA[MOLECULE]["data"]
    rp = []
    rp_keys = []
    for k, v in molecule_data.items():
        rp.append(v["hfc"])
        rp_keys.append(k)

    zrp = zip(rp_keys, rp)
    srp = sorted(zrp, key=lambda t: t[1], reverse=True)
    return srp


def get_sor():
    orca = []
    orca_keys = []
    for k, v in molecule.items():
        m = np.array(v)
        # m = isotropic(m)
        m = utils.MHz_to_mT(m)
        orca.append(m)
        orca_keys.append(k)

    zor = zip(orca_keys, orca)
    sor = sorted(zor, key=lambda t: isotropic(t[1]), reverse=True)
    return sor


def flavin_proc(sor):
    new = OrderedDict()
    dsor = OrderedDict(sor)
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
    return new


def flavin_make(new):
    new_molecule = dict(data.MOLECULE_DATA[MOLECULE])
    for k, v in new.items():
        new_molecule["data"][k]["hfc"] = list(map(lambda t: list(t), v))

    new_molecule["data"]["N10"] = new_molecule["data"].pop("N6")
    return new_molecule


if __name__ == "__main__":
    N = len(molecule)

    srp = get_srp()
    sor = get_sor()
    print(len(srp), len(sor))

    if MOLECULE == "flavin_anion":
        new = flavin_proc(sor)
        nlst = list(new.items())
        nlst = sorted(nlst, key=lambda t: isotropic(t[1]), reverse=True)
    else:
        nlst = list(sor)

    print(f"{MOLECULE}")
    print("idx  json (old)    json (new)         orca")
    for i in range(N):
        print(
            f"{i=:2} {srp[i][1]:7} {srp[i][0]:5} {nlst[i][0]:6} {isotropic(nlst[i][1]):10.5} {isotropic(sor[i][1]):10.5} {sor[i][0]:8} {i=:2}"
        )

    if MOLECULE == "flavin_anion":
        new_molecule = flavin_make(new)
    else:
        pass

    print(new_molecule)
    with open(f"{MOLECULE}.json", "w") as f:
        json.dump(new_molecule, f, indent=2)
