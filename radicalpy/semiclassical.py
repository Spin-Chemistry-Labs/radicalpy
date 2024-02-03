#!/usr/bin/env python
import numpy as np

import radicalpy as rp
from radicalpy.data import Molecule

H, N = 0.5, 1
FAD_spin = [N, N, N, N, H, H, H, H, H, H, H, H, H, H, H, H, H, H, H, H, H]
FAD_HFCs = [
    0.1238,
    -0.0591,
    0.3927,
    0.2115,
    0.0008,
    0.024,
    -0.0127,
    0.0636,
    0.3899,
    0.0319,
    0.192,
    0.4817,
    0.0749,
    -0.0091,
    -0.0495,
    -0.0209,
    0.0487,
    -0.158,
    -0.7693,
    -0.0247,
    -0.0036,
]
Trp_spin = [N, N, H, H, H, H, H, H, H, H, H, H, H, H]
Trp_HFCs = [
    0.1465,
    0.3215,
    -0.0396,
    -0.0931,
    1.6046,
    0.0457,
    -0.0104,
    0.0233,
    -0.278,
    -0.5983,
    -0.488,
    -0.3637,
    -0.2083,
    -0.04,
]


def semiclassical_calculate_tau(Is, HFCs):
    I = [Is * (Is + 1) for Is in Is]
    HFC = [HFCs**2 for HFCs in HFCs]
    return (sum(np.array(I) * np.array(HFC)) / 6) ** -0.5


def angle(num_samples):
    for i in range(num_samples):
        while True:
            theta_r = np.random.rand() * np.pi
            s_r = np.random.rand()
            if s_r < np.sin(theta_r):
                theta = theta_r
                break

        phi = 2 * np.pi * np.random.rand()
        yield theta, phi


def cmp_hfcs(mol, hfcs):
    mol_sorted = sorted(FAD.nuclei, key=lambda k: k.hfc.isotropic)
    hfcs_sorted = sorted(FAD_HFCs)
    for n1, n2 in zip(mol_sorted, hfcs_sorted):
        print(f"{n1.hfc.isotropic:10.4}, {n2:10.4}")


if __name__ == "__main__":
    sample_number = 10
    np.random.seed(42)
    FAD = Molecule.all_nuclei("flavin_anion")
    Trp = Molecule.all_nuclei("tryptophan_cation")
    print(f"{len(FAD.nuclei)=}")
    print(f"{len(FAD_HFCs)=}")
    print(f"{len(Trp.nuclei)=}")
    print(f"{len(Trp_HFCs)=}")
    cmp_hfcs(FAD, FAD_HFCs)
