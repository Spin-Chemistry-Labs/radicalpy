#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json

import radicalpy as rp
from radicalpy.plot import plot_molecule, visualise_tensor
from radicalpy.utils import (
    define_xyz,
    infer_bonds,
    parse_xyz,
)
from radicalpy.utils import is_fast_run

# Import xyz file and plot 3D structure with HFCs from RadicalPy database


def main():
    # xyz for FMN
    labels, elements, coords = parse_xyz("./data/NH2.xyz")
    bonds = infer_bonds(elements, coords)

    # Isolate key atomic positions
    (
        N00,
        H01,
        H02,
    ) = (
        coords[0],
        coords[1],
        coords[2],
    )
    x, y, z = define_xyz(N00, H01, H01, H02, N00, H02)
    rot = [x, y, z]  # Rotation matrix

    # Load HFCs from ORCA .out
    indices, isotopes, hfc_matrices = rp.utils.read_orca_hyperfine(
        "./data/NH2_A.out"
    )
    N00hfc = hfc_matrices[0]
    H01hfc = hfc_matrices[1]
    H02hfc = hfc_matrices[2]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("none")
    visualise_tensor(ax, N00hfc/np.linalg.norm(N00hfc), rot, N00, "blue")
    visualise_tensor(ax, H01hfc/np.linalg.norm(N00hfc), rot, H01, "lightcoral")
    visualise_tensor(ax, H02hfc/np.linalg.norm(N00hfc), rot, H02, "lightcoral")
    plot_molecule(
        ax, labels, elements, coords, bonds, show_labels=False, show_atoms=False
    )
    elev = 20
    azim = 30
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    ax.axis("off")
    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    if is_fast_run():
        main()
    else:
        main()
