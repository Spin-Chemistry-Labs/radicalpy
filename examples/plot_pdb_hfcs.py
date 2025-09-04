#! /usr/bin/env python

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.plot import plot_molecule, set_equal_aspect, visualise_tensor
from radicalpy.utils import (
    define_xyz,
    parse_pdb,
    rotate_axes,
)

# Import PDB and plot 3D structure with HFCs from RadicalPy database


def main():
    # Import PDB file
    data_dir = Path(__file__).parent / "data"
    kwargs = {"use_rdkit_bonds": True, "label_scheme": "atom"}
    labels, elements, coords, bonds = parse_pdb(data_dir / "fad.pdb", **kwargs)
    labels2, elements2, coords2, bonds2 = parse_pdb(data_dir / "trp.pdb", **kwargs)

    # Isolate key atomic positions
    N5 = coords[7]
    N10 = coords[16]
    C4X = coords[6]
    C9A = coords[15]
    C10 = coords[17]
    C5X = coords[8]

    NE1 = coords2[4]
    CG = coords2[1]
    CE3 = coords2[6]
    CD1 = coords2[2]
    CH2 = coords2[9]
    CD2 = coords2[3]

    fx, fy, fz = define_xyz(N5, N10, C4X, C9A, C10, C5X)
    c = (np.array(CG) + np.array(CD2)) * 0.5
    wx, wy, wz = define_xyz(NE1, c, NE1, CE3, CD1, CH2)
    rot = [fx, fy, fz]  # Rotation matrix
    trpx, trpy, trpz = rotate_axes(rot, wx, wy, wz)
    rot2 = [trpx, trpy, trpz]  # Rotation matrix

    flavin = rp.simulation.Molecule.fromdb(
        "flavin_anion", ["N5", "N10"]
    )  # , "H27", "H29"])
    trp = rp.simulation.Molecule.fromdb("tryptophan_cation", ["N1"])  # , "Hbeta1"])

    N5hfc = flavin.nuclei[0].hfc.anisotropic
    N10hfc = flavin.nuclei[1].hfc.anisotropic
    N1hfc = trp.nuclei[0].hfc.anisotropic

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("none")
    visualise_tensor(ax, N5hfc, rot, N5, "blue")
    visualise_tensor(ax, N10hfc, rot, N10, "blue")
    plot_molecule(
        ax, labels, elements, coords, bonds, show_labels=False, show_atoms=False
    )
    visualise_tensor(ax, N1hfc, rot2, NE1, "blue")
    plot_molecule(
        ax, labels2, elements2, coords2, bonds2, show_labels=False, show_atoms=False
    )
    allX = np.concatenate([coords[:, 0], coords2[:, 0]])
    allY = np.concatenate([coords[:, 1], coords2[:, 1]])
    allZ = np.concatenate([coords[:, 2], coords2[:, 2]])
    set_equal_aspect(ax, allX, allY, allZ)
    elev = 20
    azim = 30
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    ax.axis("off")
    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()
