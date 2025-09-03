#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json

import radicalpy as rp
from radicalpy.plot import set_equal_aspect, plot_molecule, visualise_tensor
from radicalpy.utils import (
    define_xyz,
    parse_pdb,
    rotate_axes,
)
from radicalpy.utils import is_fast_run

# Import PDB and plot 3D structure with HFCs from RadicalPy database


def main():
    # Import PDB file
    labels, elements, coords, bonds = parse_pdb("./data/fad.pdb", use_rdkit_bonds=True, label_scheme="atom")
    labels2, elements2, coords2, bonds2 = parse_pdb("./data/trp.pdb", use_rdkit_bonds=True, label_scheme="atom")
    print(labels2)

    # Isolate key atomic positions
    (
        N5,
        N10,
        C4X,
        C9A,
        C10,
        C5X,
    ) = (
        coords[7],
        coords[16],
        coords[6],
        coords[15],
        coords[17],
        coords[8],
    )

    (
        NE1,
        CG,
        CE3,
        CD1,
        CH2,
        CD2,
    ) = (
        coords2[4],
        coords2[1],
        coords2[6],
        coords2[2],
        coords2[9],
        coords2[3],
    )

    fx, fy, fz = define_xyz(N5, N10, C4X, C9A, C10, C5X)
    c = (np.array(CG) + np.array(CD2)) * 0.5
    wx, wy, wz = define_xyz(NE1, c, NE1, CE3, CD1, CH2)
    rot = [fx, fy, fz]  # Rotation matrix
    trpx, trpy, trpz = rotate_axes(rot, wx, wy, wz)
    rot2 = [trpx, trpy, trpz]  # Rotation matrix

    # Load HFCs from RadicalPy database
    with open(rp.data.get_data("molecules/flavin_anion.json"), encoding="utf-8") as f:
        flavin_dict = json.load(f)
    N5hfc = flavin_dict["data"]["N5"]["hfc"]
    N10hfc = flavin_dict["data"]["N10"]["hfc"]

    with open(rp.data.get_data("molecules/tryptophan_cation.json"), encoding="utf-8") as f:
        trp_dict = json.load(f)
    N1hfc = trp_dict["data"]["N1"]["hfc"]
    
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
    allX = np.concatenate([coords[:,0], coords2[:,0]])
    allY = np.concatenate([coords[:,1], coords2[:,1]])
    allZ = np.concatenate([coords[:,2], coords2[:,2]])
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
    if is_fast_run():
        main()
    else:
        main()
