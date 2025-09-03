#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json

import radicalpy as rp
from radicalpy.plot import plot_molecule, visualise_tensor
from radicalpy.utils import (
    define_xyz,
    mol_to_plot_arrays,
    smiles_to_3d,
)
from radicalpy.utils import is_fast_run

# Import SMILES and plot 3D structure with HFCs from RadicalPy database


def main():
    # SMILES for FMN
    smiles = (
        "CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)C[C@@H]([C@@H]([C@@H](COP(=O)(O)O)O)O)O"
    )

    # Generate RDKit molecule with 3D coords
    mol = smiles_to_3d(smiles, add_h=True)

    # Extract arrays for plotting
    labels, elements, coords, bonds = mol_to_plot_arrays(mol)

    # Isolate key atomic positions
    (
        N5,
        N10,
        C4X,
        C9A,
        C10,
        C5X,
        N14,
        N16,
        H20,
        H21,
        H22,
        H23,
        H24,
        H25,
        H26,
        H27,
        H28,
        H29,
        H30,
        H31,
    ) = (
        coords[17],
        coords[8],
        coords[16],
        coords[4],
        coords[9],
        coords[3],
        coords[10],
        coords[13],
        coords[35],
        coords[32],
        coords[33],
        coords[31],
        coords[34],
        coords[36],
        coords[37],
        coords[38],
        coords[39],
        coords[40],
        coords[41],
        coords[42],
    )
    fx, fy, fz = define_xyz(N5, N10, C4X, C9A, C10, C5X)
    rot = [fx, fy, fz]  # Rotation matrix

    # Load HFCs from RadicalPy database
    with open(rp.data.get_data("molecules/flavin_anion.json"), encoding="utf-8") as f:
        flavin_dict = json.load(f)
    N5hfc = flavin_dict["data"]["N5"]["hfc"]
    N10hfc = flavin_dict["data"]["N10"]["hfc"]
    N14hfc = flavin_dict["data"]["N14"]["hfc"]
    N16hfc = flavin_dict["data"]["N16"]["hfc"]
    H20hfc = flavin_dict["data"]["H20"]["hfc"]
    H21hfc = flavin_dict["data"]["H21"]["hfc"]
    H22hfc = flavin_dict["data"]["H22"]["hfc"]
    H23hfc = flavin_dict["data"]["H23"]["hfc"]
    H24hfc = flavin_dict["data"]["H24"]["hfc"]
    H25hfc = flavin_dict["data"]["H25"]["hfc"]
    H26hfc = flavin_dict["data"]["H26"]["hfc"]
    H27hfc = flavin_dict["data"]["H27"]["hfc"]
    H28hfc = flavin_dict["data"]["H28"]["hfc"]
    H29hfc = flavin_dict["data"]["H29"]["hfc"]
    H30hfc = flavin_dict["data"]["H30"]["hfc"]
    H31hfc = flavin_dict["data"]["H31"]["hfc"]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("none")
    visualise_tensor(ax, N5hfc, rot, N5, "blue")
    visualise_tensor(ax, N10hfc, rot, N10, "blue")
    visualise_tensor(ax, N14hfc, rot, N14, "blue")
    visualise_tensor(ax, N16hfc, rot, N16, "blue")
    visualise_tensor(ax, H20hfc, rot, H20, "lightcoral")
    visualise_tensor(ax, H21hfc, rot, H21, "lightcoral")
    visualise_tensor(ax, H22hfc, rot, H22, "lightcoral")
    visualise_tensor(ax, H23hfc, rot, H23, "lightcoral")
    visualise_tensor(ax, H24hfc, rot, H24, "lightcoral")
    visualise_tensor(ax, H25hfc, rot, H25, "lightcoral")
    visualise_tensor(ax, H26hfc, rot, H26, "lightcoral")
    visualise_tensor(ax, H27hfc, rot, H27, "lightcoral")
    visualise_tensor(ax, H28hfc, rot, H28, "lightcoral")
    visualise_tensor(ax, H29hfc, rot, H29, "lightcoral")
    visualise_tensor(ax, H30hfc, rot, H30, "lightcoral")
    visualise_tensor(ax, H31hfc, rot, H31, "lightcoral")
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
