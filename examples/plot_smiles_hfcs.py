#! /usr/bin/env python

import json

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.plot import plot_molecule, visualise_tensor
from radicalpy.utils import (
    define_xyz,
    mol_to_plot_arrays,
    smiles_to_3d,
)

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
    N5 = coords[17]
    N10 = coords[8]
    C4X = coords[16]
    C9A = coords[4]
    C10 = coords[9]
    C5X = coords[3]
    N14 = coords[10]
    N16 = coords[13]
    H20 = coords[35]
    H21 = coords[32]
    H22 = coords[33]
    H23 = coords[31]
    H24 = coords[34]
    H25 = coords[36]
    H26 = coords[37]
    H27 = coords[38]
    H28 = coords[39]
    H29 = coords[40]
    H30 = coords[41]
    H31 = coords[42]

    fx, fy, fz = define_xyz(N5, N10, C4X, C9A, C10, C5X)
    rot = [fx, fy, fz]  # Rotation matrix

    # Load HFCs from RadicalPy database
    flavin = rp.data.Molecule.fromdb(
        "flavin_anion",
        nuclei=[
            "N5",
            "N10",
            "N14",
            "N16",
            "H20",
            "H21",
            "H22",
            "H23",
            "H24",
            "H25",
            "H26",
            "H27",
            "H28",
            "H29",
            "H30",
            "H31",
        ],
    )

    N5hfc = flavin.nuclei[0].hfc.anisotropic
    N10hfc = flavin.nuclei[1].hfc.anisotropic
    N14hfc = flavin.nuclei[2].hfc.anisotropic
    N16hfc = flavin.nuclei[3].hfc.anisotropic
    H20hfc = flavin.nuclei[4].hfc.anisotropic
    H21hfc = flavin.nuclei[5].hfc.anisotropic
    H22hfc = flavin.nuclei[6].hfc.anisotropic
    H23hfc = flavin.nuclei[7].hfc.anisotropic
    H24hfc = flavin.nuclei[8].hfc.anisotropic
    H25hfc = flavin.nuclei[9].hfc.anisotropic
    H26hfc = flavin.nuclei[10].hfc.anisotropic
    H27hfc = flavin.nuclei[11].hfc.anisotropic
    H28hfc = flavin.nuclei[12].hfc.anisotropic
    H29hfc = flavin.nuclei[13].hfc.anisotropic
    H30hfc = flavin.nuclei[14].hfc.anisotropic
    H31hfc = flavin.nuclei[15].hfc.anisotropic

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
    main()
