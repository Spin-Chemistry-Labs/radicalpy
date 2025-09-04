#! /usr/bin/env python

from pathlib import Path

import MDAnalysis as mda
import numpy as np

from radicalpy.utils import define_xyz, get_rotation_matrix_euler_angles

# import nglview as nv


# Calculate rotation matrix from DmCry (PDB 4GU5) structure file
# Hiscock et al., PNAS, 2016, 113 (17) 4634-4639


def main():
    # Import the PDB and isolate residues
    trp_id = "342"
    uc = mda.Universe(Path(__file__).parent / "data/4GU5.pdb")
    fad = uc.select_atoms(
        "chainID A and resname FAD and name N1 C2 O2 N3 C4 O4 C4X N5 C5X C6 C7 C7M C8 C8M C9 C9A N10 C10 C1P"
    )
    trp = uc.select_atoms(
        f"chainID A and resname TRP and resid {trp_id} and name CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2"
    )
    # fad.write("fad.pdb")
    # trp.write("trp.pdb")

    # nv.show_mdanalysis(fad)
    # nv.show_mdanalysis(trp)

    # Find the centre-of-mass distance in Angstroms
    fad_com = fad.center_of_mass()
    trp_com = trp.center_of_mass()
    print(f"Centre-of-mass distance (A) = ", np.linalg.norm(fad_com - trp_com))

    # Select the atoms for calculating the xyz axes
    c4x = uc.select_atoms("chainID A and resname FAD and name C4X").positions.tolist()[
        0
    ]
    c9a = uc.select_atoms("chainID A and resname FAD and name C9A").positions.tolist()[
        0
    ]
    c10 = uc.select_atoms("chainID A and resname FAD and name C10").positions.tolist()[
        0
    ]
    c5x = uc.select_atoms("chainID A and resname FAD and name C5X").positions.tolist()[
        0
    ]
    n5 = uc.select_atoms("chainID A and resname FAD and name N5").positions.tolist()[0]
    n10 = uc.select_atoms("chainID A and resname FAD and name N10").positions.tolist()[
        0
    ]

    ne1 = uc.select_atoms(
        f"chainID A and resname TRP and resid {trp_id} and name NE1"
    ).positions.tolist()[0]
    ce3 = uc.select_atoms(
        f"chainID A and resname TRP and resid {trp_id} and name CE3"
    ).positions.tolist()[0]
    cd1 = uc.select_atoms(
        f"chainID A and resname TRP and resid {trp_id} and name CD1"
    ).positions.tolist()[0]
    ch2 = uc.select_atoms(
        f"chainID A and resname TRP and resid {trp_id} and name CH2"
    ).positions.tolist()[0]
    cg = uc.select_atoms(
        f"chainID A and resname TRP and resid {trp_id} and name CG"
    ).positions.tolist()[0]
    cd2 = uc.select_atoms(
        f"chainID A and resname TRP and resid {trp_id} and name CD2"
    ).positions.tolist()[0]

    # Define xyz for FAD
    fx, fy, fz = define_xyz(n5, n10, c4x, c9a, c10, c5x)

    # Define xyz for Trp
    c = (np.array(cg) + np.array(cd2)) * 0.5
    wx, wy, wz = define_xyz(ne1, c, ne1, ce3, cd1, ch2)

    # Normalise FAD to the centre
    rf = [fx, fy, fz]

    # Calculate the rotation matrix and euler angles
    w = [wx, -wy, -wz]
    R, alpha, beta, gamma = get_rotation_matrix_euler_angles(rf, w)
    print(f"Rotation matrix = ", R)
    print(
        f"Euler angles (deg) = ", np.degrees(alpha), np.degrees(beta), np.degrees(gamma)
    )


if __name__ == "__main__":
    main()
