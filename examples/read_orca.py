from pathlib import Path

import radicalpy as rp

print("Read from .out file")
data_dir = Path(__file__).parent / "data"
indices, isotopes, hfc_matrices = rp.utils.read_orca_hyperfine(data_dir / "NH2_A.out")
nuclei = [
    rp.data.Nucleus.fromisotope(isotope, hfc_matrix.tolist())
    for isotope, hfc_matrix in zip(isotopes, hfc_matrices)
]
molecule = rp.simulation.Molecule("mol1", nuclei)
print(molecule)


print("\nRead from .property.txt file")
indices, isotopes, hfc_matrices = rp.utils.read_orca_hyperfine(
    data_dir / "NH2_A.property.txt"
)
nuclei = [
    rp.data.Nucleus.fromisotope(isotope, hfc_matrix.tolist())
    for isotope, hfc_matrix in zip(isotopes, hfc_matrices)
]
molecule = rp.simulation.Molecule("mol1", nuclei)
print(molecule)
