#! /usr/bin/env python
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp


def spin_quantum_number(multiplicity: int) -> float:
    """Spin multiplicity to spin quantum number.

    Args:
            multiplicity (int): Spin multiplicity.

    Returns:
            float: Spin quantum number.

    """
    return float(multiplicity - 1) / 2.0


def isotropic(anisotropic: np.ndarray or list) -> float:
    """Anisotropic tensor to isotropic value.

    Args:
            anisotropic (np.ndarray or list): The 3x3 interaction tensor matrix.

    Returns:
            float: isotropic value.
    """
    return np.trace(anisotropic) / 3


DATA_DIR = Path(__file__).parent / "data"
SPIN_DATA_JSON = DATA_DIR / "spin_data.json"
MOLECULES_DIR = DATA_DIR / "molecules"
CONSTANTS_JSON = DATA_DIR / "constants.json"

with open(SPIN_DATA_JSON) as f:
    SPIN_DATA = json.load(f)
    """Dictionary containing spin data for elements.

    :meta hide-value:"""


def get_molecules(molecules_dir=MOLECULES_DIR):
    molecules = {}
    for json_path in sorted(molecules_dir.glob("*.json")):
        molecule_name = json_path.with_suffix("").name
        with open(json_path) as f:
            molecules[molecule_name] = json.load(f)
    return molecules


MOLECULE_DATA = get_molecules()
"""Dictionary containing data for each molecule.

:meta hide-value: """


def gamma_T(element: str):
    """Return the `gamma` value of an element in Tesla."""
    return SPIN_DATA[element]["gamma"]


def gamma_mT(element: str):
    """Return the `gamma` value of an element in milli-Tesla."""
    return SPIN_DATA[element]["gamma"] * 0.001


def multiplicity(element: str):
    return SPIN_DATA[element]["multiplicity"]


def pauli(mult: int):
    """Generate Pauli matrices.

    Generates the Pauli matrices corresponding to a given multiplicity.

    Args:
        mult (int): The multiplicity of the element.

    Return:
        dict: A dictionary containing 6 `np.array` matrices of
        shape `(mult, mult)`:

        - the unit operator `result["u"]`,
        - raising operator `result["p"]`,
        - lowering operator `result["m"]`,
        - Pauli matrix for x axis `result["x"]`,
        - Pauli matrix for y axis `result["y"]`,
        - Pauli matrix for z axis `result["z"]`.
    """
    assert mult > 1
    result = {}
    if mult == 2:
        result["u"] = np.array([[1, 0], [0, 1]])
        result["p"] = np.array([[0, 1], [0, 0]])
        result["m"] = np.array([[0, 0], [1, 0]])
        result["x"] = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
        result["y"] = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
        result["z"] = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    else:
        spin = (mult - 1) / 2
        prjs = np.arange(mult - 1, -1, -1) - spin

        p_data = np.sqrt(spin * (spin + 1) - prjs * (prjs + 1))
        m_data = np.sqrt(spin * (spin + 1) - prjs * (prjs - 1))

        result["u"] = np.eye(mult)
        result["p"] = sp.spdiags(p_data, [1], mult, mult).toarray()
        result["m"] = sp.spdiags(m_data, [-1], mult, mult).toarray()
        result["x"] = 0.5 * (result["p"] + result["m"])
        result["y"] = -0.5 * 1j * (result["p"] - result["m"])
        result["z"] = sp.spdiags(prjs, 0, mult, mult).toarray()
    return result


class Constant(float):
    def __new__(cls, details):
        obj = super().__new__(cls, details.pop("value"))
        obj.details = SimpleNamespace(**details)
        return obj

    @staticmethod
    def fromjson(json_file):
        with open(json_file) as f:
            data = json.load(f)
        return SimpleNamespace(**{k: Constant(v) for k, v in data.items()})


class IsotopeOld(str):
    """Interface to isotope database."""

    def __new__(cls, name, data):
        # gamma_unit = ureg("rad/s/T")
        obj = super().__new__(cls, name)
        if "gamma" in data:
            gamma = data.pop("gamma")
            obj.gamma = gamma  # * gamma_unit
            obj.gamma_mT = gamma / 1000
        if "multiplicity" in data:
            multiplicity = data.pop("multiplicity")
            obj.multiplicity = multiplicity
        obj.details = data
        return obj

    @staticmethod
    def fromjson(json_file):
        with open(json_file) as f:
            data = json.load(f)
        return SimpleNamespace(**{k: __class__(k, v) for k, v in data.items()})


class Isotope:
    """Isotope."""

    json_dir = DATA_DIR / "isotopes"

    def __init__(self, name):
        """Constructor."""
        path = (self.json_dir / name).with_suffix(".json")
        if not path.exists():
            raise ValueError("Isotope doesn't exists!")
        with open(path) as f:
            data = json.load(f)
            self.data = data
            self.name = name

    def __repr__(self):
        """Isotope representation."""
        return str(self.data)

    @classmethod
    @property
    def available_isotopes(cls):
        """List isotopes available in the database."""
        paths = cls.json_dir.glob("*.json")
        return [path.with_suffix("").name for path in paths]

    def _get_key(self, key: str) -> Any:
        if key not in self.data:
            raise ValueError(f"Isotope {self.name} has no {key}")
        return self.data[key]

    @property
    def multiplicity(self) -> int:
        """Spin multiplicity of the isotope.

        Returns:
            int: Spin multiplicity.
        """
        return self._get_key("multiplicity")

    @property
    def spin_quantum_number(self) -> float:
        return float(self.multiplicity - 1) / 2.0

    @staticmethod
    def spin2multiplicity():
        """Spin quantum number to multiplicity.

        Args:
                spin (float): Spin quantum number.

        Returns:
                int: Spin multiplicity.

        """

    @staticmethod
    def multiplicity2spin():
        """Spin multiplicity to spin quantum number.

        Args:
                multiplicity (int): Spin multiplicity.

        Returns:
                float: Spin quantum number.

        """


class Molecule:
    """Representation of a molecule for the simulation.

    Args:
        radical (str): the name of the `Molecule`, defaults to `""`

        nuclei (list[str]): list of atoms from the molecule (or from
            the database), defaults to `[]`

        multiplicities (list[int]): list of multiplicities of the
            atoms and their isotopes (when not using the database),
            defaults to `[]`

        gammas_mT (list[float]): list of gyromagnetic ratios of the
            atoms and their isotopes (when not using the database),
            defaults to `[]`

        hfcs (list[float]): list of hyperfine coupling constants of
            the atoms and their isotopes (when not using the
            database), defaults to `[]`

    A molecule is represented by hyperfine coupling constants, spin
    multiplicities and gyromagnetic ratios (gammas, specified in mT)
    of its nuclei.  When using the database, one needs to specify the
    name of the molecule and the list of its nuclei.

    >>> Molecule(radical="adenine_cation",
    ...          nuclei=["N6-H1", "N6-H2"])
    Molecule: adenine_cation
      HFCs: [-0.63, -0.66]
      Multiplicities: [3, 3]
      Magnetogyric ratios (mT): [19337.792, 19337.792]
      Number of particles: 2


    If the wrong molecule name is given, the error helps you find the
    valid options.

    >>> Molecule("foobar", ["H1"])
    Traceback (most recent call last):
    ...
    ValueError: Available molecules below:
    2_6_aqds
    adenine_cation
    flavin_anion
    flavin_neutral
    tryptophan_cation
    tyrosine_neutral

    Similarly, giving a list of incorrect atom names will also result
    in a helpful error message listing the available atoms.

    >>> Molecule("tryptophan_cation", ["buz"])
    Traceback (most recent call last):
    ...
    ValueError: Available nuclei below.
    Hbeta1 (hfc = 1.6045)
    H1 (hfc = -0.5983)
    H4 (hfc = -0.4879)
    H7 (hfc = -0.3634)
    N1 (hfc = 0.32156666666666667)
    H2 (hfc = -0.278)
    N* (hfc = 0.1465)
    Halpha (hfc = -0.09306666666666667)
    Hbeta2 (hfc = 0.04566666666666666)
    H5 (hfc = -0.04)
    H6 (hfc = -0.032133333333333326)

    >>> Molecule("adenine_cation", ["buz"])
    Traceback (most recent call last):
    ...
    ValueError: Available nuclei below.
    N6-H2 (hfc = -0.66)
    N6-H1 (hfc = -0.63)
    C8-H (hfc = -0.55)

    One can also specify a list of custom hyperfine coupling constants
    along with a list of their respective isotope names.

    >>> Molecule(nuclei=["1H", "14N"], hfcs=[0.41, 1.82])
    Molecule: N/A
      HFCs: [0.41, 1.82]
      Multiplicities: [2, 3]
      Magnetogyric ratios (mT): [267522.18744, 19337.792]
      Number of particles: 2

    Same as above, but with an informative molecule name (doesn't
    affect behaviour):

    >>> Molecule("isotopes", nuclei=["15N", "15N"], hfcs=[0.3, 1.7])
    Molecule: isotopes
      HFCs: [0.3, 1.7]
      Multiplicities: [2, 2]
      Magnetogyric ratios (mT): [-27126.180399999997, -27126.180399999997]
      Number of particles: 2

    A molecule with no HFCs, for one proton radical pair simulations
    (for simple simulations -- often with *fantastic* low-field
    effects):

    >>> Molecule("kryptonite")
    Molecule: kryptonite
      HFCs: []
      Multiplicities: []
      Magnetogyric ratios (mT): []
      Number of particles: 0

    Manual input for all relevant values (multiplicities, gammas,
    HFCs):

    >>> Molecule(multiplicities=[2, 2, 3],
    ...          gammas_mT=[267522.18744, 267522.18744, 19337.792],
    ...          hfcs=[0.42, 1.01, 1.33])
    Molecule: N/A
      HFCs: [0.42, 1.01, 1.33]
      Multiplicities: [2, 2, 3]
      Magnetogyric ratios (mT): [267522.18744, 267522.18744, 19337.792]
      Number of particles: 3

    Same as above with an informative molecule name:

    >>> Molecule("my_flavin", multiplicities=[2], gammas_mT=[267522.18744], hfcs=[0.5])
    Molecule: my_flavin
      HFCs: [0.5]
      Multiplicities: [2]
      Magnetogyric ratios (mT): [267522.18744]
      Number of particles: 1
    """

    def __init__(
        self,
        radical: str = "",
        nuclei: list[str] = [],
        multiplicities: list[int] = [],
        gammas_mT: list[float] = [],
        hfcs: list[float] = [],
    ):
        self.radical = radical if radical else "N/A"
        self.nuclei = nuclei
        self.custom_molecule = True
        if nuclei:
            if self._check_molecule_or_spin_db(radical, nuclei):
                self._init_from_molecule_db(radical, nuclei)
            else:
                self._init_from_spin_db(radical, nuclei, hfcs)
        else:
            if self._check_molecule_or_spin_db(radical, nuclei):
                self._init_from_molecule_db(radical, nuclei)
            else:
                self.multiplicities = multiplicities
                self.gammas_mT = gammas_mT
                self.hfcs = hfcs
        if self.hfcs and isinstance(self.hfcs[0], list):
            self.hfcs = [np.array(h) for h in self.hfcs]
        assert len(self.multiplicities) == self.num_particles
        assert len(self.gammas_mT) == self.num_particles
        assert len(self.hfcs) == self.num_particles

    def __repr__(self) -> str:
        """Pretty print the molecule.

        Returns:
            str: Representation of a molecule.
        """
        return (
            f"Molecule: {self.radical}"
            # f"\n  Nuclei: {self.nuclei}"
            f"\n  HFCs: {self.hfcs}"
            f"\n  Multiplicities: {self.multiplicities}"
            f"\n  Magnetogyric ratios (mT): {self.gammas_mT}"
            f"\n  Number of particles: {self.num_particles}"
            # f"\n  elements: {self.elements}"
        )

    def _check_molecule_or_spin_db(self, radical, nuclei):
        if radical in MOLECULE_DATA:
            self._check_nuclei(nuclei)
            return True
        else:
            # TODO: needs to fail with nuclei == [] + wrong molecule
            # name
            if all(n in SPIN_DATA for n in nuclei):
                return False
            else:
                available = "\n".join(get_molecules().keys())
                raise ValueError(f"Available molecules below:\n{available}")

    def _check_nuclei(self, nuclei: list[str]) -> None:
        molecule_data = MOLECULE_DATA[self.radical]["data"]
        for nucleus in nuclei:
            if nucleus not in molecule_data:
                keys = molecule_data.keys()
                hfcs = [molecule_data[k]["hfc"] for k in keys]
                hfcs = [
                    isotropic(np.array(h)) if isinstance(h, list) else h for h in hfcs
                ]
                pairs = sorted(
                    zip(keys, hfcs), key=lambda t: np.abs(t[1]), reverse=True
                )
                available = "\n".join([f"{k} (hfc = {h})" for k, h in pairs])
                raise ValueError(f"Available nuclei below.\n{available}")

    def _init_from_molecule_db(self, radical: str, nuclei: list[str]) -> None:
        data = MOLECULE_DATA[radical]["data"]
        elem = [data[n]["element"] for n in nuclei]
        self.radical = radical
        self.gammas_mT = [gamma_mT(e) for e in elem]
        self.multiplicities = [multiplicity(e) for e in elem]
        self.hfcs = [data[n]["hfc"] for n in nuclei]
        self.custom_molecule = False

    def _init_from_spin_db(
        self, radical: str, nuclei: list[str], hfcs: list[float]
    ) -> None:
        self.multiplicities = [multiplicity(e) for e in nuclei]
        self.gammas_mT = [gamma_mT(e) for e in nuclei]
        self.hfcs = hfcs

    @property
    def effective_hyperfine(self) -> float:
        if self.custom_molecule:
            multiplicities = self.multiplicities
            hfcs = self.hfcs
        else:
            # TODO: this can fail with wrong molecule name
            data = MOLECULE_DATA[self.radical]["data"]
            nuclei = list(data.keys())
            elem = [data[n]["element"] for n in nuclei]
            multiplicities = [multiplicity(e) for e in elem]
            hfcs = [data[n]["hfc"] for n in nuclei]

        # spin quantum number
        s = np.array(list(map(spin_quantum_number, multiplicities)))
        hfcs = [isotropic(h) if isinstance(h, list) else h for h in hfcs]
        hfcs = np.array(hfcs)
        return np.sqrt((4 / 3) * sum((hfcs**2 * s) * (s + 1)))

    @property
    def num_particles(self) -> int:
        """Return the number of isotopes in the molecule."""
        return len(self.multiplicities)


constants = Constant.fromjson(CONSTANTS_JSON)
isotopes = IsotopeOld.fromjson(SPIN_DATA_JSON)
