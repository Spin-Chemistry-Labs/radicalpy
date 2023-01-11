#! /usr/bin/env python
import json
from functools import cache, singledispatchmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def spin_to_multiplicity(spin: float) -> int:
    """Spin quantum number to multiplicity.

    Args:
            spin (float): Spin quantum number.

    Returns:
            int: Spin multiplicity.

    """
    if int(2 * spin) != 2 * spin:
        raise ValueError("Spin needs to be half of an integer.")
    return int(2 * spin) + 1


def multiplicity_to_spin(multiplicity: int) -> float:
    """Spin multiplicity to spin quantum number.

    Args:
            multiplicity (int): Spin multiplicity.

    Returns:
            float: Spin quantum number.

    """
    return float(multiplicity - 1) / 2.0


DATA_DIR = Path(__file__).parent / "data"
SPIN_DATA_JSON = DATA_DIR / "spin_data.json"
MOLECULES_DIR = DATA_DIR / "molecules"

with open(SPIN_DATA_JSON, encoding="utf-8") as file:
    SPIN_DATA = json.load(file)
    """Dictionary containing spin data for elements.

    :meta hide-value:"""


def get_molecules(molecules_dir=MOLECULES_DIR):
    """Delete this."""
    molecules = {}
    for json_path in sorted(molecules_dir.glob("*.json")):
        molecule_name = json_path.with_suffix("").name
        with open(json_path, encoding="utf-8") as f:
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
    """Return the `multiplicity` value of an element."""
    return SPIN_DATA[element]["multiplicity"]


class Constant(float):
    """Constan class.

    Extends float with the `Constant.details` member.
    """

    details: SimpleNamespace
    """Details (e.g. units) of the constant."""

    def __new__(cls, details: dict):  # noqa D102
        obj = super().__new__(cls, details.pop("value"))
        obj.details = SimpleNamespace(**details)
        return obj

    @staticmethod
    def fromjson(json_file: Path) -> SimpleNamespace:
        """Read all constants from the JSON file.

        Args:
            json_file (str)

        Returns:
            SimpleNamespace: A namespace containing all constants.
        """
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        return SimpleNamespace(**{k: Constant(v) for k, v in data.items()})


class Isotope:
    """Class representing an isotope.

    Examples:
    >>> E = Isotope("E")
    >>> E
    Symbol: E
    Multiplicity: 2
    Gamma: -176085963023.0
    Details: {'name': 'Electron', 'source': 'CODATA 2018'}
    >>> E.multiplicity
    2
    >>> E.details
    {'name': 'Electron', 'source': 'CODATA 2018'}
    """

    def __repr__(self) -> str:  # noqa D105
        lines = [
            f"Symbol: {self.symbol}",
            f"Multiplicity: {self.multiplicity}",
            f"Gamma: {self.gamma}",
            f"Details: {self.details}",
        ]
        return "\n".join(lines)

    @staticmethod
    @cache
    def _load_data() -> dict:
        with open(DATA_DIR / "spin_data.json", encoding="utf-8") as f:
            return json.load(f)

    def __init__(self, symbol: str):  # noqa D105
        isotopes_data = self._load_data()
        isotope = dict(isotopes_data[symbol])
        self.symbol = symbol
        self.multiplicity = isotope.pop("multiplicity")
        self.gamma = isotope.pop("gamma")
        self.details = isotope

    @property
    def gamma_mT(self):
        """Return gamma value in mT."""
        return self.gamma * 0.001

    @classmethod
    def available(cls) -> list[str]:
        """List isotopes available in the database.

        Returns:
            list[str]: List of available isotopes (symbols).

        Example:

        >>> available = Isotope.available()
        >>> available[-5:]
        ['E', 'G', 'M', 'N', 'P']

        >>> Isotope(available[-5])
        Symbol: E
        Multiplicity: 2
        Gamma: -176085963023.0
        Details: {'name': 'Electron', 'source': 'CODATA 2018'}

        >>> Isotope(available[-2])
        Symbol: N
        Multiplicity: 2
        Gamma: -183247171.0
        Details: {'name': 'Neutron', 'source': 'CODATA 2018'}

        """
        items = cls._load_data().items()
        return sorted([k for k, v in items if "multiplicity" in v and "gamma" in v])

    @property
    def spin_quantum_number(self) -> float:
        """Spin quantum numer of `Isotope`."""
        return multiplicity_to_spin(self.multiplicity)


class Hfc:
    """The Hfc class represents isotropic and anisotropic HFC values.

    Args:
        hfc (float or list[list[float]]): The HFC value.  In case of a
            single `float`, only the isotropic value is set.  In case
            of a 3x3 matrix both the isotropic and anisotropic values
            are stored.

    Examples:
    >>> with open(DATA_DIR/"molecules/flavin_anion.json") as f:
    ...      flavin_dict = json.load(f)
    >>> hfc_3d_data = flavin_dict["data"]["N5"]["hfc"]
    >>> hfc_3d_obj = Hfc(hfc_3d_data)
    >>> hfc_3d_obj
    0.5141 <anisotropic available>
    >>> hfc_3d_obj.isotropic
    0.5141406139911681
    >>> hfc_3d_obj.anisotropic
    array([[-0.06819637,  0.01570029,  0.08701531],
           [ 0.01570029, -0.03652102,  0.27142597],
           [ 0.08701531,  0.27142597,  1.64713923]])

    >>> with open(DATA_DIR/"molecules/adenine_cation.json") as f:
    ...      adenine_dict = json.load(f)
    >>> hfc_1d_data = adenine_dict["data"]["N6-H1"]["hfc"]
    >>> hfc_1d_obj = Hfc(hfc_1d_data)
    >>> hfc_1d_obj
    -0.63 <anisotropic not available>
    >>> hfc_1d_obj.isotropic
    -0.63
    >>> hfc_1d_obj.anisotropic
    Traceback (most recent call last):
    ...
    ValueError: The molecule doesn't support anisotropic HFCs
    """

    _anisotropic: Optional[NDArray]
    """Optional anisotropic HFC value."""

    isotropic: float
    """Isotropic HFC value."""

    def __repr__(self) -> str:  # noqa D105
        available = "not " if self._anisotropic is None else ""
        return f"{self.isotropic:.4} <anisotropic {available}available>"

    @singledispatchmethod
    def __init__(self, hfc: list[list[float]]):  # noqa D105
        self._anisotropic = np.array(hfc)
        if self._anisotropic.shape != (3, 3):
            raise ValueError("Anisotropic HFCs should be a float or a 3x3 matrix!")
        self.isotropic = self._anisotropic.trace() / 3

    @__init__.register
    def _(self, hfc: float):  # noqa D105
        self._anisotropic = None
        self.isotropic = hfc

    @property
    def anisotropic(self) -> NDArray:
        """Anisotropic value if available.

        Returns:
            NDarray: The anisotropic HFC values.
        """
        if self._anisotropic is None:
            raise ValueError("The molecule doesn't support anisotropic HFCs")
        return self._anisotropic


class Nucleus:
    """A nucleus in a molecue.

    >>> Nucleus.fromisotope("1H", Hfc(1.1))
    Nucleus(267522187.44, 2, 1.1 <anisotropic not available>)
    >>> Nucleus(1.0, 2, Hfc(3.0))
    Nucleus(1.0, 2, 3.0 <anisotropic not available>)
    """

    gamma: float
    multiplicity: int
    hfc: Hfc

    def __repr__(self) -> str:  # noqa D105
        return f"Nucleus({self.gamma}, {self.multiplicity}, {self.hfc})"

    def __init__(self, gamma, multiplicity, hfc):  # noqa D105
        self.gamma = gamma
        self.multiplicity = multiplicity
        self.hfc = hfc

    @classmethod
    def fromisotope(cls, isotope: str, hfc: Hfc):
        """Construct a `Nucleus` from an `Isotope`."""
        iso = Isotope(isotope)
        return cls(iso.gamma, iso.multiplicity, hfc)

    @property
    def gamma_mT(self):
        """Return gamma value in mT."""
        return self.gamma * 0.001


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

    Examples:
    >>> Molecule(radical="adenine_cation",
    ...          nuclei=["N6-H1", "N6-H2"])
    Molecule: adenine_cation
      HFCs: [-0.63 <anisotropic not available>, -0.66 <anisotropic not available>]
      Multiplicities: [3, 3]
      Magnetogyric ratios (mT): [19337.792, 19337.792]
      Number of particles: 2


    If the wrong molecule name is given, the error helps you find the
    valid options (the second argument `nuclei` must not be empty).

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
      HFCs: [0.41 <anisotropic not available>, 1.82 <anisotropic not available>]
      Multiplicities: [2, 3]
      Magnetogyric ratios (mT): [267522.18744, 19337.792]
      Number of particles: 2

    Same as above, but with an informative molecule name (doesn't
    affect behaviour):

    >>> Molecule("isotopes", nuclei=["15N", "15N"], hfcs=[0.3, 1.7])
    Molecule: isotopes
      HFCs: [0.3 <anisotropic not available>, 1.7 <anisotropic not available>]
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

    Note: If the second argument (`nuclei`) is empty, no error is
    triggered, you can get the look up the available molecules
    with `Molecule.available`.

    Manual input for all relevant values (multiplicities, gammas,
    HFCs):

    >>> Molecule(multiplicities=[2, 2, 3],
    ...          gammas_mT=[267522.18744, 267522.18744, 19337.792],
    ...          hfcs=[0.42, 1.01, 1.33])
    Molecule: N/A
      HFCs: [0.42 <anisotropic not available>, 1.01 <anisotropic not available>, 1.33 <anisotropic not available>]
      Multiplicities: [2, 2, 3]
      Magnetogyric ratios (mT): [267522.18744, 267522.18744, 19337.792]
      Number of particles: 3

    Same as above with an informative molecule name:

    >>> Molecule("my_flavin", multiplicities=[2], gammas_mT=[267522.18744], hfcs=[0.5])
    Molecule: my_flavin
      HFCs: [0.5 <anisotropic not available>]
      Multiplicities: [2]
      Magnetogyric ratios (mT): [267522.18744]
      Number of particles: 1

    """

    def __repr__(self) -> str:  # noqa D105
        return (
            f"Molecule: {self.radical}"
            # f"\n  Nuclei: {self.nuclei}"
            f"\n  HFCs: {self.hfcs}"
            f"\n  Multiplicities: {self.multiplicities}"
            f"\n  Magnetogyric ratios (mT): {self.gammas_mT}"
            f"\n  Number of particles: {self.num_particles}"
            # f"\n  elements: {self.elements}"
        )

    def __init__(  # noqa D105
        self,
        radical: str = "",
        nuclei: list[str] = [],
        multiplicities: list[int] = [],
        gammas_mT: list[float] = [],
        hfcs: list[float] = [],
    ):
        self.radical = radical if radical else "N/A"
        self.nuclei = nuclei
        self.custom_molecule = True  # todo(vatai): use info instead of this
        if self._check_molecule_or_spin_db(radical, nuclei):
            self._init_from_molecule_db(radical, nuclei)
        else:
            if nuclei:
                self._init_from_spin_db(nuclei, hfcs)
            else:
                self.multiplicities = multiplicities
                self.gammas_mT = gammas_mT
                self.hfcs = [Hfc(h) for h in hfcs]
        assert len(self.multiplicities) == self.num_particles
        assert len(self.gammas_mT) == self.num_particles
        assert len(self.hfcs) == self.num_particles

    def _check_molecule_or_spin_db(self, radical, nuclei):
        if radical in self.available():
            self._check_nuclei(nuclei)
            return True
        # To error on creating an empty (no nuclei) molecule with
        # a custom name, modify the line below to include the
        # comment. Lewis said it's okay like this.
        if all(n in Isotope.available() for n in nuclei):  # and nuclei != []:
            return False
        available = "\n".join(Molecule.available())
        raise ValueError(f"Available molecules below:\n{available}")

    def _check_nuclei(self, nuclei: list[str]) -> None:  # raises ValueError()
        molecule_data = MOLECULE_DATA[self.radical]["data"]
        for nucleus in nuclei:
            if nucleus not in molecule_data:
                keys = molecule_data.keys()
                hfcs = [molecule_data[k]["hfc"] for k in keys]
                hfcs = [Hfc(h).isotropic for h in hfcs]
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
        self.hfcs = [Hfc(data[n]["hfc"]) for n in nuclei]
        self.custom_molecule = False

    def _init_from_spin_db(self, nuclei: list[str], hfcs: list[float]) -> None:
        self.multiplicities = [multiplicity(e) for e in nuclei]
        self.gammas_mT = [gamma_mT(e) for e in nuclei]
        self.hfcs = [Hfc(h) for h in hfcs]

    @classmethod
    def available(cls):
        """List molecules available in the database.

        Returns:
            list[str]: List of available molecules (names).

        Example:

        >>> available = Molecule.available()
        >>> available[:10]
        ['2_6_aqds', 'adenine_cation', 'flavin_anion', 'flavin_neutral', 'tryptophan_cation', 'tyrosine_neutral']

        """
        paths = (DATA_DIR / "molecules").glob("*.json")
        return sorted([path.with_suffix("").name for path in paths])

    @property
    def effective_hyperfine(self) -> float:
        """Effective hyperfine for the entire molecule."""
        if self.custom_molecule:
            multiplicities = self.multiplicities
            hfcs = self.hfcs
        else:
            # TODO: this can fail with wrong molecule name
            data = MOLECULE_DATA[self.radical]["data"]
            nuclei = list(data.keys())
            elem = [data[n]["element"] for n in nuclei]
            multiplicities = [multiplicity(e) for e in elem]
            hfcs = [Hfc(data[n]["hfc"]) for n in nuclei]

        # spin quantum number
        spns_np = np.array(list(map(multiplicity_to_spin, multiplicities)))
        hfcs_np = np.array([h.isotropic for h in hfcs])
        return np.sqrt((4 / 3) * sum((hfcs_np**2 * spns_np) * (spns_np + 1)))

    @property
    def num_particles(self) -> int:
        """Return the number of isotopes in the molecule."""
        return len(self.multiplicities)


constants = Constant.fromjson(DATA_DIR / "constants.json")
