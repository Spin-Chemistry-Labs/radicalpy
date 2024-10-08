#! /usr/bin/env python
import json
from functools import singledispatchmethod
from typing import Iterator, Optional, Tuple

import numpy as np
import scipy as sp
from importlib_resources import files
from importlib_resources.abc import Traversable
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


def get_data(suffix: str = "") -> Traversable:
    """Get the directory containing data files."""
    return files(__package__) / "data_files" / suffix


class Isotope:
    """Class representing an isotope.

    Args:
        symbol (str): The symbol/identifier of the isotope in the
            database.

    Examples:
        Create an isotope using the database.

        >>> E = Isotope("E")
        >>> E
        Symbol: E
        Multiplicity: 2
        Magnetogyric ratio: -176085963023.0
        Details: {'name': 'Electron', 'source': 'CODATA 2018'}

        Query the multiplicity:

        >>> E.multiplicity
        2

        Query other details:

        >>> E.details
        {'name': 'Electron', 'source': 'CODATA 2018'}
    """

    _isotope_data: Optional[dict] = None

    def __repr__(self) -> str:  # noqa D105
        """Isotope representation."""
        lines = [
            f"Symbol: {self.symbol}",
            f"Multiplicity: {self.multiplicity}",
            f"Magnetogyric ratio: {self.magnetogyric_ratio}",
            f"Details: {self.details}",
        ]
        return "\n".join(lines)

    def __init__(self, symbol: str):  # noqa D105
        """Isotope constructor."""
        self._ensure_isotope_data()
        if not self._isotope_data or symbol not in self._isotope_data:
            raise ValueError(
                f"Isotpoe {symbol} not in database. See `Isotope.available()`"
            )
        isotope = dict(self._isotope_data[symbol])
        self.symbol = symbol
        self.multiplicity = isotope.pop("multiplicity")
        self.magnetogyric_ratio = isotope.pop("gamma")
        self.details = isotope

    @classmethod
    def _ensure_isotope_data(cls):
        if cls._isotope_data is None:
            with open(get_data() / "spin_data.json", encoding="utf-8") as f:
                cls._isotope_data = json.load(f)

    @classmethod
    def available(cls) -> list[str]:
        """List isotopes available in the database.

        Returns:
            list[str]: List of available isotopes (symbols).

        Examples:
            >>> available = Isotope.available()
            >>> available[:5]
            ['G', 'E', 'N', 'M', 'P']

            >>> Isotope(available[1])
            Symbol: E
            Multiplicity: 2
            Magnetogyric ratio: -176085963023.0
            Details: {'name': 'Electron', 'source': 'CODATA 2018'}

            >>> Isotope(available[2])
            Symbol: N
            Multiplicity: 2
            Magnetogyric ratio: -183247171.0
            Details: {'name': 'Neutron', 'source': 'CODATA 2018'}
        """
        cls._ensure_isotope_data()
        items = cls._isotope_data.items()
        return [k for k, v in items if "multiplicity" in v and "gamma" in v]

    @property
    def gamma_mT(self):
        """Return gamma value in rad/s/mT."""
        return self.magnetogyric_ratio * 0.001

    @property
    def spin_quantum_number(self) -> float:
        """Spin quantum numer of `Isotope`."""
        return multiplicity_to_spin(self.multiplicity)


class Hfc:
    """The Hfc class represents isotropic and anisotropic HFC values.

    Args:
        hfc (float | list[list[float]]): The HFC value.  In case of a
            single `float`, only the isotropic value is set.  In case
            of a 3x3 matrix both the isotropic and anisotropic values
            are stored.

    Examples:
        Initialising the HFC with a 3-by-3 matrix (list of lists):

        >>> with open(get_data("molecules/flavin_anion.json"), encoding="utf-8") as f:
        ...      flavin_dict = json.load(f)
        >>> hfc_3d_data = flavin_dict["data"]["N5"]["hfc"]
        >>> hfc_3d_obj = Hfc(hfc_3d_data)
        >>> hfc_3d_obj
        0.5141 <anisotropic available>

        we can obtain both the isotropic value:

        >>> hfc_3d_obj.isotropic
        0.5141406139911681

        and the anisotropic tensor:

        >>> hfc_3d_obj.anisotropic
        array([[-0.06819637,  0.01570029,  0.08701531],
               [ 0.01570029, -0.03652102,  0.27142597],
               [ 0.08701531,  0.27142597,  1.64713923]])

        Initialising the HFC with a single float:

        >>> with open(get_data("molecules/adenine_cation.json"), encoding="utf-8") as f:
        ...      adenine_dict = json.load(f)
        >>> hfc_1d_data = adenine_dict["data"]["N6-H1"]["hfc"]
        >>> hfc_1d_obj = Hfc(hfc_1d_data)
        >>> hfc_1d_obj
        -0.63 <anisotropic not available>

        we can obtain both the isotropic value:

        >>> hfc_1d_obj.isotropic
        -0.63

        but not the anisotropic tensor:

        >>> hfc_1d_obj.anisotropic
        Traceback (most recent call last):
        ...
        ValueError: No anisotropic HFC data available.
    """

    _anisotropic: Optional[NDArray]
    _isotropic: float

    def __repr__(self) -> str:  # noqa D105
        available = "not " if self._anisotropic is None else ""
        return f"{self.isotropic:.4} <anisotropic {available}available>"

    # `singledispatchmethod` and `__init__.register` is used to have
    # one `__init__` function with two implementations BASED ON THE
    # ARGUMENT TYPE!
    @singledispatchmethod
    def __init__(self, hfc: list[list[float]]):  # noqa D105
        self._anisotropic = np.array(hfc)
        if self._anisotropic.shape != (3, 3):
            lines = [
                "Anisotropic HFCs should be a float or a 3x3 matrix!",
                f"Got: {hfc}",
            ]
            raise ValueError("\n".join(lines))
        self._isotropic = self._anisotropic.trace() / 3

    # See the comment above the `singledispatchmethod` decorator.
    @__init__.register
    def _(self, hfc: float):  # noqa D105
        self._anisotropic = None
        self._isotropic = hfc

    @property
    def anisotropic(self) -> NDArray:
        """Anisotropic value if available.

        Returns:
            NDarray: The anisotropic HFC values.
        """
        if self._anisotropic is None:
            raise ValueError("No anisotropic HFC data available.")
        return self._anisotropic

    @property
    def isotropic(self) -> float:
        """Isotropic value.

        Returns:
            float: The isotropic HFC value.
        """
        return float(self._isotropic)


class Nucleus:
    """A nucleus in a molecule.

    Construct a nucleus from an `Isotope` and an `Hfc`.

    >>> Nucleus.fromisotope("1H", 1.1)
    1H(267522187.44, 2, 1.1 <anisotropic not available>)

    The default constructor needs the magnetogyric ratio, the
    multiplicity and the HFC values.

    >>> Nucleus(0.001, 2, Hfc(3.0))
    Nucleus(1.0, 2, 3.0 <anisotropic not available>)

    Additionally a name can also be added.

    >>> Nucleus(0.001, 2, Hfc(3.0), "Adamantium")
    Adamantium(1.0, 2, 3.0 <anisotropic not available>)
    """

    magnetogyric_ratio: float
    multiplicity: int
    hfc: Hfc
    name: Optional[str]

    def __repr__(self) -> str:  # noqa D105
        name = self.name if self.name else "Nucleus"
        return f"{name}({self.magnetogyric_ratio}, {self.multiplicity}, {self.hfc})"

    def __init__(
        self,
        magnetogyric_ratio: float,
        multiplicity: int,
        hfc: Hfc,
        name: Optional[str] = None,
    ):
        """Nucleus constructor."""
        self.magnetogyric_ratio = 1000 * magnetogyric_ratio  # gamma_mT
        self.multiplicity = multiplicity
        self.hfc = hfc
        self.name = name

    @classmethod
    def fromisotope(cls, isotope: str, hfc: float | list[list[float]]):
        """Construct a `Nucleus` from an `Isotope`.

        Args:
            isotope (str): Name/symbol of the `Isotope`.
            hfc (float | list[list[float]]): The HFC value (see `Hfc`
                class).

        Returns:
            Nucleus: A nucleus with magnetogyric ratio, multiplicity
                and name determined by the `isotope` and the `hfc`
                value.

        """
        iso = Isotope(isotope)
        nucleus = cls(
            iso.magnetogyric_ratio / 1000, iso.multiplicity, Hfc(hfc)
        )  # gamma_mT
        nucleus.name = isotope
        return nucleus

    @property
    def pauli(self):
        """Generate Pauli matrices.

        Generates the Pauli matrices corresponding to a given multiplicity.

        Args:

            mult (int): The multiplicity of the element.

        Returns:
            dict:

                A dictionary containing 6 `np.array` matrices of
                shape `(mult, mult)`:

                - the unit operator `result["u"]`,
                - raising operator `result["p"]`,
                - lowering operator `result["m"]`,
                - Pauli matrix for x axis `result["x"]`,
                - Pauli matrix for y axis `result["y"]`,
                - Pauli matrix for z axis `result["z"]`.
        """
        mult = self.multiplicity
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
            result["p"] = sp.sparse.spdiags(p_data, [1], mult, mult).toarray()
            result["m"] = sp.sparse.spdiags(m_data, [-1], mult, mult).toarray()
            result["x"] = 0.5 * (result["p"] + result["m"])
            result["y"] = -0.5 * 1j * (result["p"] - result["m"])
            result["z"] = sp.sparse.spdiags(prjs, 0, mult, mult).toarray()
        return result

    @property
    def gamma_mT(self):
        r"""Return magnetogyric ratio, :math:`\gamma` (rad/s/mT)."""
        return self.magnetogyric_ratio * 0.001

    @property
    def spin_quantum_number(self) -> float:
        """Spin quantum numer of `Isotope`."""
        return multiplicity_to_spin(self.multiplicity)


class Molecule:
    """Representation of a molecule for the simulation.

    A molecule is described by a name and a list of nuclei (see
    `Nucleus`).  Using the default constructor is **cumbersome and not
    recommended**.  The preferred way is to use the following
    convenience methods (click on the method name to see its
    documentation):

    - `Molecule.fromdb`
    - `Molecule.fromisotopes`

    Args:

        name (str): The name of the `Molecule`.

        nuclei (list[Nucleus]): List of nuclei/atoms which should be
            simulated (see `Nucleus`).

        radical (Nucleus): The radical of the molecule. (Default
            `Nucleus.fromisotope("E", 0.0)`).

        info (dict[str, str]): Dictionary of miscellaneous information
            about the molecule.

    Examples:
        The default constructor takes an arbitrary name and a list of
        molecules to construct a molecule.

        >>> gamma_1H = 267522.187
        >>> gamma_14N = 19337.792
        >>> Molecule("kryptonite", [Nucleus(gamma_1H, 2, Hfc(1.0), "Hydrogen"),
        ...                            Nucleus(gamma_14N, 3, Hfc(-0.5), "Nitrogen")])
        Molecule: kryptonite
        Nuclei:
          Hydrogen(267522186.99999997, 2, 1.0 <anisotropic not available>)
          Nitrogen(19337792.0, 3, -0.5 <anisotropic not available>)
        Radical: E(-176085963023.0, 2, 0.0 <anisotropic not available>)

        Or alternatively:

        >>> gammas = [267522.187, 19337.792]
        >>> multis = [2, 3]
        >>> hfcs = [1.0, -0.5]
        >>> names = ["Hydrogen", "Nitrogen"]
        >>> params = zip(gammas, multis, map(Hfc, hfcs), names)
        >>> Molecule("kryptonite", [Nucleus(*param) for param in params])
        Molecule: kryptonite
        Nuclei:
          Hydrogen(267522186.99999997, 2, 1.0 <anisotropic not available>)
          Nitrogen(19337792.0, 3, -0.5 <anisotropic not available>)
        Radical: E(-176085963023.0, 2, 0.0 <anisotropic not available>)

    """

    name: str
    nuclei: list[Nucleus]
    info: dict[str, str]
    radical: Nucleus
    custom: bool

    def __repr__(self) -> str:
        """Pretty print the molecule.

        Returns:
            str: Representation of a molecule.
        """
        nuclei = "\n".join([f"  {n}" for n in self.nuclei])
        lines = [
            f"Molecule: {self.name}",
            f"Nuclei:\n{nuclei}" if self.nuclei else "No nuclei specified.",
            f"Radical: {self.radical}",
            # f"\n  Number of particles: {self.num_particles}"
        ]
        if self.info:
            lines.append(f"Info: {self.info}")
        return "\n".join(lines)

    def __init__(
        self,
        name: str = "",
        nuclei: list[Nucleus] = [],
        radical: Nucleus = Nucleus.fromisotope("E", 0.0),
        info: dict[str, str] = {},
    ):
        """Default constructor."""
        # todo(vatai): check types?
        self.name = name
        self.nuclei = nuclei  # list[gamma, multi, hfc]
        self.info = info
        self.radical = radical
        self.custom = True

        self.hfc_rng = np.random.default_rng(42)  ##################
        self.ang_rng = np.random.default_rng(43)  ##################

    @classmethod
    def load_molecule_json(cls, molecule: str) -> dict:
        json_path = get_data(f"molecules/{molecule}.json")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data

    @classmethod
    def available(cls):
        """List molecules available in the database.

        Returns:
            list[str]: List of available molecules (names).

        Examples:
            >>> available = Molecule.available()
            >>> available[:4]
            ['2_6_aqds', 'adenine_cation', 'fad', 'flavin_anion']
        """
        paths = get_data("molecules").glob("*.json")
        return sorted([path.with_suffix("").name for path in paths])

    @classmethod
    def _check_molecule_available(cls, name):
        if name not in cls.available():
            lines = [f"Molecule `{name}` not found in database."]
            lines += ["Available molecules:"]
            lines += cls.available()
            raise ValueError("\n".join(lines))

    @classmethod
    def all_nuclei(cls, name: str):
        """Construct a molecule from the database with all nuclei.

        Args:
            name (str): A name of the molecule available in the
                database (see `Molecule.available()`).

        Examples:
            >>> Molecule.all_nuclei("flavin_anion")
            Molecule: flavin_anion
            Nuclei:
              14N(19337792.0, 3, 0.5141 <anisotropic available>)
              14N(19337792.0, 3, -0.001275 <anisotropic available>)
              14N(19337792.0, 3, -0.03654 <anisotropic available>)
              1H(267522187.44, 2, 0.05075 <anisotropic available>)
              1H(267522187.44, 2, -0.1371 <anisotropic available>)
              1H(267522187.44, 2, -0.1371 <anisotropic available>)
              1H(267522187.44, 2, -0.1371 <anisotropic available>)
              1H(267522187.44, 2, -0.4403 <anisotropic available>)
              1H(267522187.44, 2, 0.4546 <anisotropic available>)
              1H(267522187.44, 2, 0.4546 <anisotropic available>)
              1H(267522187.44, 2, 0.4546 <anisotropic available>)
              1H(267522187.44, 2, 0.009597 <anisotropic available>)
              1H(267522187.44, 2, 0.4263 <anisotropic available>)
              1H(267522187.44, 2, 0.4233 <anisotropic available>)
              1H(267522187.44, 2, -0.02004 <anisotropic available>)
              14N(19337792.0, 3, 0.1784 <anisotropic available>)
            Radical: E(-176085963023.0, 2, 0.0 <anisotropic not available>)
            Info: {'units': 'mT', 'name': 'Flavin radical anion'}
        """
        cls._check_molecule_available(name)
        molecule_json = cls.load_molecule_json(name)
        info = molecule_json["info"]
        data = molecule_json["data"]
        nuclei_list = []
        for nucleus in data.keys():
            isotope = data[nucleus]["element"]
            hfc = data[nucleus]["hfc"]
            nuclei_list.append(Nucleus.fromisotope(isotope, hfc))
        molecule = cls(name=name, nuclei=nuclei_list, info=info)
        molecule.custom = False
        return molecule

    @classmethod
    def fromdb(cls, name: str, nuclei: list[str] = []):
        """Construct a molecule from the database.

        Args:
            name (str): A name of the molecule available in the
                database (see `Molecule.available()`).
            nuclei (list[str]): A list of nuclei names found in the
                molecule specified by `name`.

        Examples:
            >>> Molecule.fromdb("flavin_anion", nuclei=["N14"])
            Molecule: flavin_anion
            Nuclei:
              14N(19337792.0, 3, -0.001275 <anisotropic available>)
            Radical: E(-176085963023.0, 2, 0.0 <anisotropic not available>)
            Info: {'units': 'mT', 'name': 'Flavin radical anion'}
        """
        cls._check_molecule_available(name)
        molecule_json = cls.load_molecule_json(name)
        info = molecule_json["info"]
        data = molecule_json["data"]
        keys = list(data.keys())
        nuclei_list = []
        for nucleus in nuclei:
            if nucleus not in keys:
                nuclei_hfcs = sorted(
                    zip(keys, [Hfc(data[n]["hfc"]).isotropic for n in keys]),
                    key=lambda t: np.abs(t[1]),
                    reverse=True,
                )
                lines = ["Available nuclei:"]
                lines += [f"{n} (hfc = {h})" for n, h in nuclei_hfcs]
                raise ValueError("\n".join(lines))
            isotope = data[nucleus]["element"]
            hfc = data[nucleus]["hfc"]
            nuclei_list.append(Nucleus.fromisotope(isotope, hfc))
        molecule = cls(name=name, nuclei=nuclei_list, info=info)
        molecule.custom = False
        return molecule

    @classmethod
    def fromisotopes(cls, isotopes: list[str], hfcs: list, name: str = ""):
        """Construct molecule from isotopes.

        Args:
            isotopes (list[str]): A list of `Isotope` names found in
                the isotope database.
            hfcs (list): A list of HFC values (see `Hfc` constructor).
            name (str): An optional name for the molecule.

        Examples:
            >>> Molecule.fromisotopes(isotopes=["1H", "14N"],
            ...                          hfcs=[1.5, 0.9],
            ...                          name="kryptonite")
            Molecule: kryptonite
            Nuclei:
              1H(267522187.44, 2, 1.5 <anisotropic not available>)
              14N(19337792.0, 3, 0.9 <anisotropic not available>)
            Radical: E(-176085963023.0, 2, 0.0 <anisotropic not available>)
        """
        isos = []
        for iso in isotopes:
            if iso not in Isotope.available():
                raise ValueError(
                    f"Isotope {iso} not in database! See `Isotope.available()`"
                )
            isos.append(iso)
        nuclei = [Nucleus.fromisotope(i, h) for i, h in zip(isos, hfcs)]
        return cls(name, nuclei)

    @property
    def effective_hyperfine(self) -> float:
        """Effective hyperfine for the entire molecule."""
        if self.custom:
            multiplicities = [n.multiplicity for n in self.nuclei]
            hfcs = [n.hfc for n in self.nuclei]
        else:
            # TODO: this can fail with wrong molecule name
            data = self.load_molecule_json(self.name)["data"]
            nuclei = list(data.keys())
            elem = [data[n]["element"] for n in nuclei]
            multiplicities = [Isotope(e).multiplicity for e in elem]
            hfcs = [Hfc(data[n]["hfc"]) for n in nuclei]

        # spin quantum number
        spns_np = np.array(list(map(multiplicity_to_spin, multiplicities)))
        hfcs_np = np.array([h.isotropic for h in hfcs])
        return np.sqrt((4 / 3) * sum((hfcs_np**2 * spns_np) * (spns_np + 1)))

    @property  ############ TODO(calc only once)
    def semiclassical_std(self) -> float:
        r"""The standard deviation for the semiclassical HFCs.

        Calculate the standard deviation :math:`\sigma` where

        .. math::
           \sigma = \sqrt{\frac{2}{\tau^2}}

        and

        .. math::
           \tau_i^{-2} = \frac{1}{6} \sum_k a_k^2 I_k (I_k + 1)

        where :math:`a_k` is the hyperfine coupling and :math:`I_k`
        the spin quantum number of each nucleus, respectively.

        Examples:
            >>> m = Molecule.fromdb("flavin_anion", nuclei=["N14"])
            >>> m.semiclassical_std
            7.663920853309001e-07

        .. todo::
           reference
        """
        tau = 6 / sum(
            n.spin_quantum_number * (n.spin_quantum_number + 1) * n.hfc.isotropic**2
            for n in self.nuclei
        )
        return float(np.sqrt(2) / tau)


class Triplet(Molecule):
    def __init__(self):
        gamma = Isotope("E").gamma_mT
        triplet = Nucleus(magnetogyric_ratio=gamma, multiplicity=3, hfc=0.0)
        super().__init__(name="Triplet", nuclei=[], radical=triplet)
