from math import prod

import numpy as np

from .simulation import (
    HilbertIncoherentProcessBase,
    LiouvilleIncoherentProcessBase,
    LiouvilleSimulation,
    State,
)


class HilbertKineticsBase(HilbertIncoherentProcessBase):
    """Base class for kinetics operators (Hilbert space)."""

    def _name(self):
        name = super()._name()
        return f"Kinetics: {name}"


class LiouvilleKineticsBase(LiouvilleIncoherentProcessBase):
    """Base class for kinetics superoperators (Liouville space)."""

    def _name(self):
        name = super()._name()
        return f"Kinetics: {name}"

    @staticmethod
    def _convert(Q: np.ndarray) -> np.ndarray:
        return np.kron(Q, np.eye(len(Q))) + np.kron(np.eye(len(Q)), Q)


class Exponential(HilbertKineticsBase):
    """Exponential model kinetics operator.

    Source: `Kaptein et al. Chem. Phys. Lett. 4, 4, 195-197 (1969)`_.

    >>> Exponential(rate_constant=1e6)
    Kinetics: Exponential
    Rate constant: 1000000.0

    .. _Kaptein et al. Chem. Phys. Lett. 4, 4, 195-197 (1969):
       https://doi.org/10.1016/0009-2614(69)80098-9
    """

    def adjust_product_probabilities(
        self,
        product_probabilities: np.ndarray,
        time: np.ndarray,
    ):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.adjust_product_probabilities`."""
        product_probabilities *= np.exp(-self.rate * time)


class Haberkorn(LiouvilleKineticsBase):
    """Haberkorn kinetics superoperator for singlet/triplet recombination/product formation.

    Source: `Haberkorn, Mol. Phys. 32:5, 1491-1493 (1976)`_.

    Args:
        rate_constant (float): The kinetic rate constant (1/s).
        target (State): The target state of the reaction pathway
            (singlet or triplet states only).

    >>> Haberkorn(rate_constant=1e6, target=State.SINGLET)
    Kinetics: Haberkorn
    Rate constant: 1000000.0
    Target: S

    >>> Haberkorn(rate_constant=1e6, target=State.TRIPLET)
    Kinetics: Haberkorn
    Rate constant: 1000000.0
    Target: T

    .. _Haberkorn, Mol. Phys. 32:5, 1491-1493 (1976):
       http://dx.doi.org/10.1080/00268977600102851
    """

    def __init__(self, rate_constant: float, target: State):
        super().__init__(rate_constant)
        self.target = target
        if target not in {State.SINGLET, State.TRIPLET}:
            raise ValueError(
                "Haberkorn kinetics supports only SINGLET and TRIPLET targets"
            )

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        Q = sim.projection_operator(self.target)
        self.subH = 0.5 * self.rate * self._convert(Q)

    def __repr__(self):
        lines = [
            super().__repr__(),
            f"Target: {self.target.value}",
        ]
        return "\n".join(lines)


class HaberkornFree(LiouvilleKineticsBase):
    """Haberkorn kinetics superoperator for free radical/RP2 formation.

    Source: `Haberkorn, Mol. Phys. 32:5, 1491-1493 (1976)`_.

    >>> HaberkornFree(rate_constant=1e6)
    Kinetics: HaberkornFree
    Rate constant: 1000000.0

    .. _Haberkorn, Mol. Phys. 32:5, 1491-1493 (1976):
       http://dx.doi.org/10.1080/00268977600102851
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        size = prod(p.multiplicity for p in sim.particles) ** 2
        self.subH = self.rate * np.eye(size)


class JonesHore(LiouvilleKineticsBase):
    """Jones-Hore kinetics superoperator for two-site models.

    Source: `Jones et al. Chem. Phys. Lett. 507, 269-273 (2011)`_.

    Args:
        singlet_rate (float): Singlet recombination rate constant (1/s).
        triplet_rate (float): Triplet product formation rate constant (1/s).

    >>> JonesHore(1e7, 1e6)
    Kinetics: JonesHore
    Singlet rate: 10000000.0
    Triplet rate: 1000000.0

    .. _Jones et al. Chem. Phys. Lett. 507, 269-273 (2011):
       https://doi.org/10.1016/j.cplett.2011.03.082
    """

    def __init__(self, singlet_rate: float, triplet_rate: float):
        self.singlet_rate = singlet_rate
        self.triplet_rate = triplet_rate

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        QS = sim.projection_operator(State.SINGLET)
        QT = sim.projection_operator(State.TRIPLET)
        self.subH = (
            0.5 * self.singlet_rate * self._convert(QS)
            + 0.5 * self.triplet_rate * self._convert(QT)
            + (0.5 * (self.singlet_rate + self.triplet_rate))
            * (np.kron(QS, QT) + np.kron(QT, QS))
        )

    def __repr__(self):
        lines = [
            self._name(),
            f"Singlet rate: {self.singlet_rate}",
            f"Triplet rate: {self.triplet_rate}",
        ]
        return "\n".join(lines)

    @property
    def rate_constant(self) -> float:
        """Average rate of the kinetic processes."""
        return (self.singlet_rate + self.triplet_rate) / 2
