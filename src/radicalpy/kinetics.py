from math import prod

import numpy as np

from .simulation import (KineticsRelaxationBase,
                         LiouvilleKineticsRelaxationBase, LiouvilleSimulation,
                         State)


class Exponential(KineticsRelaxationBase):
    def adjust_product_probabilities(
        self,
        product_probabilities: np.ndarray,
        time: np.ndarray,
    ):
        product_probabilities *= np.exp(-self.rate * time)


class KineticsBase(LiouvilleKineticsRelaxationBase):
    @staticmethod
    def _convert(Q: np.ndarray) -> np.ndarray:
        return np.kron(Q, np.eye(len(Q))) + np.kron(np.eye(len(Q)), Q.T)


class Haberkorn(KineticsBase):
    """
    >>> Haberkorn(rate_constant=1e6, target=State.SINGLET)
    Kinetics: Haberkorn
    Rate constant: 1000000.0
    Target: S

    >>> Haberkorn(rate_constant=1e6, target=State.TRIPLET)
    Kinetics: Haberkorn
    Rate constant: 1000000.0
    Target: T
    """

    def __init__(self, rate_constant: float, target: State):
        super().__init__(rate_constant)
        self.target = target
        if target not in {State.SINGLET, State.TRIPLET}:
            raise ValueError(
                "Haberkorn kinetics supports only SINGLET and TRIPLET targets"
            )

    def init(self, sim: LiouvilleSimulation):
        Q = sim.projection_operator(self.target)
        self.subH = 0.5 * self.rate * self._convert(Q)

    def __repr__(self):
        lines = [
            super().__repr__(),
            f"Target: {self.target.value}",
        ]
        return "\n".join(lines)


class HaberkornFree(KineticsBase):
    """
    >>> HaberkornFree(rate_constant=1e6)
    Kinetics: HaberkornFree
    Rate constant: 1000000.0
    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def init(self, sim: LiouvilleSimulation):
        size = prod(m for m in sim.multiplicities) ** 2
        self.subH = 0.5 * self.rate * np.eye(size)


class JonesHore(KineticsBase):
    """
    >>> JonesHore(1e6, 1e7)
    Kinetics: JonesHore
    Singlet rate: 1000000.0
    Triplet rate: 10000000.0
    """

    def __init__(self, singlet_rate: float, triplet_rate: float):
        self.singlet_rate = singlet_rate
        self.triplet_rate = triplet_rate

    def init(self, sim: LiouvilleSimulation):
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
        return (self.singlet_rate + self.triplet_rate) / 2
