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

    @property
    def rate_constant(self) -> float:
        return self.rate


class Diffusion(KineticsRelaxationBase):
    def __init__(self, r_sigma=5e-10, r0=9e-10, diffusion_coefficient=1e-9):
        self.r_sigma = r_sigma
        self.r0 = r0
        self.diffusion_coefficient = diffusion_coefficient
        numerator = self.r_sigma * (self.r0 - self.r_sigma)
        denominator = self.r0 * np.sqrt(4 * np.pi * self.diffusion_coefficient)
        self.term0 = numerator / denominator
        self.term1 = ((self.r0 - self.r_sigma) ** 2) / (4 * self.diffusion_coefficient)

    def adjust_product_probabilities(
        self, product_probabilities: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        product_probabilities *= (
            self.term0 * time ** (-3 / 2) * np.exp(-self.term1 / time)
        )

    def __repr__(self):
        lines = [
            self._name(),
            f"r_sigma: {self.r_sigma}",
            f"r0: {self.r0}",
            f"diffusion_coefficient: {self.diffusion_coefficient}",
        ]
        return "\n".join(lines)


class KineticsBase(LiouvilleKineticsRelaxationBase):
    @staticmethod
    def _convert(Q: np.ndarray) -> np.ndarray:
        return np.kron(Q, np.eye(len(Q))) + np.kron(np.eye(len(Q)), Q.T)


class Haberkorn(KineticsBase):
    """
    >>> Haberkorn(LiouvilleSimulation([]), rate_constant=1e6, target=State.SINGLET)
    Kinetics: Haberkorn
    Rate constant: 1000000.0
    Target: S

    >>> Haberkorn(LiouvilleSimulation([]), rate_constant=1e6, target=State.TRIPLET)
    Kinetics: Haberkorn
    Rate constant: 1000000.0
    Target: T
    """

    def __init__(self, sim: LiouvilleSimulation, rate_constant: float, target: State):
        super().__init__(rate_constant)
        self.target = target
        if target not in {State.SINGLET, State.TRIPLET}:
            raise ValueError(
                "Haberkorn kinetics supports only SINGLET and TRIPLET targets"
            )
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
    >>> HaberkornFree(LiouvilleSimulation([]), rate_constant=1e6)
    Kinetics: HaberkornFree
    Rate constant: 1000000.0
    """

    def __init__(self, sim: LiouvilleSimulation, rate_constant: float):
        super().__init__(rate_constant)
        size = prod(m for m in sim.multiplicities)
        self.subH = 0.5 * self.rate * np.eye(size)


class JonesHore(KineticsBase):
    """
    >>> JonesHore(LiouvilleSimulation([]), 1e6, 1e7)
    Kinetics: JonesHore
    Singlet rate: 1000000.0
    Triplet rate: 10000000.0
    """

    def __init__(
        self, sim: LiouvilleSimulation, singlet_rate: float, triplet_rate: float
    ):
        self.singlet_rate = singlet_rate
        self.triplet_rate = triplet_rate
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
