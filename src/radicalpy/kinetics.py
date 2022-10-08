import numpy as np

from .simulation import KineticsRelaxationBase, LiouvilleSimulation, State


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

    def adjust_product_probabilities(
        self, product_probabilities: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        numerator = self.r_sigma * (self.r0 - self.r_sigma)
        denominator = self.r0 * np.sqrt(4 * np.pi * self.diffusion_coefficient)
        A = numerator / denominator
        B = ((self.r0 - self.r_sigma) ** 2) / (4 * self.diffusion_coefficient)
        product_probabilities *= A * time ** (-3 / 2) * np.exp(-B / time)


class KineticsBase(KineticsRelaxationBase):
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

    def __repr__(self):
        lines = [
            super().__repr__(),
            f"Target: {self.target.value}",
        ]
        return "\n".join(lines)

    def adjust_hamiltonian(self, H: np.ndarray, sim: LiouvilleSimulation):
        Q = sim.projection_operator(self.target)
        H -= 0.5 * self.rate * self._convert(Q)


class HaberkornFree(KineticsBase):
    """
    >>> HaberkornFree(rate_constant=1e6)
    Kinetics: HaberkornFree
    Rate constant: 1000000.0
    """

    def adjust_hamiltonian(self, H: np.ndarray, sim: LiouvilleSimulation):
        QL = np.eye(len(H))
        H -= 0.5 * self.rate * QL


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

    def __repr__(self):
        lines = [
            self._name(),
            f"Singlet rate: {self.singlet_rate}",
            f"Triplet rate: {self.triplet_rate}",
        ]
        return "\n".join(lines)

    def adjust_hamiltonian(self, H: np.ndarray, sim: LiouvilleSimulation):
        QS = sim.projection_operator(State.SINGLET)
        QT = sim.projection_operator(State.TRIPLET)
        H -= (
            0.5 * self.singlet_rate * self._convert(QS)
            + 0.5 * self.triplet_rate * self._convert(QT)
            + (0.5 * (self.singlet_rate + self.triplet_rate))
            * (np.kron(QS, QT) + np.kron(QT, QS))
        )
