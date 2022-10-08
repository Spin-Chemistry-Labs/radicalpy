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


class Haberkorn(KineticsRelaxationBase):
    def __init__(self, rate_constant: float, type: State or list[State]):
        super().__init__(rate_constant)
        self.type = type if isinstance(type, list) else [type]

    @staticmethod
    def _convert(Q):
        return np.kron(Q, np.eye(len(Q))) + (np.kron(np.eye(len(Q)), Q))

    def adjust_hamiltonian(self, H: np.ndarray, sim: LiouvilleSimulation):
        Q = sum(sim.hilbert_projop(t) for t in self.type)
        H -= 0.5 * self.k * self._convert(Q)


class JonesHore(KineticsRelaxationBase):
    def __init__(self, rate_constant: float, ks: float, kt: float):
        super().__init__(rate_constant)
        self.type = type if isinstance(type, list) else [type]

    def adjust_hamiltonian(self, H: np.ndarray, sim: LiouvilleSimulation):
        QS = sim.hilbert_projop(State.SINGLET)
        QT = sim.hilbert_projop(State.TRIPLET)
        H -= (
            0.5 * self.ks * self.convert(QS)
            + 0.5 * self.kt * self.convert(QT)
            + (0.5 * (self.ks + self.kt)) * (np.kron(QS, QT) + np.kron(QT, QS))
        )
