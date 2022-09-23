import numpy as np

from .simulation import KineticsRelaxationBase


class KineticsExponential(KineticsRelaxationBase):
    def adjust_product_probabilities(
        self,
        product_probabilities: np.ndarray,
        time: np.ndarray,
    ):
        product_probabilities *= np.exp(-self.rate * time)

    @property
    def rate_constant(self) -> float:
        return self.rate


class KineticsDiffusion(KineticsRelaxationBase):
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
