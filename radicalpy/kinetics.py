"""Kinetics operators and superoperators for radical-pair dynamics.

This module implements common incoherent **kinetic** processes acting on
spin systems in either Hilbert space (operators) or Liouville space
(superoperators). These terms model recombination, product formation,
and phenomenological decay channels that act alongside coherent spin
evolution.

Classes:
        - `HilbertKineticsBase`: Base class for Hilbert-space kinetics operators.
        - `LiouvilleKineticsBase`: Base class for Liouville-space kinetics superoperators.
        - `Exponential`: Simple exponential decay of product probabilities in Hilbert space
          (Kaptein model).
        - `Haberkorn`: Haberkorn singlet/triplet selective recombination superoperator.
        - `HaberkornFree`: Haberkorn-style uniform decay (free radical / RP2 formation).
        - `JonesHore`: Jones–Hore two-site kinetics with separate S/T rates.

Usage pattern:
        1) Instantiate a kinetics object with the desired rate parameters
           (e.g., `rate_constant`, `singlet_rate`, `triplet_rate`, `target`).
        2) Call `.init(sim)` with a simulation exposing the required API
           (projection operators, sizes, and Liouville conversion).
        3) Combine the returned term with your total generator:
           - Hilbert space: use the operator’s effect where appropriate.
           - Liouville space: add `subH` to the Liouvillian.

Args conventions (per class):
        - `Exponential(rate_constant)`: Scales product probabilities as
          `exp(-rate * t)`.
        - `Haberkorn(rate_constant, target)`: Selective S/T recombination where
          `target` is one of `State.SINGLET`, `State.TRIPLET`, or specific
          triplet substates supported by the code.
        - `HaberkornFree(rate_constant)`: Uniform (state-independent) decay
          proportional to identity in Liouville space.
        - `JonesHore(singlet_rate, triplet_rate)`: Two-channel model with separate
          S and T rates and a coupling term constructed from S/T projectors.

Notes:
        - **Hilbert vs Liouville**: Hilbert-space `Exponential` modifies product
          probabilities directly; Liouville-space classes construct superoperators
          (`subH`) via Kronecker products and projector conversions.
        - **Projectors**: `QS`, `QT`, and triplet sublevel projectors are obtained
          from the simulation (`sim.projection_operator(...)`); Haberkorn and
          Jones–Hore rely on these for state selectivity.
        - **Rates and units**: All rate parameters are in s⁻¹. Ensure time arrays
          passed to `Exponential.adjust_product_probabilities` are in seconds.

References:
        - [Haberkorn, *Mol. Phys.* **32** (5), 1491–1493 (1976)](http://dx.doi.org/10.1080/00268977600102851).
        - [Kaptein et al., *Chem. Phys. Lett.* **4** (4), 195–197 (1969)](https://doi.org/10.1016/0009-2614(69)80098-9).
        - [Jones et al., *Chem. Phys. Lett.* **507**, 269–273 (2011)](https://doi.org/10.1016/j.cplett.2011.03.082).

Requirements:
        - `numpy` for array operations.
        - A simulation object with:
          `projection_operator(State.*)`, `hamiltonian_size`,
          and a Liouville conversion method (e.g., `sim._convert(Q)`).

Raises:
        ValueError: `Haberkorn` validates that `target` is singlet or triplet
            (or supported triplet sublevels) and will raise if unsupported.

See also:
        `radicalpy.simulation.HilbertIncoherentProcessBase`,
        `radicalpy.simulation.LiouvilleIncoherentProcessBase`,
        and related relaxation modules for non-kinetic incoherent processes.
"""

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
        if target not in {
            State.SINGLET,
            State.TP_SINGLET,
            State.TP_TRIPLET,
            State.TP_TRIPLET_ZERO,
            State.TP_TRIPLET_PLUS,
            State.TP_TRIPLET_MINUS,
            State.TP_QUINTET,
            State.TP_QUINTET_ZERO,
            State.TP_QUINTET_PLUS_TWO,
            State.TP_QUINTET_PLUS_ONE,
            State.TP_QUINTET_MINUS_TWO,
            State.TP_QUINTET_MINUS_ONE,
            State.TRIPLET,
            State.TRIPLET_MINUS,
            State.TRIPLET_PLUS,
            State.TRIPLET_PLUS_MINUS,
            State.TRIPLET_ZERO,
        }:
            raise ValueError(
                "Haberkorn kinetics supports only SINGLET and TRIPLET targets"
            )

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        Q = sim.projection_operator(self.target)
        self.subH = 0.5 * self.rate * sim._convert(Q)

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
        self.subH = self.rate * np.eye(sim.hamiltonian_size)


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
            0.5 * self.singlet_rate * sim._convert(QS)
            + 0.5 * self.triplet_rate * sim._convert(QT)
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
