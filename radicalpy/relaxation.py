"""Relaxation superoperators.

.. todo:: Add module docstring.
"""
import numpy as np

from .simulation import LiouvilleIncoherentProcessBase, LiouvilleSimulation, State
from .utils import spectral_density


class LiouvilleRelaxationBase(LiouvilleIncoherentProcessBase):
    """Base class for relaxation superoperators (Liouville space)."""

    def _name(self):
        """First line of `__repr__()`."""
        name = super()._name()
        return f"Relaxation: {name}"


class DipolarModulation(LiouvilleRelaxationBase):
    """Dipolar modulation relaxation superoperator.

    Source: `Kattnig et al. New J. Phys., 18, 063007 (2016)`_.

    >>> DipolarModulation(rate_constant=1e6)
    Relaxation: DipolarModulation
    Rate constant: 1000000.0
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QTp = sim.projection_operator(State.TRIPLET_PLUS)
        QTm = sim.projection_operator(State.TRIPLET_MINUS)
        QT0 = sim.projection_operator(State.TRIPLET_ZERO)
        QS = sim.projection_operator(State.SINGLET)
        self.subH = self.rate * (
            1 / 9 * np.kron(QS, QTp)
            + 1 / 9 * np.kron(QTp, QS)
            + 1 / 9 * np.kron(QS, QTm)
            + 1 / 9 * np.kron(QTm, QS)
            + 4 / 9 * np.kron(QS, QT0)
            + 4 / 9 * np.kron(QT0, QS)
            + np.kron(QTp, QT0)
            + np.kron(QT0, QTp)
            + np.kron(QTm, QT0)
            + np.kron(QT0, QTm)
        )


# !!!!!!!!!! omega depends on B, which changes in every step (MARY loop)
# See note below
# Instead of omega1 & omega2 use B and calculate omegas inside
class GTensorAnisotropy(LiouvilleRelaxationBase):
    """g-tensor anisotropy relaxation superoperator.

    Source: `Kivelson, J. Chem. Phys. 33, 1094 (1960)`_.

    Args:
        g1 (list): The principle components of g-tensor of the first radical.
        g2 (list): The principle components of g-tensor of the second radical.
        omega1 (float): The Larmor frequency of the first radical (rad/s/mT).
        omega2 (float): The Larmor frequency of the second radical (rad/s/mT).
        tau_c1 (float): The rotational correlation time of the first radical (s).
        tau_c2 (float): The rotational correlation time of the second radical (s).

    >>> GTensorAnisotropy(g1=[2.0032, 1.9975, 2.0014],
    ...                   g2=[2.00429, 2.00389, 2.00216],
    ...                   omega1=-158477366720.7,
    ...                   omega2=-158477366720.7,
    ...                   tau_c1=5e-12,
    ...                   tau_c2=100e-12)
    Relaxation: GTensorAnisotropy
    g1: [2.0032, 1.9975, 2.0014]
    g2: [2.00429, 2.00389, 2.00216]
    omega1: -158477366720.7
    omega2: -158477366720.7
    tau_c1: 5e-12
    tau_c2: 1e-10

    .. _Kivelson, J. Chem. Phys. 33, 1094 (1960):
       https://doi.org/10.1063/1.1731340
    """

    def __init__(
        self,
        g1: list,
        g2: list,
        omega1: float,
        omega2: float,
        tau_c1: float,
        tau_c2: float,
    ):
        self.g1 = g1
        self.g2 = g2
        self.omega1 = omega1
        self.omega2 = omega2
        self.tau_c1 = tau_c1
        self.tau_c2 = tau_c2

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        self.subH = _g_tensor_anisotropy_term(sim, 0, self.g1, self.omega1, self.tau_c1)
        self.subH += _g_tensor_anisotropy_term(
            sim, 1, self.g2, self.omega2, self.tau_c2
        )

    def __repr__(self):
        lines = [
            self._name(),
            f"g1: {self.g1}",
            f"g2: {self.g2}",
            f"omega1: {self.omega1}",
            f"omega2: {self.omega2}",
            f"tau_c1: {self.tau_c1}",
            f"tau_c2: {self.tau_c2}",
        ]
        return "\n".join(lines)


class RandomFields(LiouvilleRelaxationBase):
    """Random fields relaxation superoperator.

    Source: `Kattnig et al. New J. Phys., 18, 063007 (2016)`_.

    >>> RandomFields(rate_constant=1e6)
    Relaxation: RandomFields
    Rate constant: 1000000.0
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QS = sim.projection_operator(State.SINGLET)
        idxs = range(len(sim.radicals))
        self.SABxyz = [sim.spin_operator(e, a) for e in idxs for a in "xyz"]

        term0 = np.kron(np.eye(len(QS)), np.eye(len(QS)))
        term1 = sum([np.kron(S, S.T) for S in self.SABxyz])
        self.subH = self.rate * (1.5 * term0 - term1)


class SingletTripletDephasing(LiouvilleRelaxationBase):
    """Singlet-triplet dephasing relaxation superoperator.

    Source: `Shushin, Chem. Phys. Lett. 181, 2,3, 274-278 (1991)`_.

    >>> SingletTripletDephasing(rate_constant=1e6)
    Relaxation: SingletTripletDephasing
    Rate constant: 1000000.0

    .. _Shushin, Chem. Phys. Lett. 181, 2,3, 274-278 (1991):
       https://doi.org/10.1016/0009-2614(91)90366-H
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QS = sim.projection_operator(State.SINGLET)
        QT = sim.projection_operator(State.TRIPLET)
        self.subH = self.rate * (np.kron(QS, QT) + np.kron(QT, QS))


class T1Relaxation(LiouvilleRelaxationBase):
    """T1 (spin-lattice, longitudinal, thermal) relaxation superoperator.

    Source: `Bloch, Phys. Rev. 70, 460-474 (1946)`_.

    >>> T1Relaxation(rate_constant=1e6)
    Relaxation: T1Relaxation
    Rate constant: 1000000.0

    .. _Bloch, Phys. Rev. 70, 460-474 (1946):
       https://doi.org/10.1103/PhysRev.70.460
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        SAz = sim.spin_operator(0, "z")
        SBz = sim.spin_operator(1, "z")

        self.subH = self.rate * (
            np.eye(len(SAz) * len(SAz)) - np.kron(SAz, SAz.T) - np.kron(SBz, SBz.T)
        )


class T2Relaxation(LiouvilleRelaxationBase):
    """T2 (spin-spin, transverse) relaxation superoperator.

    Source: `Bloch, Phys. Rev. 70, 460-474 (1946)`_.

    >>> T2Relaxation(rate_constant=1e6)
    Relaxation: T2Relaxation
    Rate constant: 1000000.0

    .. _Bloch, Phys. Rev. 70, 460-474 (1946):
       https://doi.org/10.1103/PhysRev.70.460
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        SAx, SAy = sim.spin_operator(0, "x"), sim.spin_operator(0, "y")
        SBx, SBy = sim.spin_operator(1, "x"), sim.spin_operator(1, "y")

        self.subH = self.rate * (
            np.eye(len(SAx) * len(SAx))
            - np.kron(SAx, SAx.T)
            - np.kron(SBx, SBx.T)
            - np.kron(SAy, SAy.T)
            - np.kron(SBy, SBy.T)
        )


class TripletTripletDephasing(LiouvilleRelaxationBase):
    """Triplet-triplet dephasing relaxation superoperator.

    Source: `Gorelik et al. J. Phys. Chem. A 105, 8011-8017 (2001)`_.

    >>> TripletTripletDephasing(rate_constant=1e6)
    Relaxation: TripletTripletDephasing
    Rate constant: 1000000.0

    .. _Gorelik et al. J. Phys. Chem. A 105, 8011-8017 (2001):
       https://doi.org/10.1021/jp0109628
    """

    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        super().init(sim)
        QTp = sim.projection_operator(State.TRIPLET_PLUS)
        QTm = sim.projection_operator(State.TRIPLET_MINUS)
        QT0 = sim.projection_operator(State.TRIPLET_ZERO)
        self.subH = self.rate * (
            np.kron(QTp, QTm)
            + np.kron(QTm, QTp)
            + np.kron(QT0, QTm)
            + np.kron(QTm, QT0)
            + np.kron(QTp, QT0)
            + np.kron(QT0, QTp)
        )


class TripletTripletRelaxation(LiouvilleRelaxationBase):
    """Triplet-triplet relaxation superoperator.

    Source: `Miura et al. J. Phys. Chem. A 119, 5534−5544 (2015)`_.

    >>> TripletTripletRelaxation(rate_constant=1e6)
    Relaxation: TripletTripletRelaxation
    Rate constant: 1000000.0

    .. _Miura et al. J. Phys. Chem. A 119, 5534−5544 (2015):
       https://doi.org/10.1021/acs.jpca.5b02183
    """

    # restrict to
    # init_state=rpsim.State.TRIPLET_ZERO,
    # obs_state=rpsim.State.TRIPLET_ZERO,
    def init(self, sim: LiouvilleSimulation):
        """See `radicalpy.simulation.HilbertIncoherentProcessBase.init`."""
        QTp = sim.projection_operator(State.TRIPLET_PLUS)
        QTm = sim.projection_operator(State.TRIPLET_MINUS)
        QT0 = sim.projection_operator(State.TRIPLET_ZERO)
        super().init(sim)
        term0 = np.kron(QT0, QT0)
        term1 = np.kron(QTp, QTp) + np.kron(QTm, QTm)
        term2 = (
            np.kron(QTp, QT0)
            + np.kron(QT0, QTp)
            + np.kron(QTm, QT0)
            + np.kron(QT0, QTm)
        )
        self.subH = self.rate * (2 / 3 * term0 + 1 / 3 * (term1 - term2))


def _g_tensor_anisotropy_term(
    sim: LiouvilleSimulation, idx: int, g: list, omega: float, tau_c: float
) -> np.ndarray:
    giso = np.mean(g)
    SAx, SAy, SAz = [sim.spin_operator(idx, ax) for ax in "xyz"]
    H = 0.5 * np.eye(len(SAx) * len(SAx), dtype=complex)
    H -= np.kron(SAx, SAx.T)
    H -= np.kron(SAy, SAy.T)
    H *= 3 * spectral_density(omega, tau_c)
    H += (
        2
        * spectral_density(0, tau_c)
        * (0.5 * np.eye(len(SAx) * len(SAx)) - 2 * np.kron(SAz, SAz.T))
    )
    H *= 1 / 15 * sum([((gj - giso) / giso) ** 2 for gj in g]) * omega**2
    return H
