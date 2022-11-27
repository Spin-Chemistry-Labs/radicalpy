import numpy as np

from .simulation import (KineticsRelaxationBase,
                         LiouvilleKineticsRelaxationBase, LiouvilleSimulation,
                         State)
from .utils import spectral_density


class SingletTripletDephasing(LiouvilleKineticsRelaxationBase):
    def init(self, sim: LiouvilleSimulation):
        super().init(sim)
        QS = sim.projection_operator(State.SINGLET)
        QT = sim.projection_operator(State.TRIPLET)
        self.subH = self.rate * (np.kron(QS, QT) + np.kron(QT, QS))


class TripletTripletDephasing(LiouvilleKineticsRelaxationBase):
    def init(self, sim: LiouvilleSimulation):
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


class TripletTripletRelaxation(LiouvilleKineticsRelaxationBase):
    # restrict to
    # init_state=rpsim.State.TRIPLET_ZERO,
    # obs_state=rpsim.State.TRIPLET_ZERO,
    def init(self, sim: LiouvilleSimulation):
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


class RandomFields(LiouvilleKineticsRelaxationBase):
    def init(self, sim: LiouvilleSimulation):
        super().init(sim)
        QS = sim.projection_operator(State.SINGLET)
        self.SABxyz = [
            sim.spin_operator(e, a) for e in range(sim.num_electrons) for a in "xyz"
        ]

        term0 = np.kron(np.eye(len(QS)), np.eye(len(QS)))
        term1 = sum([np.kron(S, S.T) for S in self.SABxyz])
        self.subH = self.rate * (1.5 * term0 - term1)


class DipolarModulation(LiouvilleKineticsRelaxationBase):
    def init(self, sim: LiouvilleSimulation):
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


def g_tensor_anisotropy_term(sim: LiouvilleSimulation, idx, g, omega, tau_c):
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


# !!!!!!!!!! omega depends on B, which changes in every step (MARY loop)
# See note below
# Instead of omega1 & omega2 use B and calculate omegas inside
class GTensorAnisotropy(LiouvilleKineticsRelaxationBase):
    def __init__(self, g1, g2, omega1, omega2, tau_c1, tau_c2):
        self.g1 = g1
        self.g2 = g2
        self.omega1 = omega1
        self.omega2 = omega2
        self.tau_c1 = tau_c1
        self.tau_c2 = tau_c2

    def init(self, sim: LiouvilleSimulation):
        self.subH = g_tensor_anisotropy_term(sim, 0, self.g1, self.omega1, self.tau_c1)
        self.subH += g_tensor_anisotropy_term(sim, 1, self.g2, self.omega2, self.tau_c2)


class T1Relaxation(LiouvilleKineticsRelaxationBase):
    def init(self, sim: LiouvilleSimulation):
        SAz = sim.spin_operator(0, "z")
        SBz = sim.spin_operator(1, "z")

        self.subH = self.rate * (
            np.eye(len(SAz) * len(SAz)) - np.kron(SAz, SAz.T) - np.kron(SBz, SBz.T)
        )


class T2Relaxation(LiouvilleKineticsRelaxationBase):
    def init(self, sim: LiouvilleSimulation):
        SAx, SAy = sim.spin_operator(0, "x"), sim.spin_operator(0, "y")
        SBx, SBy = sim.spin_operator(1, "x"), sim.spin_operator(1, "y")

        self.subH = self.rate * (
            np.eye(len(SAx) * len(SAx))
            - np.kron(SAx, SAx.T)
            - np.kron(SBx, SBx.T)
            - np.kron(SAy, SAy.T)
            - np.kron(SBy, SBy.T)
        )
