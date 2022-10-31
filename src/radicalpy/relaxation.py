import numpy as np
from .utils import spectral_density

from .simulation import (KineticsRelaxationBase,
                         LiouvilleKineticsRelaxationBase, LiouvilleSimulation,
                         State)


class RelaxationBaseST(LiouvilleKineticsRelaxationBase):
    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def init(self, sim: LiouvilleSimulation):
        self.QS = sim.projection_operator(State.SINGLET)
        self.QT = sim.projection_operator(State.TRIPLET)


class RelaxationBaseAll(RelaxationBaseST):
    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def init(self, sim: LiouvilleSimulation):
        self.QTp = sim.projection_operator(State.TRIPLET_PLUS)
        self.QTm = sim.projection_operator(State.TRIPLET_MINUS)
        self.QT0 = sim.projection_operator(State.TRIPLET_ZERO)


class SingletTripletDephasing(RelaxationBaseAll):
    def init(self, sim: LiouvilleSimulation):
        super().init(sim)
        self.subH = self.rate * (np.kron(self.QS, self.QT) + np.kron(self.QT, self.QS))


class TripleTripletDephasing(RelaxationBaseAll):
    def init(self, sim: LiouvilleSimulation):
        super().init(sim)
        self.subH = self.rate * (
            np.kron(self.QTp, self.QTm)
            + np.kron(self.QTm, self.QTp)
            + np.kron(self.QT0, self.QTm)
            + np.kron(self.QTm, self.QT0)
            + np.kron(self.QTp, self.QT0)
            + np.kron(self.QT0, self.QTp)
        )


class TripletTripletRelaxation(RelaxationBaseAll):
    # restrict to
    # init_state=rpsim.State.TRIPLET_ZERO,
    # obs_state=rpsim.State.TRIPLET_ZERO,
    def init(self, sim: LiouvilleSimulation):
        super().init(sim)
        term0 = np.kron(self.QT0, self.QT0)
        term1 = np.kron(self.QTp, self.QTp) + np.kron(self.QTm, self.QTm)
        term2 = (
            np.kron(self.QTp, self.QT0)
            + np.kron(self.QT0, self.QTp)
            + np.kron(self.QTm, self.QT0)
            + np.kron(self.QT0, self.QTm)
        )
        self.subH = self.rate * (2 / 3 * term0 + 1 / 3 * (term1 - term2))


class RandomFields(RelaxationBaseAll):
    def init(self, sim: LiouvilleSimulation):
        super().init(sim)
        self.QS = sim.projection_operator(State.SINGLET)
        self.SABxyz = [
            sim.spin_operator(e, a) for e in range(sim.num_electrons) for a in "xyz"
        ]

        term0 = np.kron(np.eye(len(self.QS)), np.eye(len(self.QS)))
        term1 = sum([np.kron(S, S.T) for S in self.SABxyz])
        self.subH = self.rate * (1.5 * term0 - term1)


class DipolarModulation(RelaxationBaseAll):
    def init(self, sim: LiouvilleSimulation):
        super().init(sim)
        self.subH = self.rate * (
            1 / 9 * np.kron(self.QS, self.QTp)
            + 1 / 9 * np.kron(self.QTp, self.QS)
            + 1 / 9 * np.kron(self.QS, self.QTm)
            + 1 / 9 * np.kron(self.QTm, self.QS)
            + 4 / 9 * np.kron(self.QS, self.QT0)
            + 4 / 9 * np.kron(self.QT0, self.QS)
            + np.kron(self.QTp, self.QT0)
            + np.kron(self.QT0, self.QTp)
            + np.kron(self.QTm, self.QT0)
            + np.kron(self.QT0, self.QTm)
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