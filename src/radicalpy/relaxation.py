import numpy as np

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
