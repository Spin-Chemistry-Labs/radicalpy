import numpy as np

from .simulation import KineticsRelaxationBase, LiouvilleSimulation, State


class RelaxationBaseST(KineticsRelaxationBase):
    @staticmethod
    def __init__(self, rate_constant: float, sim: LiouvilleSimulation):
        super().__init__(rate_constant)
        self.QS = sim.projection_operator(State.SINGLET)
        self.QT = sim.projection_operator(State.TRIPLET)


class RelaxationBaseAll(RelaxationBaseST):
    @staticmethod
    def __init__(self, rate_constant: float, sim: LiouvilleSimulation):
        super().__init__(rate_constant, sim)
        self.QTp = sim.projection_operator(State.TRIPLET_PLUS)
        self.QTm = sim.projection_operator(State.TRIPLET_MINUS)
        self.QT0 = sim.projection_operator(State.TRIPLET_ZERO)


def fun():
    SAx, SAy, SAz = spinops(0, spins)
    SBx, SBy, SBz = spinops(1, spins)


class STD(RelaxationBaseAll):
    def adjust_hamiltonian(self, H: np.ndarray):
        H -= self.k * (np.kron(self.QS, self.QT) + np.kron(self.QT, self.QS))


class TTF(RelaxationBaseAll):
    def adjust_hamiltonian(self, H: np.ndarray):
        H -= self.k * (
            np.kron(self.QTp, self.QTm)
            + np.kron(self.QTm, self.QTp)
            + np.kron(self.QT0, self.QTm)
            + np.kron(self.QTm, self.QT0)
            + np.kron(self.QTp, self.QT0)
            + np.kron(self.QT0, self.QTp)
        )


class TTR(RelaxationBaseAll):
    def adjust_hamiltonian(self, H: np.ndarray):
        temr0 = np.kron(self.QT0, self.QT0)
        term1 = (
            np.kron(self.QTp, self.QTp)
            + np.kron(self.QTm, self.QTm)
            + np.kron(self.QTp, self.QTm)
            + np.kron(self.QTm, self.QTp)
        )
        temr2 = (
            np.kron(self.QTp, self.QT0)
            - np.kron(self.QT0, self.QTp)
            - np.kron(self.QTm, self.QT0)
            - np.kron(self.QT0, self.QTm)
            - np.kron(self.QTp, self.QTm)
            - np.kron(self.QTm, self.QTp)
        )
        H -= self.k * (2 / 3 * temr0 + 1 / 3 * (term1 - temr2))


class RFR(RelaxationBaseAll):
    ################ TODO SAx
    def adjust_hamiltonian(self, H: np.ndarray):
        H -= self.k * (
            1.5 * np.kron(np.eye(len(self.QS)), np.eye(len(self.QS)))
            - np.kron(self.SAx, self.SAx.T)
            - np.kron(self.SAy, self.SAy.T)
            - np.kron(self.SAz, self.SAz.T)
            - np.kron(self.SBx, self.SBx.T)
            - np.kron(self.SBy, self.SBy.T)
            - np.kron(self.SBz, self.SBz.T)
        )


class DM(RelaxationBaseAll):
    def adjust_hamiltonian(self, H: np.ndarray):
        H -= self.k * (
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
