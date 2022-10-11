import numpy as np

from .simulation import KineticsRelaxationBase


def fun():
    SAx, SAy, SAz = spinops(0, spins)
    SBx, SBy, SBz = spinops(1, spins)

    QS = projop(spins, "S")
    QT = projop(spins, "T")
    QTp = projop(spins, "Tp")
    QTm = projop(spins, "Tm")
    QT0 = projop(spins, "T0")


def STD():
    return k * (np.kron(QS, QT) + np.kron(QT, QS))


def TTD():
    return k * (
        np.kron(QTp, QTm)
        + np.kron(QTm, QTp)
        + np.kron(QT0, QTm)
        + np.kron(QTm, QT0)
        + np.kron(QTp, QT0)
        + np.kron(QT0, QTp)
    )


def TTR():
    return k * (
        (
            2 / 3 * (np.kron(QT0, QT0))
            + (
                1
                / 3
                * (
                    (
                        np.kron(QTp, QTp)
                        + np.kron(QTm, QTm)
                        + np.kron(QTp, QTm)
                        + np.kron(QTm, QTp)
                    )
                    - (
                        np.kron(QTp, QT0)
                        - np.kron(QT0, QTp)
                        - np.kron(QTm, QT0)
                        - np.kron(QT0, QTm)
                        - np.kron(QTp, QTm)
                        - np.kron(QTm, QTp)
                    )
                )
            )
        )
    )


def RFR():
    return k * (
        1.5 * np.kron(np.eye(len(QS)), np.eye(len(QS)))
        - np.kron(SAx, SAx.T)
        - np.kron(SAy, SAy.T)
        - np.kron(SAz, SAz.T)
        - np.kron(SBx, SBx.T)
        - np.kron(SBy, SBy.T)
        - np.kron(SBz, SBz.T)
    )


def DM():
    return k * (
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
