import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import sympy as smp
from scipy import integrate, linalg
from scipy.linalg import expm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

sns.set_theme()


# Spin dynamics functions ----------------------------------------------------------------

# Pauli matrices
smp_Sx = smp.Matrix([[0, 0.5], [0.5, 0]])
smp_Sy = smp.Matrix([[0, -0.5j], [0.5j, 0]])
smp_Sz = smp.Matrix([[0.5, 0], [0, -0.5]])
smp_Sxyz = [smp_Sx, smp_Sy, smp_Sz]

np_Sx = np.array([[0, 0.5], [0.5, 0]])
np_Sy = np.array([[0, -0.5j], [0.5j, 0]])
np_Sz = np.array([[0.5, 0], [0, -0.5]])
np_Sxyz = [np_Sx, np_Sy, np_Sz]


# ST-basis transformation
def ST_basis(M, spins):
    # T+  T0  S  T-
    ST = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 0, 1],
        ]
    )

    C = np.kron(ST, np.eye(2 ** (spins - 2)))
    return C @ M @ C.T


def smp_spinop(S, pos, num):
    args = [S if i == pos else smp.eye(2) for i in range(num)]
    return smp.kronecker_product(*args)


def smp_spinop_eq(S, pos, num):
    args = [S if i == pos else smp.eye(2) for i in range(num)]
    return smp.Eq(smp.KroneckerProduct(*args), smp.kronecker_product(*args))


def smp_spinops_eq(pos, num, spnop=smp_spinop_eq):
    return [spnop(S, pos, num) for S in (Sx, Sy, Sz)]


def np_spinop(S, pos, num):
    args = [S if i == pos else np.eye(2) for i in range(num)]
    return functools.reduce(np.kron, args, np.eye(1))


def spinops(pos, num, spnop=np_spinop):
    if spnop == smp_spinop:
        return [spnop(S, pos, num) for S in (smp_Sx, smp_Sy, smp_Sz)]
    else:
        return [spnop(S, pos, num) for S in (np_Sx, np_Sy, np_Sz)]


def spinops_sum(pos, num, spnop=np_spinop):
    return sum(spinops(pos, num, spnop))


def prodops(pos, pos2, num, spnop=np_spinop):
    if spnop == smp_spinop:
        return [
            spnop(S, pos, num) @ spnop(S, pos2, num) for S in [smp_Sx, smp_Sy, smp_Sz]
        ]
    else:
        return [spnop(S, pos, num) @ spnop(S, pos2, num) for S in [np_Sx, np_Sy, np_Sz]]


def prodop(pos, pos2, num, spnop=np_spinop):
    return sum(prodops(pos, pos2, num, spnop))


def projop(spins, state):
    # Spin operators
    SAx, SAy, SAz = spinops(0, spins)
    SBx, SBy, SBz = spinops(1, spins)

    # Product operators
    SASB = prodop(0, 1, spins)

    # Projection operators
    match state:
        case "S":
            return (1 / 4) * np.eye(len(SASB)) - SASB
        case "T":
            return (3 / 4) * np.eye(len(SASB)) + SASB
        case "Tp":
            return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz)
        case "Tm":
            return (2 * SAz**2 - SAz) * (2 * SBz**2 - SBz)
        case "T0":
            return (1 / 4) * np.eye(len(SASB)) + SAx @ SBx + SAy @ SBy - SAz @ SBz
        case "Tpm":
            return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz) + (
                2 * SAz**2 - SAz
            ) * (2 * SBz**2 - SBz)
        case "Eq":
            return 1.05459e-34 / (1.38e-23 * 298)


def projop_Liouville(spins, state):
    if state == "Eq":
        return 1.05459e-34 / (1.38e-23 * 298)
    else:
        return np.reshape(projop(spins, state), (-1, 1))


def projop_3spin(spins, pos1, pos2, state):
    # Spin operators
    SAx, SAy, SAz = spinops(pos1, spins)
    SBx, SBy, SBz = spinops(pos2, spins)

    # Product operators
    SASB = prodop(pos1, pos2, spins)

    # Projection operators
    match state:
        case "S":
            return (1 / 4) * np.eye(len(SASB)) - SASB
        case "T":
            return (3 / 4) * np.eye(len(SASB)) + SASB
        case "Tp":
            return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz)
        case "Tm":
            return (2 * SAz**2 - SAz) * (2 * SBz**2 - SBz)
        case "T0":
            return (1 / 4) * np.eye(len(SASB)) + SAx @ SBx + SAy @ SBy - SAz @ SBz
        case "Tpm":
            return (2 * SAz**2 + SAz) * (2 * SBz**2 + SBz) + (
                2 * SAz**2 - SAz
            ) * (2 * SBz**2 - SBz)
        case "Eq":
            return 1.05459e-34 / (1.38e-23 * 298)


def rotate_x(x, theta, phi):
    return np.sin(theta) * np.cos(phi) * x


def rotate_y(y, theta, phi):
    return np.sin(theta) * np.sin(phi) * y


def rotate_z(z, theta, phi):
    return np.cos(theta) * z


def rotate(particles, theta, phi):
    rots = [rotate_x, rotate_y, rotate_z]
    zipped = sum((list(zip(rots, particle_axes)) for particle_axes in particles), [])
    return sum([rot(ax, theta, phi) for rot, ax in zipped])


def HamiltonianZeeman_RadicalPair(spins, B):
    omega = B * 1.76e8
    particles = sum([np_spinop(np_Sz, i, spins) for i in range(2)])
    return omega * particles


def HamiltonianZeeman3D(spins, B, theta=0, phi=0, gamma=1.76e8):
    omega = B * gamma
    particles = [spinops(i, spins) for i in range(spins)]
    return omega * rotate(particles, theta, phi)


def HamiltonianHyperfine(spins, pos, pos2, HFC, gamma_mT):
    omega = HFC * gamma_mT
    particles = prodop(pos, pos2, spins)
    return omega * particles


def ExchangeInteraction(r, model="solution"):
    match model:
        case "solution":
            J0, alpha = -570e-3, 2e10
            return -J0 * np.exp(-alpha * r)
        #             J0rad, rj, gamma = 1.7e17, 0.049e-9, 1.76e8
        #             J0 = J0rad / gamma / 10 # convert to mT
        #             return J0 * np.exp(-r / rj)
        case "protein":
            beta, J0 = 1.4e10, 8e13
            return (J0 * np.exp(-beta * r)) / 1000


def HamiltonianExchange(spins, J, gamma=1.76e8):
    Jcoupling = gamma * J
    SASB = prodop(0, 1, spins)
    return Jcoupling * ((2 * SASB) + (0.5 * np.eye(len(SASB))))


def DipolarInteraction1D(r):
    return -2.785 / r**3


def DipolarInteraction(r, gamma):
    dipolar = gamma * (2 / 3) * (-2.785 / r**3)
    return dipolar * np.diag([-1, -1, 2])


def HamiltonianDipolar(spins, D, gamma=1.76e8):
    Dint = gamma * D
    SASB = prodop(0, 1, spins)
    SAz = np_spinop(np_Sz, 0, spins)
    SBz = np_spinop(np_Sz, 1, spins)
    return (2 / 3) * Dint * ((3 * SAz * SBz) - SASB)


def HamiltonianDipolar3D(DipolarInteractions):
    ds = len(DipolarInteractions)
    assert ds in {1, 3}, (
        "Only 2 or 3 radicals supported, "
        "i.e. len(DipolarInteractions) should be 1 or 3"
    )
    ids = [] if ds == 1 else [np.eye(2)]
    terms = [
        [dk[i, j]] + [si, sj][:k] + ids + [si, sj][k:]
        for i, si in enumerate(np_Sxyz)
        for j, sj in enumerate(np_Sxyz)
        for k, dk in enumerate(DipolarInteractions)
    ]
    return sum(
        [
            t[0] * np.kron(np.kron(t[1], t[2]), t[3])
            if ds == 3
            else np.kron(t[1], t[2])
            for t in terms
        ]
    )


def Kinetics(spins, k=0, time=0, model="Haberkorn-singlet"):

    """
    Kinetic models include:
    "Exponential"
    "Diffusion"
    Haberkorn superoperators (singlet and triplet recombination, free radical (RP2) production)
    Jones-Hore superoperator

    Arguments:
        spins: an integer = sum of the number of electrons and nuclei
        k: a floating point number = kinetic rate constant in s^-1
        time: evenly spaced sequence in a specified interval i.e., np.linspace = used for "Exponential" and "Diffusion" model only
        model: string = select the kinetic model

    Returns:
        An array for "Exponential" and "Diffusion" models
        A superoperator matrix (Liouville space)

    Example:
        K = Kinetics(3, 1e6, 0, "Haberkorn-singlet")
    """

    QS = projop(spins, "S")
    QT = projop(spins, "T")

    match model:
        case "Exponential":
            return np.exp(-k * time)
        case "Diffusion":
            rsig = 5e-10  # Recombination distance
            r0 = 9e-10  # Created separation distance
            Dif = 1e-5 / 10000  # m^2/s Relative diffusion coefficient
            a_dif = (rsig * (r0 - rsig)) / (r0 * np.sqrt(4 * np.pi * Dif))
            b_dif = ((r0 - rsig) ** 2) / (4 * Dif)
            return a_dif * time ** (-3 / 2) * np.exp(-b_dif / time)
        case "Haberkorn-singlet":
            return (
                0.5
                * k
                * (np.kron(QS, np.eye(len(QS))) + (np.kron(np.eye(len(QS)), QS)))
            )
        case "Haberkorn-triplet":
            return (
                0.5
                * k
                * (np.kron(QT, np.eye(len(QT))) + (np.kron(np.eye(len(QT)), QT)))
            )
        case "Haberkorn-free":
            return k * np.kron(np.eye(len(QS)), np.eye(len(QS)))
        case "Jones-Hore":
            return (
                0.5
                * ks
                * (np.kron(QS, np.eye(len(QS))) + (np.kron(np.eye(len(QS)), QS)))
                + 0.5
                * kt
                * (np.kron(QT, np.eye(len(QT))) + (np.kron(np.eye(len(QT)), QT)))
                + (0.5 * (ks + kt)) * (np.kron(QS, QT) + np.kron(QT, QS))
            )


def Relaxation(spins, k=0, model="ST-Dephasing"):

    """
    Relaxation models include:
    Singlet-Triplet Dephasing (STD)
    Triplet-Triplet Dephasing (TTD)
    Triplet-Triplet Relaxation (TTR)
    Random Field Relaxation (RFR)
    Dipolar Modulation (DM)

    Arguments:
        spins: an integer = sum of the number of electrons and nuclei
        k: a floating point number = relaxation rate constant in s^-1
        model: string = select the relaxation model

    Returns:
        A superoperator matrix (Liouville space)

    Example:
        R = Relaxation(3, 1e6, "STD")
    """

    SAx, SAy, SAz = spinops(0, spins)
    SBx, SBy, SBz = spinops(1, spins)

    QS = projop(spins, "S")
    QT = projop(spins, "T")
    QTp = projop(spins, "Tp")
    QTm = projop(spins, "Tm")
    QT0 = projop(spins, "T0")

    match model:
        case "STD":
            return k * (np.kron(QS, QT) + np.kron(QT, QS))
        case "TTD":
            return k * (
                np.kron(QTp, QTm)
                + np.kron(QTm, QTp)
                + np.kron(QT0, QTm)
                + np.kron(QTm, QT0)
                + np.kron(QTp, QT0)
                + np.kron(QT0, QTp)
            )
        case "TTR":
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
        case "RFR":
            return k * (
                1.5 * np.kron(np.eye(len(QS)), np.eye(len(QS)))
                - np.kron(SAx, SAx.T)
                - np.kron(SAy, SAy.T)
                - np.kron(SAz, SAz.T)
                - np.kron(SBx, SBx.T)
                - np.kron(SBy, SBy.T)
                - np.kron(SBz, SBz.T)
            )
        case "DM":
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


def Hilbert2Liouville(H):

    """
    Converts a spin Hamiltonian matrix in Hilbert space to Liouville space

    Arguments:
        H: a matrix = spin Hamiltonian in Hilbert space

    Returns:
        A Liouvillian spin Hamiltonian

    Example:
        HL = Hilbert2Liouville(H)
    """

    return 1j * (np.kron(H, np.eye(len(H))) - np.kron(np.eye(len(H)), H.T))


def Hilbert_initial(state, spins, H):

    """
    Creates an initial density matrix for time evolution of the spin Hamiltonian density matrix

    Arguments:
        state: a string = spin state projection operator
        spins: an integer = sum of the number of electrons and nuclei
        H: a matrix = spin Hamiltonian in Hilbert space

    Returns:
        A matrix in Hilbert space

    Example:
        rho0 = Hilbert_initial("S", 3, H)
    """

    Pi = projop(spins, state)

    if np.array_equal(Pi, projop(spins, "Eq")):
        rho0eq = expm(-1j * H * Pi)
        rho0 = rho0eq / np.trace(rho0eq)
    else:
        rho0 = Pi / np.trace(Pi)
    return rho0


def Hilbert_observable(state, spins):

    """
    Creates an observable density matrix for time evolution of the spin Hamiltonian density matrix

    Arguments:
        state: a string = spin state projection operator
        spins: an integer = sum of the number of electrons and nuclei

    Returns:
        Two matrices in Hilbert space

    Example:
        obs, Pobs = Hilbert_observable("S", 3)
    """

    Pobs = projop(spins, state)

    rhoobs = Pobs / np.trace(Pobs)

    # Observables
    if np.array_equal(Pobs, projop(spins, "T")):
        obs = 1 - np.real(
            np.trace(
                np.matmul(
                    projop(spins, "S"),
                    (projop(spins, "S") / np.trace(projop(spins, "S"))),
                )
            )
        )
    else:
        obs = np.real(np.trace(np.matmul(Pobs, rhoobs)))
    return [obs, Pobs]


def Liouville_initial(state, spins, H):

    """
    Creates an initial density matrix for time evolution of the spin Hamiltonian density matrix

    Arguments:
        state: a string = spin state projection operator
        spins: an integer = sum of the number of electrons and nuclei
        H: a matrix = spin Hamiltonian in Hilbert space

    Returns:
        A matrix in Liouville space

    Example:
        rho0 = Liouville_initial("S", 3, H)
    """

    Pi = projop_Liouville(spins, state)

    if np.array_equal(Pi, projop_Liouville(spins, "Eq")):
        rho0eq = expm(-1j * H * Pi)
        rho0 = rho0eq / np.trace(rho0eq)
        rho0 = np.reshape(rho0, (len(H) ** 2, 1))
    else:
        rho0 = Pi / np.vdot(Pi, Pi)
    return rho0


def Liouville_observable(state, spins):

    """
    Creates an observable density matrix for time evolution of the spin Hamiltonian density matrix

    Arguments:
        state: a string = spin state projection operator
        spins: an integer = sum of the number of electrons and nuclei

    Returns:
        Two matrices in Liouville space

    Example:
        obs, Pobs = Liouville_observable("S", 3)
    """

    Pobs = projop_Liouville(spins, state)

    rhoobs = Pobs / np.vdot(Pobs, Pobs)

    # Observables
    if np.array_equal(Pobs, projop_Liouville(spins, "T")):
        obs = 1 - np.real(
            np.trace(
                np.matmul(
                    projop_Liouville(spins, "S").T,
                    (
                        projop_Liouville(spins, "S")
                        / np.vdot(
                            projop_Liouville(spins, "S"), projop_Liouville(spins, "S")
                        )
                    ),
                )
            )
        )
    else:
        obs = np.real(np.trace(np.matmul(Pobs.T, rhoobs)))
    return [obs, Pobs]


def UnitaryPropagator(H, dt, space="Hilbert"):

    """
    Creates unitary propagator matrices for time evolution of the spin Hamiltonian density matrix in both Hilbert and Liouville space

    Arguments:
        H: a matrix = spin Hamiltonian in Hilbert or Liouville space
        dt: a floating point number = time evolution timestep
        space: a string = select the spin space

    Returns:
        Matrices in either Hilbert or Liouville space

    Example:
        Up, Um = UnitaryPropagator(H, 3e-9, "Hilbert")
        UL = UnitaryPropagator(HL, 3e-9, "Liouville")
    """

    match space:
        case "Hilbert":
            UnitaryPropagator_plus = expm(1j * H * dt)
            UnitaryPropagator_minus = expm(-1j * H * dt)
            return [UnitaryPropagator_plus, UnitaryPropagator_minus]
        case "Liouville":
            return expm(H * dt)


def TimeEvolution(
    spins,
    initial,
    observable,
    t_max,
    dt,
    k,
    B,
    H,
    space="Hilbert",
    model="Exponential",
):

    #     time = np.linspace(t_min, t_max, t_stepsize)
    #     dt = time[1] - time[0]
    time = np.arange(0, t_max, dt)

    match space:
        case "Hilbert":

            HZ = HamiltonianZeeman_RadicalPair(spins, B)
            H_total = H + HZ
            rho0 = Hilbert_initial(initial, spins, H_total)
            obs, Pobs = Hilbert_observable(observable, spins)

            Up, Um = UnitaryPropagator(H_total, dt, space="Hilbert")
            evol = np.zeros(len(time))
            evol[0] = obs
            rho = []

            for i in range(len(time)):
                rhot = Um @ rho0 @ Up
                rhot = rhot / np.trace(rhot)
                rho0 = rhot

                evol[i] = np.real(np.trace(np.matmul(Pobs, rhot)))
                rho.append(rhot)

            K = Kinetics(spins, k, time, model=model)
            evol_reaction = evol * K

            ProductYield = integrate.cumtrapz(evol_reaction, time, initial=0) * k
            ProductYieldSum = np.max(ProductYield)
            # print('Product yield: ', '%.2f' % ProductYieldSum)

            return [time, evol_reaction, ProductYield, ProductYieldSum, rho]

        case "Liouville":

            HZ = HamiltonianZeeman_RadicalPair(spins, B)
            HZ = Hilbert2Liouville(HZ)
            H_total = H + HZ
            rho0 = Liouville_initial(initial, spins, H_total)
            obs, Pobs = Liouville_observable(observable, spins)

            UL = UnitaryPropagator(H_total, dt, space="Liouville")
            evol = np.zeros(len(time))
            evol[0] = obs
            rho = []

            for i in range(len(time)):
                rhotL = UL @ rho0
                rho0 = rhotL

                evol[i] = np.real(np.trace(np.matmul(Pobs.T, rhotL)))
                rho.append(rhotL)

            ProductYield = integrate.cumtrapz(evol, time, initial=0) * k
            ProductYieldSum = np.max(ProductYield)
            #             print('Product yield: ', '%.2f' % ProductYieldSum)

            return [time, evol, ProductYield, ProductYieldSum, rho]


def MARY(spins, initial, observable, t_max, t_stepsize, k, B, Hplot, space="Hilbert"):

    timing = np.arange(0, t_max, t_stepsize)
    MFE = np.zeros((len(B), len(timing)))

    for i, B0 in enumerate(B):
        time, MFE[i, :], productyield, ProductYieldSum, rho = TimeEvolution(
            spins, initial, observable, t_max, t_stepsize, k, B0, Hplot, space=space
        )

    raw = MFE
    dt = t_stepsize
    MARY = np.sum(raw, axis=1) * dt * k

    if B[0] != 0:
        middle = int(len(MARY) / 2)
        HFE = ((MARY[-1] - MARY[middle]) / MARY[middle]) * 100
        if initial == "S":
            LFE = ((max(MARY) - MARY[middle]) / MARY[middle]) * 100
        else:
            LFE = ((min(MARY) - MARY[middle]) / MARY[middle]) * 100
        MARY = ((MARY - MARY[middle]) / MARY[middle]) * 100
    else:
        HFE = ((MARY[-1] - MARY[0]) / MARY[0]) * 100
        if initial == "S":
            LFE = ((max(MARY) - MARY[0]) / MARY[0]) * 100
        else:
            LFE = ((min(MARY) - MARY[0]) / MARY[0]) * 100
        MARY = ((MARY - MARY[0]) / MARY[0]) * 100
    return [time, MFE, HFE, LFE, MARY, productyield, ProductYieldSum, rho]


def Lorentzian_fit(x, A, Bhalf):
    return (A / Bhalf**2) - (A / (x**2 + Bhalf**2))


def Bhalf_fit(B, MARY):
    popt_MARY, pcov_MARY = curve_fit(
        Lorentzian_fit, B, MARY, p0=[MARY[-1], int(len(B) / 2)]
    )
    MARY_fit_error = np.sqrt(np.diag(pcov_MARY))

    A_opt_MARY, Bhalf_opt_MARY = popt_MARY
    x_model_MARY = np.linspace(min(B), max(B), len(B))
    y_model_MARY = Lorentzian_fit(x_model_MARY, *popt_MARY)
    Bhalf = np.abs(Bhalf_opt_MARY)

    y_pred_MARY = Lorentzian_fit(B, *popt_MARY)
    R2 = r2_score(MARY, y_pred_MARY)

    return Bhalf, x_model_MARY, y_model_MARY, MARY_fit_error, R2


def RotationalCorrelationTime_protein(Mr, temp):
    eta, V, Na, Kb, rw = 0.89e-3, 0.00073, 6.022e23, 1.38e-23, 2.4e-10

    # Calculate Rh - effective hydrodynamic radius of the protein in m
    Rh = ((3 * V * Mr) / (4 * np.pi * Na)) ** 0.33 + rw

    # Calculate isotropic rotational correlation time (tau_c) in s
    tau_c = (4 * np.pi * eta * Rh**3) / (3 * Kb * temp)
    return tau_c


def T1_T2_RelaxationTimes(g_tensors, B, tau_c):
    hbar, muB, gamma = 6.626e-34 / (2 * np.pi), 9.274e-24, 1.76e11
    g_mean = np.mean(g_tensors)
    g_innerproduct = (
        (g_tensors[0] - g_mean) ** 2
        + (g_tensors[1] - g_mean) ** 2
        + (g_tensors[2] - g_mean) ** 2
    )
    omega = gamma * B

    T1 = (
        (1 / 5)
        * ((muB * B) / hbar) ** 2
        * g_innerproduct
        * (tau_c / (1 + omega**2 * tau_c**2))
    ) ** (-1)
    T2 = (
        (1 / 30)
        * ((muB * B) / hbar) ** 2
        * g_innerproduct
        * (4 * tau_c + (3 * tau_c / (1 + omega**2 * tau_c**2)))
    ) ** (-1)
    return [T1, T2]


def EffectiveHyperfine(radical_hfc, radical_spin):

    radical_HFC = np.array(radical_hfc)
    spin_quantum_number = np.array(radical_spin)

    return np.sqrt(
        (4 / 3)
        * sum((radical_HFC**2 * spin_quantum_number) * (spin_quantum_number + 1))
    )


def BhalfTheoretical(radicalA_effectiveHFC, radicalB_effectiveHFC):
    return np.sqrt(3) * (
        (radicalA_effectiveHFC**2 + radicalB_effectiveHFC**2)
        / (radicalA_effectiveHFC + radicalB_effectiveHFC)
    )


# --------------------------------------------------------------------------------------------------

# Two-site model simulations ---------------------------------------------------------------------------


def SingletYieldTwoSite(
    B, gamma, theta, phi, r12A, r13A, r23A, r12B, r13B, r23B, kS, kF, tau, spins
):

    P12S = projop(spins, "S")
    P12SA = np.concatenate((np.ndarray.flatten(P12S), np.zeros(len(P12S) ** 2)))
    P12SB = np.concatenate((np.zeros(len(P12S) ** 2), np.ndarray.flatten(P12S)))
    rho0 = 0.5 * (P12SA + P12SB)
    M = 2
    kAB = kBA = 0.5 / tau
    HZ = HamiltonianZeeman3D(spins, B, theta, phi, gamma)

    r_distances = [
        [r23A, r13A, r12A],  # r23A, r13A, r12A
        [r23B, r13B, r12B],  # r23B, r13B, r12B
    ]

    DipolarInteractions = [
        [DipolarInteraction(r, gamma) for r in r_distance] for r_distance in r_distances
    ]

    HA = HZ + HamiltonianDipolar3D(DipolarInteractions[0])
    HB = HZ + HamiltonianDipolar3D(DipolarInteractions[1])

    K = [
        Kinetics(spins, (kF + kAB), 0, model="Haberkorn-free"),
        Kinetics(spins, (kF + kBA), 0, model="Haberkorn-free"),
    ]

    LA = (
        Hilbert2Liouville(HA) + Kinetics(spins, kS, 0, model="Haberkorn-singlet") + K[0]
    )
    LB = (
        Hilbert2Liouville(HB) + Kinetics(spins, kS, 0, model="Haberkorn-singlet") + K[1]
    )

    K2 = [
        Kinetics(spins, kAB, 0, model="Haberkorn-free"),
        Kinetics(spins, kBA, 0, model="Haberkorn-free"),
    ]

    L = np.block([[LA, -K2[1]], [-K2[0], LB]])

    singletyield = (kS / M) * (np.dot((P12SA + P12SB), np.linalg.solve(L, rho0)))

    return singletyield


def SingletYieldTwoSiteRelaxation(
    B, gamma, theta, phi, r12A, r13A, r23A, r12B, r13B, r23B, kS, kF, tau, spins
):

    P12S = projop(spins, "S")
    rho0 = P12S
    M = 2
    HZ = HamiltonianZeeman3D(spins, B, theta, phi, gamma)

    r_distance = [(r23A + r23B) / 2, (r13A + r13B) / 2, (r12A + r12B) / 2]

    DipolarInteractions = [DipolarInteraction(r, gamma) for r in r_distance]

    DD = HamiltonianDipolar3D(DipolarInteractions)
    H = HZ + DD
    K = Kinetics(spins, kS, 0, model="Haberkorn-singlet") + Kinetics(
        spins, kF, 0, model="Haberkorn-free"
    )
    L = Hilbert2Liouville(H) + K

    relaxation13 = (
        0.25
        * tau
        * gamma**2
        * (DipolarInteraction1D(r13A) - DipolarInteraction1D(r13B)) ** 2
    )
    relaxation23 = (
        0.25
        * tau
        * gamma**2
        * (DipolarInteraction1D(r23A) - DipolarInteraction1D(r23B)) ** 2
    )

    h13 = (
        projop_3spin(spins, 0, 2, "Tp") / 3
        + projop_3spin(spins, 0, 2, "Tm") / 3
        - 2 * projop_3spin(spins, 0, 2, "T0") / 3
    )
    h23 = (
        projop_3spin(spins, 1, 2, "Tp") / 3
        + projop_3spin(spins, 1, 2, "Tm") / 3
        - 2 * projop_3spin(spins, 1, 2, "T0") / 3
    )

    relaxh13 = np.dot(h13, h13)
    relaxh23 = np.dot(h23, h23)

    R = (
        relaxation13
        * (
            np.kron(relaxh13, np.eye(len(relaxh13)))
            + np.kron(np.eye(len(relaxh13)), relaxh13.T)
            - 2 * np.kron(h13, h13.T)
        )
    ) + (
        relaxation23
        * (
            np.kron(relaxh23, np.eye(len(relaxh23)))
            + np.kron(np.eye(len(relaxh23)), relaxh23.T)
            - 2 * np.kron(h23, h23.T)
        )
    )
    L += R

    singletyield = (kS / M) * (
        np.dot(np.ndarray.flatten(P12S), np.linalg.solve(L, np.ndarray.flatten(rho0)))
    )

    return singletyield


def SingletYieldTwoSiteApprox(
    B,
    gamma,
    theta,
    phi,
    r12A,
    r13A,
    r23A,
    r12B,
    r13B,
    r23B,
    k,
    kAB,
    kBA,
    tmax,
    tstep,
    spins,
):

    P12S = projop(spins, "S")
    HZ = HamiltonianZeeman3D(spins, B, theta, phi, gamma)

    r_distances = [
        [r23A, r13A, r12A],  # r23A, r13A, r12A
        [r23B, r13B, r12B],  # r23B, r13B, r12B
    ]

    DipolarInteractions = [
        [DipolarInteraction(r, gamma) for r in r_distance] for r_distance in r_distances
    ]

    HA = HZ + HamiltonianDipolar3D(DipolarInteractions[0])
    HB = HZ + HamiltonianDipolar3D(DipolarInteractions[1])

    K = [kAB * np.eye(len(HA)), kBA * np.eye(len(HB))]

    return singletyield


# Liouville space, exact time propagation, kAB = kBA
def SingletYieldTwoSiteTimePropagation(
    B,
    gamma,
    theta,
    phi,
    r12A,
    r13A,
    r23A,
    r12B,
    r13B,
    r23B,
    k,
    kAB,
    kBA,
    tmax,
    tstep,
    spins,
):

    P12S = projop(spins, "S")
    P12SA = np.concatenate((np.ndarray.flatten(P12S), np.zeros(len(P12S) ** 2)))
    P12SB = np.concatenate((np.zeros(len(P12S) ** 2), np.ndarray.flatten(P12S)))
    HZ = HamiltonianZeeman3D(spins, B, theta, phi, gamma)

    r_distances = [
        [r23A, r13A, r12A],  # r23A, r13A, r12A
        [r23B, r13B, r12B],  # r23B, r13B, r12B
    ]

    DipolarInteractions = [
        [DipolarInteraction(r, gamma) for r in r_distance] for r_distance in r_distances
    ]

    HA = HZ + HamiltonianDipolar3D(DipolarInteractions[0])
    HB = HZ + HamiltonianDipolar3D(DipolarInteractions[1])

    K = [
        Kinetics(spins, kAB, 0, model="Haberkorn-free"),
        Kinetics(spins, kBA, 0, model="Haberkorn-free"),
    ]

    LA = Hilbert2Liouville(HA) + K[0]
    LB = Hilbert2Liouville(HB) + K[1]
    L = np.block([[LA, -K[0]], [-K[1], LB]])

    nt = np.round((tmax / tstep + 1)).astype(int)
    rho0 = 0.5 * (P12SA + P12SB)
    propagator = expm(-L * tstep)
    singletfraction = np.zeros(nt, dtype=np.complex_)

    for i in range(0, nt):
        t = i * tstep
        singletfraction[i] = ((np.dot((P12SA + P12SB), rho0))) * np.exp(-k * t) / 2
        # singletfraction[i] = ((np.dot((P12SA + P12SB), rho0))) * Kinetics(0, k, t, model="Exponential") / 2
        rho = np.dot(propagator, rho0)

    weight = 4 * np.ones(nt)
    for i in range(2, nt - 1, 2):
        weight[i] = 2

    weight[0] = weight[-1] = 1

    singletyield = sum(singletfraction * weight) * tstep / 3

    return singletyield


# Hilbert space, approximate time propagation
def SingletYieldTwoSiteTimePropagationApprox(
    B,
    gamma,
    theta,
    phi,
    r12A,
    r13A,
    r23A,
    r12B,
    r13B,
    r23B,
    k,
    kAB,
    kBA,
    tmax,
    tstep,
    spins,
):

    P12S = projop(spins, "S")
    HZ = HamiltonianZeeman3D(spins, B, theta, phi, gamma)

    r_distances = [
        [r23A, r13A, r12A],  # r23A, r13A, r12A
        [r23B, r13B, r12B],  # r23B, r13B, r12B
    ]

    DipolarInteractions = [
        [DipolarInteraction(r, gamma) for r in r_distance] for r_distance in r_distances
    ]

    HA = HZ + HamiltonianDipolar3D(DipolarInteractions[0])
    HB = HZ + HamiltonianDipolar3D(DipolarInteractions[1])

    K = [kAB * np.eye(len(HA)), kBA * np.eye(len(HB))]

    nt = np.round((tmax / tstep + 1)).astype(int)
    rhoA = rhoB = P12S / 4
    propagatorA = expm((-1j * HA - 0.5 * K[0]) * tstep)
    propagatorB = expm((-1j * HB - 0.5 * K[1]) * tstep)
    singletfraction = np.zeros(nt, dtype=np.complex_)

    for i in range(0, nt):
        t = i * tstep
        singletfraction[i] = np.trace(np.dot((rhoA + rhoB), P12S))
        rhoAnew = (propagatorA @ rhoA @ np.conjugate(propagatorA)) + kBA * rhoB * tstep
        rhoBnew = (propagatorB @ rhoB @ np.conjugate(propagatorB)) + kAB * rhoA * tstep
        rhoA = rhoAnew / np.trace(rhoAnew + rhoBnew)
        rhoB = rhoBnew / np.trace(rhoAnew + rhoBnew)

    for i in range(0, nt):
        t = i * tstep
        singletfraction[i] = singletfraction[i] * np.exp(-k * t)
        # singletfraction[i] = singletfraction[i] * Kinetics(0, k, t, model="Exponential")

    weight = 4 * np.ones(nt)
    for i in range(2, nt - 1, 2):
        weight[i] = 2

    weight[0] = weight[-1] = 1

    singletyield = sum(singletfraction * weight) * tstep / 3

    return singletyield


# -------------------------------------------------------------------------------------------------------

# Plotting --------------------------------------------------------------------------------------------


def TimeEvolutionPlot2D(
    t_max, x, y, markercolour, y2, markercolour2, xlabel, ylabel, title
):
    fig = plt.figure()
    figure = plt.gcf()
    fig.set_facecolor("none")
    mpl.style.use("default")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.grid(False)
    ax.plot(x, y, color=markercolour, linewidth=2)
    plt.fill_between(x, y2, color=markercolour2, alpha=0.2)
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, t_max)
    ax.set_title(title, size=18)
    ax.legend([r"$P_i(t)$", r"$\Phi_i$"])
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)
    plt.tick_params(labelsize=14)
    figure.set_size_inches(10, 5)
    plt.show()


def DensityMatrixPlot2D(
    rhot, x_axis_labels, y_axis_labels, space="Hilbert", colourmap="viridis"
):

    match space:
        case "Hilbert":
            fig = plt.figure()
            figure = plt.gcf()
            x_axis_labels = x_axis_labels
            y_axis_labels = y_axis_labels
            sns.heatmap(
                np.abs(rhot),
                annot=True,
                linewidths=0.5,
                cmap=colourmap,
                cbar=False,
                xticklabels=x_axis_labels,
                yticklabels=y_axis_labels,
            )
            figure.set_size_inches(5, 5)
        #             plt.show()

        case "Liouville":
            dims = np.int64(np.sqrt(len(rhot)))
            fig = plt.figure()
            figure = plt.gcf()
            x_axis_labels = x_axis_labels
            y_axis_labels = y_axis_labels
            sns.heatmap(
                np.abs(np.reshape(rhot, (dims, dims))),
                annot=True,
                linewidths=0.5,
                cmap=colourmap,
                cbar=False,
                xticklabels=x_axis_labels,
                yticklabels=y_axis_labels,
            )
            figure.set_size_inches(5, 5)


#             plt.show()


def LinearEnergyLevelPlot2D(H, B, linecolour, title):

    eigval = np.linalg.eigh(H)
    E = np.real(eigval[0])  # 0 = eigenvalues, 1 = eigenvectors

    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.eventplot(E, orientation="vertical", color=linecolour, linewidth=3)
    ax.set_title(title, size=18)
    ax.set_ylabel("Spin state energy (J)", size=14)
    plt.tick_params(labelsize=14)


def EnergyLevelPlot2D(spins, HFC, B_max, B_steps, J, D, xlabel, title):

    H = HamiltonianZeeman_RadicalPair(spins, B_max)
    B = np.linspace(0, B_max, B_steps)
    E = np.zeros([B_steps, len(H)], dtype=np.complex_)

    for i, B0 in enumerate(B):
        HZ = HamiltonianZeeman_RadicalPair(spins, B0)
        HH = HamiltonianHyperfine(spins, HFC)
        HJ = HamiltonianExchange(spins, J)
        HD = HamiltonianDipolar(spins, D)
        Htotal = HZ + HH + HJ + HD
        eigval = np.linalg.eigh(Htotal)
        E[i] = eigval[0]  # 0 = eigenvalues, 1 = eigenvectors

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(B, np.real(E[:, ::-1]), linewidth=2)
    ax.set_title(title, size=18)
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel("Spin state energy (J)", size=14)
    plt.tick_params(labelsize=14)


def MARYplot2D(B, MARY, x_model_MARY, y_model_MARY, linecolour, title):
    fig = plt.figure()
    figure = plt.gcf()
    fig.set_facecolor("none")
    mpl.style.use("default")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.grid(False)
    ax.plot(B, np.real(MARY), linecolour, linewidth=2, label="Simulation")
    ax.plot(x_model_MARY, y_model_MARY, "k--", linewidth=1, label="Lorentzian fit")
    ax.legend()
    ax.set_title(title, size=16)
    ax.set_xlabel("$B_0$ ($mT$)", size=16)
    ax.set_ylabel("$MFE$ ($\%$)", size=16)
    plt.tick_params(labelsize=16)
    figure.set_size_inches(10, 5)


def plot2D(x, y, markercolour, errorbar, errorbarcolour, xscale, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.set_facecolor("none")
    plt.xscale(xscale)
    plt.rc("axes", edgecolor="k")
    ax.grid(False)
    plt.plot(x, y, markercolour, markersize=15)
    plt.errorbar(x, y, errorbar, ls="", color=errorbarcolour, linewidth=2)
    ax.set_title(title, size=18)
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    plt.show()


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


def plot3Dlog(X, Y, Z, xlabel, ylabel, zlabel, title, colourmap):
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "auto"})
    ax.set_facecolor("none")
    plt.rc("axes", edgecolor="k")
    ax.grid(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.axis("on")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colourmap, edgecolor="none")
    ax.set_title(title, size=18)
    ax.set_xlabel(xlabel, size=14)
    ax.set_ylabel(ylabel, size=14)
    ax.set_zlabel(zlabel, size=14)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)
    plt.show()


# ----------------------------------------------------------------------------------------------

# Monte Carlo random walk molecular diffusion -----------------------------------------------------------


def MC_randomwalk3D(n_steps, r_max, x_0, y_0, z_0, mut_D, del_T):
    Dab = mut_D
    deltaT = del_T
    deltaR = np.sqrt(6 * Dab * deltaT)  # diffusional motion

    x, y, z, dist, angle = (
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps + 1),
    )
    x[0], y[0], z[0] = x_0, y_0, z_0

    for i in range(1, n_steps):
        theta = np.pi * np.random.rand()
        angle[i] = theta
        phi = 2 * np.pi * np.random.rand()

        dist_sq = (
            (x[i] + x[i - 1]) ** 2 + (y[i] + y[i - 1]) ** 2 + (z[i] + z[i - 1]) ** 2
        )
        dist[i] = np.sqrt(dist_sq)

        x[i] = deltaR * np.cos(theta) * np.sin(phi)
        y[i] = deltaR * np.sin(theta) * np.sin(phi)
        z[i] = deltaR * np.cos(phi)

        x[i] += x[i - 1]
        y[i] += y[i - 1]
        z[i] += z[i - 1]

    f = 1e9

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "auto"})
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    ax.plot(x * f, y * f, z * f, alpha=0.9, color="cyan")
    ax.plot(x[0] * f, y[0] * f, z[0] * f, "bo", markersize=15)
    ax.plot(0, 0, 0, "mo", markersize=15)
    ax.set_title(
        "3D Monte Carlo random walk simulation for a radical pair in water", size=16
    )
    ax.set_xlabel("$X$ (nm)", size=14)
    ax.set_ylabel("$Y$ (nm)", size=14)
    ax.set_zlabel("$Z$ (nm)", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)
    plt.show()

    return x, y, z, dist, angle


def MC_randomwalk3D_cage(n_steps, r_max, x_0, y_0, z_0, mut_D, del_T):
    Dab = mut_D
    deltaT = del_T
    deltaR = np.sqrt(6 * Dab * deltaT)  # diffusional motion

    x, y, z, dist, angle = (
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps),
        np.zeros(n_steps + 1),
    )
    x[0], y[0], z[0] = x_0, y_0, z_0

    for i in range(1, n_steps):
        theta = np.pi * np.random.rand()
        angle[i] = theta
        phi = 2 * np.pi * np.random.rand()

        x[i] = deltaR * np.cos(theta) * np.sin(phi)
        y[i] = deltaR * np.sin(theta) * np.sin(phi)
        z[i] = deltaR * np.cos(phi)

        dist_sq = (
            (x[i] + x[i - 1]) ** 2 + (y[i] + y[i - 1]) ** 2 + (z[i] + z[i - 1]) ** 2
        )
        dist[i] = np.sqrt(dist_sq)

        if dist_sq > r_max**2:
            x[i] = x[i - 1] - x[i]
            y[i] = y[i - 1] - y[i]
            z[i] = z[i - 1] - z[i]
        else:
            x[i] += x[i - 1]
            y[i] += y[i - 1]
            z[i] += z[i - 1]

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x_frame = r_max * np.outer(np.sin(theta), np.cos(phi))
    y_frame = r_max * np.outer(np.sin(theta), np.sin(phi))
    z_frame = r_max * np.outer(np.cos(theta), np.ones_like(phi))

    f = 1e9

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "auto"})
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    ax.plot_wireframe(
        x_frame * f,
        y_frame * f,
        z_frame * f,
        color="k",
        alpha=0.1,
        rstride=1,
        cstride=1,
    )
    ax.plot(x * f, y * f, z * f, alpha=0.9, color="cyan")
    ax.plot(x[0] * f, y[0] * f, z[0] * f, "bo", markersize=15)
    ax.plot(0, 0, 0, "ro", markersize=15)
    #     ax.set_title("3D Monte Carlo random walk simulation for an encapsulated radical pair", size=16)
    ax.set_xlabel("$X$ (nm)", size=14)
    ax.set_ylabel("$Y$ (nm)", size=14)
    ax.set_zlabel("$Z$ (nm)", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)
    plt.show()

    return x, y, z, dist, angle


def MC_exchange_dipolar(n_steps, r_min, del_T, radA_x, dist, angle):

    r_min = radA_x[0]
    t = np.linspace(0, n_steps, n_steps)
    dist[0] = r_min
    r = dist
    t_tot = n_steps * del_T * 1e9
    t = np.linspace(0, t_tot, n_steps)

    # Constants and variables
    mu0 = 8.85418782e-12
    g = 2.0023
    muB = 9.274e-24
    hbar = 1.054e-34
    theta = angle[1::]
    r_D = r + r_min
    J0 = -570e-3
    alpha = 2e10

    # J-coupling
    J = -J0 * np.exp(-alpha * (r))

    # D-coupling
    D = (
        -(3 / 2)
        * (mu0 / (4 * np.pi))
        * ((g**2 * muB**2) / (hbar * r_D**3))
        * (3 * np.cos(theta) ** 2 - 1)
    ) / 1e3

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(t, (r + r_min) * 1e9, "r")
    ax.set_title("Time evolution of radical pair separation", size=16)
    ax.set_xlabel("$t$ (ns)", size=14)
    ax.set_ylabel("$r$ (nm)", size=14)
    plt.tick_params(labelsize=14)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(t, -J * 1e3)
    ax.set_title("Time evolution of the exchange interaction", size=16)
    ax.set_xlabel("$t$ (ns)", size=14)
    ax.set_ylabel("$J$ (mT)", size=14)
    plt.tick_params(labelsize=14)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    plt.plot(t, D, "g")
    ax.set_title("Time evolution of the dipolar interaction", size=16)
    ax.set_xlabel("$t$ (ns)", size=14)
    ax.set_ylabel("$D$ (mT)", size=14)
    plt.tick_params(labelsize=14)
    plt.show()

    return t, J, D


def MC_kSTD_kD(J, D, tau_c):
    # !this was changed (two tau_c's)
    mT2MHz = 28.025  # Conversion factor for mT to MHz

    # J-modulation rate
    JJ = np.var(J * 1e3)
    kSTD = (4 * tau_c) * JJ * 4 * np.pi**2 * 1e12 * mT2MHz  # (s^-1) J-modulation rate
    print("J-modulation rate (s^-1) =", "{:.2e}".format(kSTD))
    print("J-modulation rate (s) =", "{:.2e}".format(1 / kSTD))
    print()

    # D-modulation rate
    DD = np.var(D)
    kD = tau_c * DD * 4 * np.pi**2 * 1e12 * mT2MHz  # (s^-1) D-modulation rate
    print("D-modulation rate (s^-1) =", "{:.2e}".format(kD))
    print("D-modulation rate (s) =", "{:.2e}".format(1 / kD))

    return kSTD, kD


# ---------------------------------------------------------------------------------------------


# Kinetic simulations---------------------------------------------------------


def firstorder(C, t, k):
    Ca, Cb = C[0], C[1]
    # k = 1

    dAdt = -k * Ca
    dBdt = k * Ca

    return [dAdt, dBdt]


def electrontransfer(C, t, kab, kbr):
    Ca, Cb, Car, Cbr = C[0], C[1], C[2], C[3]

    dAdt = -kab * Ca * Cb
    dBdt = -kab * Ca * Cb + kbr * Cbr
    dArdt = kab * Ca * Cb
    dBrdt = kab * Ca * Cb - kbr * Cbr

    return [dAdt, dBdt, dArdt, dBrdt]


def cyclic(C, t, kab, kbcd, kca, kda):
    Ca, Cb, Cc, Cd = C[0], C[1], C[2], C[3]

    dAdt = -kab * Ca + kca * Cc + kda * Cd
    dBdt = kab * Ca - kbcd * Cb
    dCdt = (kbcd / 2) * Cb - kca * Cc
    dDdt = (kbcd / 2) * Cb - kda * Cd

    return [dAdt, dBdt, dCdt, dDdt]


def Bzero(C, t, kr, ke, kst):
    Cs, Ctp, Ct0, Ctm = C[0], C[1], C[2], C[3]

    dSdt = kst * (Ctp + Ct0 + Ctm) - (3 * kst + kr + ke) * Cs
    dTpdt = kst * Cs - (kst + ke) * Ctp
    dT0dt = kst * Cs - (kst + ke) * Ct0
    dTmdt = kst * Cs - (kst + ke) * Ctm

    return [dSdt, dTpdt, dT0dt, dTmdt]


def singlet_Bnonzero(C, t, kr, ke, kst, krlx, krlxp):
    Cs, Ct0 = C[0], C[1]

    dSdt = kst * Ct0 - (kst + kr + ke + 2 * krlxp) * Cs
    dT0dt = kst * Cs - (kst + ke + 2 * krlx) * Ct0

    return [dSdt, dT0dt]


def triplet_Bnonzero(C, t, kr, ke, kst, krlx, krlxp):
    Cs, Ctp, Ct0, Ctm = C[0], C[1], C[2], C[3]

    dSdt = kst * Ct0 + krlxp * (Ctp + Ctm) - (kst + kr + ke + 2 * krlxp) * Cs
    dT0dt = kst * Cs + krlx * (Ctp + Ctm) - (kst + ke + 2 * krlx) * Ct0
    dTpdt = krlxp * Cs + krlx * Ct0 - (ke + krlx + krlxp) * Ctp
    dTmdt = krlxp * Cs + krlx * Ct0 - (ke + krlx + krlxp) * Ctm

    return [dSdt, dTpdt, dT0dt, dTmdt]


def FreeRadical_Bzero(C, t, kr, ke, kst, kq, kp, Q):
    Cs, Ctp, Ct0, Ctm, Cfr = C[0], C[1], C[2], C[3], C[4]

    dSdt = kst * (Ctp + Ct0 + Ctm) - (3 * kst + kr + ke) * Cs
    dTpdt = kst * Cs - (kst + ke) * Ctp
    dT0dt = kst * Cs - (kst + ke) * Ct0
    dTmdt = kst * Cs - (kst + ke) * Ctm
    dFRdt = (kq * Q) * (Cs + Ctp + Ct0 + Ctm) - kp * Cfr

    return [dSdt, dTpdt, dT0dt, dTmdt, dFRdt]


def FreeRadical_singlet_Bnonzero(C, t, kr, ke, kst, krlx, krlxp, kq, kp, Q):
    Cs, Ct0, Cfr = C[0], C[1], C[2]

    dSdt = kst * Ct0 - (kst + kr + ke + 2 * krlxp) * Cs
    dT0dt = kst * Cs - (kst + ke + 2 * krlx) * Ct0
    dFRdt = (kq * Q) * (Cs + Ct0) - kp * Cfr

    return [dSdt, dT0dt, dFRdt]


def FreeRadical_triplet_Bnonzero(C, t, kr, ke, kst, krlx, krlxp, kq, kp, Q):
    Cs, Ctp, Ct0, Ctm, Cfr = C[0], C[1], C[2], C[3], C[4]

    dSdt = kst * Ct0 + krlxp * (Ctp + Ctm) - (kst + kr + ke + 2 * krlxp) * Cs
    dT0dt = kst * Cs + krlx * (Ctp + Ctm) - (kst + ke + 2 * krlx) * Ct0
    dTpdt = krlxp * Cs + krlx * Ct0 - (ke + krlx + krlxp) * Ctp
    dTmdt = krlxp * Cs + krlx * Ct0 - (ke + krlx + krlxp) * Ctm
    dFRdt = (kq * Q) * (Cs + Ctp + Ct0 + Ctm) - kp * Cfr

    return [dSdt, dTpdt, dT0dt, dTmdt, dFRdt]


def FAD_Bzero(C, t, kbet, pH, khfc, kd, k1, km1, krt):
    Crs, Ctpm, Ct0, Crtpm, Crt0 = C[0], C[1], C[2], C[3], C[4]
    Hp = 10 ** (-1 * pH)  # concentration of hydrogen ions

    dRSdt = -(kbet + 3 * khfc) * Crs + khfc * Crtpm + khfc * Crt0
    dTpmdt = km1 * Hp * Crtpm - (kd + k1 + krt) * Ctpm + 2 * krt * Ct0
    dT0dt = km1 * Hp * Crt0 + krt * Ctpm - (kd + k1 + 2 * krt) * Ct0
    dRTpmdt = (
        2 * khfc * Crs - (km1 * Hp + 2 * khfc) * Crtpm + 2 * khfc * Crt0 + k1 * Ctpm
    )
    dRT0dt = khfc * Crs + khfc * Crtpm - (3 * khfc + km1 * Hp) * Crt0 + k1 * Ct0

    return [dRSdt, dTpmdt, dT0dt, dRTpmdt, dRT0dt]


def FAD_Bnonzero(C, t, kbet, kr, pH, khfc, kd, k1, km1, krt):
    Crs, Ctpm, Ct0, Crtpm, Crt0 = C[0], C[1], C[2], C[3], C[4]
    Hp = 10 ** (-1 * pH)  # concentration of hydrogen ions

    dRSdt = -(kbet + khfc + 2 * kr) * Crs + kr * Crtpm + khfc * Crt0
    dTpmdt = km1 * Hp * Crtpm - (kd + k1 + krt) * Ctpm + 2 * krt * Ct0
    dT0dt = km1 * Hp * Crt0 + krt * Ctpm - (kd + k1 + 2 * krt) * Ct0
    dRTpmdt = 2 * kr * Crs - (km1 * Hp + 2 * kr) * Crtpm + 2 * kr * Crt0 + k1 * Ctpm
    dRT0dt = khfc * Crs + kr * Crtpm - (khfc + 2 * kr + km1 * Hp) * Crt0 + k1 * Ct0

    return [dRSdt, dTpmdt, dT0dt, dRTpmdt, dRT0dt]


def FAD_Bzero_quenching(C, t, kbet, pH, khfc, kd, k1, km1, krt, kq, kp, Q):
    Crs, Ctpm, Ct0, Crtpm, Crt0, Cfr = C[0], C[1], C[2], C[3], C[4], C[5]
    Hp = 10 ** (-1 * pH)  # concentration of hydrogen ions

    dRSdt = -(kbet + 3 * khfc) * Crs + khfc * Crtpm + khfc * Crt0
    dTpmdt = km1 * Hp * Crtpm - (kd + k1 + krt) * Ctpm + 2 * krt * Ct0
    dT0dt = km1 * Hp * Crt0 + krt * Ctpm - (kd + k1 + 2 * krt) * Ct0
    dRTpmdt = (
        2 * khfc * Crs - (km1 * Hp + 2 * khfc) * Crtpm + 2 * khfc * Crt0 + k1 * Ctpm
    )
    dRT0dt = khfc * Crs + khfc * Crtpm - (3 * khfc + km1 * Hp) * Crt0 + k1 * Ct0
    dFRdt = (kq * Q) * (Crs + Ctpm + Ct0 + Crtpm + Crt0) - kp * Cfr

    return [dRSdt, dTpmdt, dT0dt, dRTpmdt, dRT0dt, dFRdt]


def FAD_Bnonzero_quenching(C, t, kbet, kr, pH, khfc, kd, k1, km1, krt, kq, kp, Q):
    Crs, Ctpm, Ct0, Crtpm, Crt0, Cfr = C[0], C[1], C[2], C[3], C[4], C[5]
    Hp = 10 ** (-1 * pH)  # concentration of hydrogen ions

    dRSdt = -(kbet + khfc + 2 * kr) * Crs + kr * Crtpm + khfc * Crt0
    dTpmdt = km1 * Hp * Crtpm - (kd + k1 + krt) * Ctpm + 2 * krt * Ct0
    dT0dt = km1 * Hp * Crt0 + krt * Ctpm - (kd + k1 + 2 * krt) * Ct0
    dRTpmdt = 2 * kr * Crs - (km1 * Hp + 2 * kr) * Crtpm + 2 * kr * Crt0 + k1 * Ctpm
    dRT0dt = khfc * Crs + kr * Crtpm - (khfc + 2 * kr + km1 * Hp) * Crt0 + k1 * Ct0
    dFRdt = (kq * Q) * (Crs + Ctpm + Ct0 + Crtpm + Crt0) - kp * Cfr

    return [dRSdt, dTpmdt, dT0dt, dRTpmdt, dRT0dt, dFRdt]


# -----------------------------------------------------------------------------------
