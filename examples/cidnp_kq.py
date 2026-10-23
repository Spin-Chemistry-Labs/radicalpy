#! /usr/bin/env python

# MARY simulation with simple kinetic rate equations replacing Haberkorn
# superoperators.

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.classical import Rate, RateEquations
from radicalpy.experiments import mary
from radicalpy.simulation import LiouvilleSimulation, Molecule
from radicalpy.utils import is_fast_run


def rate_equations():
    ke = Rate(1e6, "k_{E}")  # radical pair separation to free radical
    kr = Rate(2e6, "k_{R}")  # reverse electron transfer of RP to groundstate
    kfr = Rate(1e5, "k_{FR}")  # free radical recombination

    FR = "FR"
    SS, STp, ST0, STm = "SS", "ST_+", "ST_0", "ST_-"
    TpS, TpTp, TpT0, TpTm = "T_+S", "T_+T_+", "T_+T_0", "T_+T_-"
    T0S, T0Tp, T0T0, T0Tm = "T_0S", "T_0T_+", "T_0T_0", "T_0T_-"
    TmS, TmTp, TmT0, TmTm = "T_-S", "T_-T_+", "T_-T_0", "T_-T_-"

    base = {}
    base[FR] = {FR: -kfr, SS: ke, TpTp: ke, T0T0: ke, TmTm: ke}

    base[SS] = {SS: -(kr + ke)}
    base[STp] = {STp: -(kr + ke)}
    base[ST0] = {ST0: -(kr + ke)}
    base[STm] = {STm: -(kr + ke)}

    base[TpS] = {TpS: -(kr + ke)}
    base[TpTp] = {TpTp: -ke}
    base[TpT0] = {TpT0: -ke}
    base[TpTm] = {TpTm: -ke}

    base[T0S] = {T0S: -(kr + ke)}
    base[T0Tp] = {T0Tp: -ke}
    base[T0T0] = {T0T0: -ke}
    base[T0Tm] = {T0Tm: -ke}

    base[TmS] = {TmS: -(kr + ke)}
    base[TmTp] = {TmTp: -ke}
    base[TmT0] = {TmT0: -ke}
    base[TmTm] = {TmTm: -ke}

    rate_eq = RateEquations(base)
    return rate_eq, kfr


def main(tmax=5e-6, dt=5e-9, Bmax=20, dB=0.25, num_samples=10):
    time = np.arange(0, tmax, dt)
    B = np.arange(0, Bmax + dB, dB)

    rate_eq, kfr = rate_equations()
    mat = np.asarray(rate_eq.matrix.todense(), dtype=complex)
    rho0 = np.array(
        [
            0,  # FR
            0,  # SS
            0,  # ST+
            0,  # ST0
            0,  # ST-
            0,  # T+S
            1 / 3,  # T+T+
            0,  # T+T0
            0,  # T+T-
            0,  # T0S
            0,  # T0T+
            1 / 3,  # T0T0
            0,  # T0T-
            0,  # T-S
            0,  # T-T+
            0,  # T-T0
            1 / 3,  # T-T-
        ],
        dtype=complex,
    )

    r1 = Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.5])
    r2 = Molecule("radical 2")
    sim = LiouvilleSimulation([r1, r2])

    results = mary(
        sim,
        init_state=rho0,
        obs_state=None,
        # radical_pair=[1, 17],
        time=time,
        B=B,
        D=0,
        J=0,
        kinetics=[KineQuantumKinetics(mat)],
        relaxations=[],
    )

    dt = time[1] - time[0]
    free_radical = np.real(results["yield"][:, 0, :])
    recombination = kfr.value * free_radical
    product_yields = np.cumulative_sum(recombination, axis=0) * dt
    product_yield_sums = product_yields[-1, :]
    hary = (product_yield_sums - product_yield_sums[0]) / product_yield_sums[0] * 100
    lfe = hary.min() if abs(hary.min()) > abs(hary.max()) else hary.max()
    hfe = hary[-1]

    plt.plot(B, hary, color="red", linewidth=2)
    plt.xlabel("$B_0$ (mT)")
    plt.ylabel("MFE (%)")
    plt.title("1H radical pair kinetic-quantum MARY")

    print(f"LFE = {lfe: .2f} %")
    print(f"HFE = {hfe: .2f} %")

    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(tmax=1e-6, dt=2e-8, Bmax=5, dB=1, num_samples=10)
    else:
        main()
