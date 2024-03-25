#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as p
from radicalpy.classical import Rate, RateEquations, latex_eqlist_to_align, latexify
from radicalpy.experiments import semiclassical_kinetics_mary
from radicalpy.plot import plot_3d_results, plot_bhalf_time
from radicalpy.simulation import Molecule, SemiclassicalSimulation
from radicalpy.utils import Bhalf_fit
from radicalpy.relaxation import RandomFields


def main():

    # Parameters
    time = np.arange(0, 20e-6, 10e-9)
    Bs = np.arange(0, 30, 0.5)
    num_samples = 400
    scale_factor = 4e-4
    kr = 0  # 1.7e6  # radical pair relaxation rate
    relaxation = RandomFields(kr)  # relaxation model

    # Load reference spectra
    path = "./examples/data/fad_kinetics"
    radical_spectrum = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_radical_spectrum.txt")
        ]
    )
    triplet_spectrum = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_triplet_spectrum.txt")
        ]
    )
    wavelength = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_radical_wavelength.txt")
        ]
    )
    radical_spectrum = radical_spectrum[0, :] * 1e3
    triplet_spectrum = triplet_spectrum[0, :] * 1e3
    wavelength = wavelength[0, :]

    # Kinetic simulation of FAD at pH 2.1.

    # FAD kinetic parameters
    kex = Rate(1e4, "k_{ex}")  # groundstate excitation rate
    kfl = Rate(3.55e8, "k_{fl}")  # fluorescence rate
    kic = Rate(1.28e9, "k_{IC}")  # internal conversion rate
    kisc = Rate(3.64e8, "k_{ISC}")  # intersystem crossing rate
    kd = Rate(3e5, "k_d")  # protonated triplet to ground state
    k1 = Rate(7e6, "k_1")  # protonated triplet to RP
    km1 = Rate(2.7e9, "k_{-1}")  # RP to protonated triplet
    krt = Rate(1e9, "k^R_T")  # triplet state relaxation rate
    kbet = Rate(1.1e7, "k_{BET}")  # singlet recombination rate
    pH = 2.1  # pH of the solution
    Hp = Rate(10**-pH, "H^+")  # concentration of hydrogen ions

    # Quenching kinetic parameters
    kq = Rate(0, "k_q")  # 1e9  # quenching rate
    kp = Rate(0, "k_p")  # 3.3e3  # free radical recombination
    Q = Rate(0, "Q")  # 1e-3  # quencher concentration

    # Rate equations
    S0, S1, T1p, T10, T1m = "S0", "S1", "T1+", "T10", "T1-"
    SS, STp, ST0, STm = "SS", "ST+", "ST0", "ST-"
    TpS, TpTp, TpT0, TpTm = "T+S", "T+T+", "T+T0", "T+T-"
    T0S, T0Tp, T0T0, T0Tm = "T0S", "T0T+", "T0T0", "T0T-"
    TmS, TmTp, TmT0, TmTm = "T-S", "T-T+", "T-T0", "T-T-"
    FR = "FR"

    base = {}
    base[S0] = {
        S0: -kex,
        S1: kfl + kic,
        T1p: kd,
        T10: kd,
        T1m: kd,
        SS: kbet,
        FR: kp,
    }
    base[S1] = {
        S0: kex,
        S1: -(kfl + kic + 3 * kisc),
    }
    base[T1p] = {
        S1: kisc,
        T1p: -(kd + k1 + krt),
        T10: krt,
        TpTp: km1 * Hp,
    }
    base[T10] = {
        S1: kisc,
        T1p: krt,
        T10: -(kd + k1 + 2 * krt),
        T1m: krt,
        T0T0: km1 * Hp,
    }
    base[T1m] = {
        S1: kisc,
        T10: krt,
        T1m: -(kd + k1 + krt),
        TmTm: km1 * Hp,
    }
    base[SS] = {
        SS: -(kbet),
    }
    base[STp] = {
        STp: -(kbet + km1 * Hp) / 2,
    }
    base[ST0] = {
        ST0: -(kbet + km1 * Hp) / 2,
    }
    base[STm] = {
        STm: -(kbet + km1 * Hp) / 2,
    }
    base[TpS] = {
        TpS: -(kbet + km1 * Hp) / 2,
    }
    base[TpTp] = {
        T1p: k1,
        TpTp: -km1 * Hp,
    }
    base[TpT0] = {
        TpT0: -km1 * Hp,
    }
    base[TpTm] = {
        TpTm: -km1 * Hp,
    }
    base[T0S] = {
        T0S: -(kbet + km1 * Hp) / 2,
    }
    base[T0Tp] = {
        T0Tp: -km1 * Hp,
    }
    base[T0T0] = {
        T10: k1,
        T0T0: -km1 * Hp,
    }
    base[T0Tm] = {
        T0Tm: -km1 * Hp,
    }
    base[TmS] = {
        TmS: -(kbet + km1 * Hp) / 2,
    }
    base[TmTp] = {
        TmTp: -km1 * Hp,
    }
    base[TmT0] = {
        TmT0: -km1 * Hp,
    }
    base[TmTm] = {
        T1m: k1,
        TmTm: -km1 * Hp,
    }
    base[FR] = {
        SS: kq * Q,
        STp: kq * Q,
        ST0: kq * Q,
        STm: kq * Q,
        TpS: kq * Q,
        TpTp: kq * Q,
        TpT0: kq * Q,
        TpTm: kq * Q,
        T0S: kq * Q,
        T0Tp: kq * Q,
        T0T0: kq * Q,
        T0Tm: kq * Q,
        TmS: kq * Q,
        TmTp: kq * Q,
        TmT0: kq * Q,
        TmTm: kq * Q,
        FR: -kp,
    }

    rate_eq = RateEquations(base)
    mat = rate_eq.matrix.todense()
    rho0 = np.array(
        [0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    latex_equations = latex_eqlist_to_align(latexify(base))
    # print(latex_equations)

    flavin = Molecule.all_nuclei("fad")
    adenine = Molecule.all_nuclei("fad")
    sim = SemiclassicalSimulation([flavin, adenine])

    results = semiclassical_kinetics_mary(
        sim,
        num_samples,
        rho0,
        radical_pair=[5, 21],
        ts=time,
        Bs=Bs,
        D=0,
        J=0,
        kinetics=mat,
        relaxations=[relaxation],
    )

    zero_field = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    total_yield = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    mary = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)

    radical_pair_yield = (
        results["yield"][:, 5, :]
        + results["yield"][:, 10, :]
        + results["yield"][:, 15, :]
        + results["yield"][:, 20, :]
    )
    triplet_yield = (
        results["yield"][:, 2, :]
        + results["yield"][:, 3, :]
        + results["yield"][:, 4, :]
    )
    free_radical_yield = results["yield"][:, 21, :]
    for i, r in enumerate(radical_spectrum):
        for j, t in enumerate(triplet_spectrum):
            total_yield[:, :, j + 1] = (
                (r * radical_pair_yield) + (t * triplet_yield)  # + free_radical_yield
            ) * scale_factor

    for i in range(0, len(wavelength)):
        for j in range(0, len(Bs)):
            zero_field[:, j, i] = total_yield[:, 0, i]

    mary = np.real(total_yield - zero_field)

    mfe_max = np.zeros(len(wavelength), dtype=complex)
    for i in range(1, len(wavelength)):
        mfe_max[i] = mary[:, -1, i].max()

    plt.figure()
    plt.plot(wavelength, np.real(mfe_max), "ro", linewidth=3)
    plt.xlabel("Wavelength / nm", size=18)
    plt.ylabel("$\Delta \Delta A$", size=18)
    plt.tick_params(labelsize=14)
    plt.gcf().set_size_inches(10, 5)
    # plt.show()
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300)

    # Plot absorption TR-MARY and B1/2 time evolution
    factor = 1e6
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    ax.set_facecolor("none")
    ax.grid(False)
    X, Y = np.meshgrid(wavelength, results["ts"])
    ax.plot_surface(
        X,
        Y * factor,
        np.real(mary[:, -1, :]),
        facecolors=cmap.to_rgba(mary[:, -1, :].real),
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel("Wavelength / nm", size=18)
    ax.set_ylabel("Time / $\mu s$", size=18)
    ax.set_zlabel("$\Delta \Delta A$", size=18)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    # plt.show()
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path, dpi=300)

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    ax.set_facecolor("none")
    ax.grid(False)
    X, Y = np.meshgrid(wavelength, results["Bs"])
    ax.plot_surface(
        X,
        Y,
        np.real(mary[250, :, :]),
        facecolors=cmap.to_rgba(mary[250, :, :].real),
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel("Wavelength / nm", size=18)
    ax.set_ylabel("$B_0$ / mT", size=18)
    ax.set_zlabel("$\Delta \Delta A$", size=18)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    # plt.show()
    path = __file__[:-3] + f"_{2}.png"
    plt.savefig(path, dpi=300)

    # np.savetxt(
    #     "./examples/data/fad_kinetics/semiclassical_kinetics_new.txt",
    #     mary[:, 1],
    # )
    # np.savetxt("./examples/data/fad_kinetics/semiclassical_kinetics_time.txt", time)

    # Calculate time evolution of the B1/2
    # bhalf_time = np.zeros((len(mary), len(wavelength)))
    # fit_time = np.zeros((len(Bs), len(mary), len(wavelength)))
    # fit_error_time = np.zeros((2, len(mary), len(wavelength)))
    # R2_time = np.zeros((len(mary), len(wavelength)))

    # for i in range(2, len(mary)):
    #     for j in range(1, len(wavelength)):
    #         (
    #             bhalf_time[i, j],
    #             fit_time[:, i, j],
    #             fit_error_time[:, i, j],
    #             R2_time[i, j],
    #         ) = Bhalf_fit(Bs, mary[:, :, j])

    # plot_bhalf_time(time, bhalf_time, fit_error_time)
    # plt.figure()
    # for j in range(0, len(wavelength, 5)):
    #     for i in range(2, len(time), 35):
    #         plt.plot(time[i] * factor, bhalf_time[i, j], "ro", linewidth=3)
    #         plt.errorbar(
    #             time[i, j] * factor,
    #             bhalf_time[i, j],
    #             fit_error_time[1, i, j],
    #             color="k",
    #             linewidth=2,
    #         )
    #     plt.xlabel("Time / $\mu s$", size=18)
    #     plt.ylabel("$B_{1/2}$ / mT", size=18)
    #     plt.tick_params(labelsize=14)
    #     plt.gcf().set_size_inches(10, 5)
    # path = __file__[:-3] + f"_{3}.png"
    # plt.savefig(path, dpi=300)


if __name__ == "__main__":
    main()
