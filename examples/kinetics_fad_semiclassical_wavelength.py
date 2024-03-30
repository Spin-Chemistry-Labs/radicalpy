#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from radicalpy.classical import Rate, RateEquations, latex_eqlist_to_align, latexify
from radicalpy.experiments import semiclassical_kinetics_mary
from radicalpy.simulation import Molecule, SemiclassicalSimulation
from radicalpy.utils import Bhalf_fit
from radicalpy.relaxation import RandomFields, SingletTripletDephasing
from radicalpy.plot import plot_3d_results, plot_bhalf_time, plot_general
import radicalpy as rp


def main():

    # Parameters
    time = np.arange(0, 10e-6, 10e-9)
    Bs = np.arange(0, 35, 0.5)
    num_samples = 1000
    scale_factor = 4e-2
    kr = 7e7  # radical pair relaxation rate
    # relaxation = RandomFields(kr)  # relaxation model
    # relaxation = SingletTripletDephasing(kr)  # relaxation model

    # Axes for orientation of 3D plots
    # azim = -135
    # dist = 10
    # elev = 35

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
    groundstate_spectrum = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_groundstate_spectrum.txt")
        ]
    )
    groundstate_wavelength = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_groundstate_spectrum_wavelength.txt")
        ]
    )
    emission_spectrum = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_emission_spectrum.txt")
        ]
    )
    emission_wavelength = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_emission_spectrum_wavelength.txt")
        ]
    )
    radical_spectrum = radical_spectrum[0, :] * 1e3
    triplet_spectrum = triplet_spectrum[0, :] * 1e3
    wavelength = wavelength[0, :]
    groundstate_spectrum = groundstate_spectrum[0, :]
    groundstate_wavelength = groundstate_wavelength[0, :]
    emission_spectrum = emission_spectrum[0, :]
    emission_wavelength = emission_wavelength[0, :]

    flavin = Molecule.all_nuclei("fad")
    adenine = Molecule.all_nuclei("fad")
    sim = SemiclassicalSimulation([flavin, adenine])
    bhalf = rp.estimations.Bhalf_theoretical(sim)
    khfc_new = rp.estimations.k_ST_mixing(bhalf)
    khfc = 8e7  # spin-state mixing rate
    khfc_ratio = khfc_new / khfc

    # Kinetic simulation of FAD at pH 2.1.

    # FAD kinetic parameters
    kex = Rate(1e4, "k_{ex}")  # groundstate excitation rate
    kfl = Rate(3.55e8, "k_{fl}")  # fluorescence rate
    kic = Rate(1.28e9, "k_{IC}")  # internal conversion rate
    kisc = Rate(3.64e8, "k_{ISC}")  # intersystem crossing rate
    kd = Rate(3e5, "k_d")  # protonated triplet to ground state
    k1 = Rate(7e6, "k_1")  # protonated triplet to RP
    km1 = Rate(2.7e9 * khfc_ratio, "k_{-1}")  # RP to protonated triplet
    krt = Rate(1e9, "k^R_T")  # triplet state relaxation rate
    kbet = Rate(1.1e7 * khfc_ratio, "k_{BET}")  # singlet recombination rate
    pH = 2.1  # pH of the solution
    Hp = Rate(10**-pH, "H^+")  # concentration of hydrogen ions

    # Rate equations
    S0, S1, T1p, T10, T1m = "S_0", "S_1", "T_1^+", "T_1^0", "T_1^-"
    SS, STp, ST0, STm = "SS", "ST_+", "ST_0", "ST_-"
    TpS, TpTp, TpT0, TpTm = "T_+S", "T_+T_+", "T_+T_0", "T_+T_-"
    T0S, T0Tp, T0T0, T0Tm = "T_0S", "T_0T_+", "T_0T_0", "T_0T_-"
    TmS, TmTp, TmT0, TmTm = "T_-S", "T_-T_+", "T_-T_0", "T_-T_-"

    base = {}
    base[S0] = {
        S0: -kex,
        S1: kfl + kic,
        T1p: kd,
        T10: kd,
        T1m: kd,
        SS: kbet,
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
        TpTp: -(km1 * Hp),
    }
    base[TpT0] = {
        TpT0: -(km1 * Hp),
    }
    base[TpTm] = {
        TpTm: -(km1 * Hp),
    }

    base[T0S] = {
        T0S: -(kbet + km1 * Hp) / 2,
    }
    base[T0Tp] = {
        T0Tp: -(km1 * Hp),
    }
    base[T0T0] = {
        T10: k1,
        T0T0: -(km1 * Hp),
    }
    base[T0Tm] = {
        T0Tm: -(km1 * Hp),
    }

    base[TmS] = {
        TmS: -(kbet + km1 * Hp) / 2,
    }
    base[TmTp] = {
        TmTp: -(km1 * Hp),
    }
    base[TmT0] = {
        TmT0: -(km1 * Hp),
    }
    base[TmTm] = {
        T1m: k1,
        TmTm: -(km1 * Hp),
    }

    rate_eq = RateEquations(base)
    mat = rate_eq.matrix.todense()
    rho0 = np.array(
        [
            0,  # S0
            0,  # S1
            1 / 3,  # T1+
            1 / 3,  # T10
            1 / 3,  # T1-
            0,  # SS
            0,  # ST+
            0,  # ST0
            0,  # ST-
            0,  # T+S
            0,  # T+T+
            0,  # T+T0
            0,  # T+T-
            0,  # T0S
            0,  # T0T+
            0,  # T0T0
            0,  # T0T-
            0,  # T-S
            0,  # T-T+
            0,  # T-0
            0,  # T-T-
        ]
    )

    latex_equations = latex_eqlist_to_align(latexify(base))
    # print(latex_equations)

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
        relaxations=[RandomFields(kr), SingletTripletDephasing(kr)],
    )

    total_yield = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    zero_field = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    mary = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    total_yield_groundstate = np.zeros(
        (len(time), len(Bs), len(groundstate_wavelength)), dtype=complex
    )
    zero_field_groundstate = np.zeros(
        (len(time), len(Bs), len(groundstate_wavelength)), dtype=complex
    )
    mary_groundstate = np.zeros(
        (len(time), len(Bs), len(groundstate_wavelength)), dtype=complex
    )
    total_yield_emission = np.zeros(
        (len(time), len(Bs), len(emission_wavelength)), dtype=complex
    )
    zero_field_emission = np.zeros(
        (len(time), len(Bs), len(emission_wavelength)), dtype=complex
    )
    mary_emission = np.zeros(
        (len(time), len(Bs), len(emission_wavelength)), dtype=complex
    )

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
    groundstate_yield = results["yield"][:, 0, :]

    for i, r in enumerate(radical_spectrum):
        for j, t in enumerate(triplet_spectrum):
            total_yield[:, :, j + 1] = (
                (r * radical_pair_yield) + (t * triplet_yield)
            ) * scale_factor

    for i in range(0, len(wavelength)):
        for j in range(0, len(Bs)):
            zero_field[:, j, i] = total_yield[:, 0, i]

    for i, g in enumerate(groundstate_spectrum):
        total_yield_groundstate[:, :, i] = (g * groundstate_yield) * (scale_factor)

    for i in range(0, len(groundstate_wavelength)):
        for j in range(0, len(Bs)):
            zero_field_groundstate[:, j, i] = total_yield_groundstate[:, 0, i]

    for i, g in enumerate(emission_spectrum):
        total_yield_emission[:, :, i] = (g * groundstate_yield) * 100

    for i in range(0, len(emission_wavelength)):
        for j in range(0, len(Bs)):
            zero_field_emission[:, j, i] = total_yield_emission[:, 0, i]

    mary = np.real(total_yield - zero_field)
    mary_groundstate = np.real(total_yield_groundstate - zero_field_groundstate)
    mary_emission = np.real(total_yield_emission - zero_field_emission)

    mfe_max = np.zeros(len(wavelength), dtype=complex)
    for i in range(1, len(wavelength)):
        mfe_max[i] = mary[:, -1, i].max()

    mfe_groundstate_max = np.zeros(len(groundstate_wavelength), dtype=complex)
    for i in range(1, len(groundstate_wavelength)):
        mfe_groundstate_max[i] = mary_groundstate[:, -1, i].min()

    mfe_emission_max = np.zeros(len(emission_wavelength), dtype=complex)
    for i in range(1, len(emission_wavelength)):
        mfe_emission_max[i] = mary_emission[:, -1, i].min()

    xlabel = "Wavelength / nm"
    ylabel = "$\Delta \Delta A$"
    plot_general(
        groundstate_wavelength[1:],
        np.real(mfe_groundstate_max)[1:],
        xlabel,
        ylabel,
        "r-",
        label=f"{Bs.max(): .0f} mT",
    )
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "$\Delta F$"
    plot_general(
        emission_wavelength[1:],
        np.real(mfe_emission_max)[1:],
        xlabel,
        ylabel,
        "r-",
        label=f"{Bs.max(): .0f} mT",
    )
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "$\Delta \Delta A$"
    plot_general(
        wavelength,
        np.real(mfe_max),
        xlabel,
        ylabel,
        "r-",
        label=f"{Bs.max(): .0f} mT",
    )
    path = __file__[:-3] + f"_{2}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # np.savetxt(
    #     "./examples/data/fad_kinetics/semiclassical_kinetics_new.txt", mary[:, -1, 5]
    # )
    # np.savetxt("./examples/data/fad_kinetics/semiclassical_kinetics_time.txt", time)

    n = 100
    factor = 1e6

    xlabel = "Wavelength / nm"
    ylabel = "$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            groundstate_wavelength,
            mary_groundstate[i, -1, :],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * 1e6: .0f} $\mu s$",
        )
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "$B_0$ / mT"
    ylabel = "$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            Bs,
            mary_groundstate[i, :, 5],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * 1e6: .0f} $\mu s$",
        )
    path = __file__[:-3] + f"_{4}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "Time / $\mu s$"
    ylabel = "$\Delta \Delta A$"
    for i in range(0, len(Bs), 5):
        plot_general(
            time,
            mary_groundstate[:, i, 5],
            xlabel,
            ylabel,
            style="-",
            label=f"{Bs[i]: .1f} mT",
            factor=1e6,
        )
    path = __file__[:-3] + f"_{5}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "$\Delta F$"
    for i in range(0, len(time), n):
        plot_general(
            emission_wavelength,
            mary_emission[i, -1, :],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * 1e6: .0f} $\mu s$",
        )
    path = __file__[:-3] + f"_{6}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "$B_0$ / mT"
    ylabel = "$\Delta F$"
    for i in range(0, len(time), n):
        plot_general(
            Bs,
            mary_emission[i, :, 5],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * 1e6: .0f} $\mu s$",
        )
    path = __file__[:-3] + f"_{7}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "Time / $\mu s$"
    ylabel = "$\Delta F$"
    for i in range(0, len(Bs), 5):
        plot_general(
            time,
            mary_emission[:, i, 5],
            xlabel,
            ylabel,
            style="-",
            label=f"{Bs[i]: .1f} mT",
            factor=1e6,
        )
    path = __file__[:-3] + f"_{8}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            wavelength,
            mary[i, -1, :],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * 1e6: .0f} $\mu s$",
        )
    path = __file__[:-3] + f"_{9}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "$B_0$ / mT"
    ylabel = "$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            Bs,
            mary[i, :, 5],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * 1e6: .0f} $\mu s$",
        )
    path = __file__[:-3] + f"_{10}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = "Time / $\mu s$"
    ylabel = "$\Delta \Delta A$"
    for i in range(0, len(Bs), 5):
        plot_general(
            time,
            mary[:, i, 5],
            xlabel,
            ylabel,
            style="-",
            label=f"{Bs[i]: .1f} mT",
            factor=1e6,
        )
    path = __file__[:-3] + f"_{11}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # Calculate time evolution of the B1/2
    bhalf_time = np.zeros((len(mary_groundstate)))
    fit_time = np.zeros((len(Bs), len(mary_groundstate)))
    fit_error_time = np.zeros((2, len(mary_groundstate)))
    R2_time = np.zeros((len(mary_groundstate)))

    for i in range(2, len(mary_groundstate)):
        (
            bhalf_time[i],
            fit_time[:, i],
            fit_error_time[:, i],
            R2_time[i],
        ) = Bhalf_fit(Bs, mary_groundstate[i, :, -1])

    plot_bhalf_time(time, bhalf_time, fit_error_time)
    path = __file__[:-3] + f"_{18}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    bhalf_time = np.zeros((len(mary_emission)))
    fit_time = np.zeros((len(Bs), len(mary_emission)))
    fit_error_time = np.zeros((2, len(mary_emission)))
    R2_time = np.zeros((len(mary_emission)))

    for i in range(2, len(mary_emission)):
        (
            bhalf_time[i],
            fit_time[:, i],
            fit_error_time[:, i],
            R2_time[i],
        ) = Bhalf_fit(Bs, mary_emission[i, :, -1])

    plot_bhalf_time(time, bhalf_time, fit_error_time)
    path = __file__[:-3] + f"_{19}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    bhalf_time = np.zeros((len(mary)))
    fit_time = np.zeros((len(Bs), len(mary)))
    fit_error_time = np.zeros((2, len(mary)))
    R2_time = np.zeros((len(mary)))

    for i in range(2, len(mary)):
        (
            bhalf_time[i],
            fit_time[:, i],
            fit_error_time[:, i],
            R2_time[i],
        ) = Bhalf_fit(Bs, mary[i, :, -1])

    plot_bhalf_time(time, bhalf_time, fit_error_time)
    path = __file__[:-3] + f"_{20}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3D plots
    xlabel = "Wavelength / nm"
    ylabel = "Time / $\mu s$"
    zlabel = "$\Delta \Delta A$"
    plot_3d_results(
        groundstate_wavelength,
        results["ts"],
        mary_groundstate[:, -1, :],
        xlabel,
        ylabel,
        zlabel,
    )
    path = __file__[:-3] + f"_{12}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "$B_0$ / mT"
    zlabel = "$\Delta \Delta A$"
    plot_3d_results(
        groundstate_wavelength,
        results["Bs"],
        mary_groundstate[250, :, :],
        xlabel,
        ylabel,
        zlabel,
        factor=1,
    )
    path = __file__[:-3] + f"_{13}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "Time / $\mu s$"
    zlabel = "$\Delta \Delta A$"
    plot_3d_results(
        emission_wavelength,
        results["ts"],
        mary_emission[:, -1, :],
        xlabel,
        ylabel,
        zlabel,
    )
    path = __file__[:-3] + f"_{14}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "$B_0$ / mT"
    zlabel = "$\Delta \Delta A$"
    plot_3d_results(
        emission_wavelength,
        results["Bs"],
        mary_emission[250, :, :],
        xlabel,
        ylabel,
        zlabel,
        factor=1,
    )
    path = __file__[:-3] + f"_{15}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "Time / $\mu s$"
    zlabel = "$\Delta \Delta A$"
    plot_3d_results(
        wavelength,
        results["ts"],
        mary[:, -1, :],
        xlabel,
        ylabel,
        zlabel,
    )
    path = __file__[:-3] + f"_{16}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = "Wavelength / nm"
    ylabel = "$B_0$ / mT"
    zlabel = "$\Delta \Delta A$"
    plot_3d_results(
        wavelength,
        results["Bs"],
        mary[250, :, :],
        xlabel,
        ylabel,
        zlabel,
        factor=1,
    )
    path = __file__[:-3] + f"_{17}.png"
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
