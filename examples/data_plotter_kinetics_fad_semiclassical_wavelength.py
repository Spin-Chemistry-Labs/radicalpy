#! /usr/bin/env python


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.plot import plot_3d_results, plot_bhalf_time, plot_general
from radicalpy.utils import Bhalf_fit


def main():

    # Parameters
    num_samples = 50
    scale_factor = 2.5e-2  # 4e-1
    # Axes for orientation of 3D plots
    azim = 135
    dist = 10
    elev = 35

    # Load reference spectra
    path = Path("./examples/data/fad_kinetics")
    radical_spectrum = np.genfromtxt(path / "fad_radical_spectrum.txt")
    triplet_spectrum = np.genfromtxt(path / "fad_triplet_spectrum.txt")
    wavelength = np.genfromtxt(path / "fad_radical_wavelength.txt")
    groundstate_spectrum = np.genfromtxt(path / "fad_groundstate_spectrum.txt")
    groundstate_wavelength = np.genfromtxt(
        path / "fad_groundstate_spectrum_wavelength.txt"
    )
    emission_spectrum = np.genfromtxt(path / "fad_emission_spectrum.txt")
    emission_wavelength = np.genfromtxt(path / "fad_emission_spectrum_wavelength.txt")
    radical_spectrum = radical_spectrum * 1e3
    triplet_spectrum = triplet_spectrum * 1e3
    wavelength = wavelength
    groundstate_spectrum = groundstate_spectrum
    groundstate_wavelength = groundstate_wavelength
    emission_spectrum = emission_spectrum
    emission_wavelength = emission_wavelength

    result = np.load(
        "./examples/data/fad_kinetics/results_new.npy",
        allow_pickle=True,
    ).item()

    time = result["ts"]
    Bs = result["Bs"]
    results = result["yield"]

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
    total_yield_rp = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    zero_field_rp = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    total_yield_trip = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    zero_field_trip = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)

    radical_pair_yield = (
        results[:, 5, :] + results[:, 10, :] + results[:, 15, :] + results[:, 20, :]
    )
    triplet_yield = results[:, 2, :] + results[:, 3, :] + results[:, 4, :]
    groundstate_yield = results[:, 0, :]

    for i, r in enumerate(radical_spectrum):
        for j, t in enumerate(triplet_spectrum):
            total_yield[:, :, j + 1] = (
                (r * radical_pair_yield) + (t * (2 * triplet_yield))
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

    for i, r in enumerate(radical_spectrum):
        total_yield_rp[:, :, i] = ((r * radical_pair_yield)) * scale_factor

    for i in range(0, len(wavelength)):
        for j in range(0, len(Bs)):
            zero_field_rp[:, j, i] = total_yield_rp[:, 0, i]

    for i, t in enumerate(triplet_spectrum):
        total_yield_trip[:, :, i] = ((t * (2 * triplet_yield))) * scale_factor

    for i in range(0, len(wavelength)):
        for j in range(0, len(Bs)):
            zero_field_trip[:, j, i] = total_yield_trip[:, 0, i]

    mary = np.real(total_yield - zero_field)
    mary_groundstate = np.real(total_yield_groundstate - zero_field_groundstate)
    mary_emission = np.real(total_yield_emission - zero_field_emission)
    mary_rp = np.real(total_yield_rp - zero_field_rp)
    mary_trip = np.real(total_yield_trip - zero_field_trip)

    mfe_max = np.zeros(len(wavelength), dtype=complex)
    for i in range(1, len(wavelength)):
        mfe_max[i] = mary[:, -1, i].max()

    mfe_groundstate_max = np.zeros(len(groundstate_wavelength), dtype=complex)
    for i in range(1, len(groundstate_wavelength)):
        mfe_groundstate_max[i] = mary_groundstate[:, -1, i].min()

    mfe_emission_max = np.zeros(len(emission_wavelength), dtype=complex)
    for i in range(1, len(emission_wavelength)):
        mfe_emission_max[i] = mary_emission[:, -1, i].min()

    xlabel = r"Wavelength / nm"
    ylabel = r"$\Delta \Delta A$"
    plot_general(
        groundstate_wavelength[1:],
        np.real(mfe_groundstate_max)[1:],
        xlabel,
        ylabel,
        colors="b",
        label=f"{Bs.max(): .0f} mT",
    )
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"$\Delta I_F$"
    plot_general(
        emission_wavelength[1:],
        np.real(mfe_emission_max)[1:],
        xlabel,
        ylabel,
        colors="b",
        label=f"{Bs.max(): .0f} mT",
    )
    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"$\Delta \Delta A$"
    plot_general(
        wavelength,
        np.real(mfe_max),
        xlabel,
        ylabel,
        colors="b",
        label=f"{Bs.max(): .0f} mT",
    )
    path = __file__[:-3] + f"_{2}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    n = 100
    f = 25
    wl = 17
    factor = 1e6
    colors_time = plt.colormaps.get_cmap("viridis").resampled(len(time)).colors
    colors_field = plt.colormaps.get_cmap("viridis").resampled(len(Bs)).colors

    xlabel = r"Wavelength / nm"
    ylabel = r"$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            groundstate_wavelength,
            mary_groundstate[i, -1, :],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * factor: .0f} $\\mu s$",
            colors=colors_time[i],
        )
    path = __file__[:-3] + f"_{3}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"$B_0$ / mT"
    ylabel = r"$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            Bs,
            mary_groundstate[i, :, wl],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * factor: .0f} $\\mu s$",
            colors=colors_time[i],
        )
    path = __file__[:-3] + f"_{4}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"Time / $\mu s$"
    ylabel = r"$\Delta \Delta A$"
    for i in range(0, len(Bs), f):
        plot_general(
            time,
            mary_groundstate[:, i, wl],
            xlabel,
            ylabel,
            style="-",
            label=f"{Bs[i]: .1f} mT",
            colors=colors_field[i],
            factor=1e6,
        )
    path = __file__[:-3] + f"_{5}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"$\Delta I_F$"
    for i in range(0, len(time), n):
        plot_general(
            emission_wavelength,
            mary_emission[i, -1, :],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * factor: .0f} $\\mu s$",
            colors=colors_time[i],
        )
    path = __file__[:-3] + f"_{6}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"$B_0$ / mT"
    ylabel = r"$\Delta I_F$"
    for i in range(0, len(time), n):
        plot_general(
            Bs,
            mary_emission[i, :, wl],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * factor: .0f} $\\mu s$",
            colors=colors_time[i],
        )
    path = __file__[:-3] + f"_{7}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"Time / $\mu s$"
    ylabel = r"$\Delta I_F$"
    for i in range(0, len(Bs), f):
        plot_general(
            time,
            mary_emission[:, i, wl],
            xlabel,
            ylabel,
            style="-",
            label=f"{Bs[i]: .1f} mT",
            colors=colors_field[i],
            factor=1e6,
        )
    path = __file__[:-3] + f"_{8}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            wavelength,
            mary[i, -1, :],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * factor: .0f} $\\mu s$",
            colors=colors_time[i],
        )
    path = __file__[:-3] + f"_{9}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"$B_0$ / mT"
    ylabel = r"$\Delta \Delta A$"
    for i in range(0, len(time), n):
        plot_general(
            Bs,
            mary[i, :, wl],
            xlabel,
            ylabel,
            style="-",
            label=f"{time[i] * factor: .0f} $\\mu s$",
            colors=colors_time[i],
        )
    path = __file__[:-3] + f"_{10}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    xlabel = r"Time / $\mu s$"
    ylabel = r"$\Delta \Delta A$"
    for i in range(0, len(Bs), f):
        plot_general(
            time,
            mary[:, i, wl],
            xlabel,
            ylabel,
            style="-",
            label=f"{Bs[i]: .1f} mT",
            colors=colors_field[i],
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
        ) = Bhalf_fit(Bs, mary_groundstate[i, :, wl])

    plot_bhalf_time(
        time[2:],
        bhalf_time[2:],
        fit_error_time[:, 2:]
        / np.sqrt(
            num_samples,
        ),
    )
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
        ) = Bhalf_fit(Bs, mary_emission[i, :, wl])

    plot_bhalf_time(
        time,
        bhalf_time,
        fit_error_time / np.sqrt(num_samples),
    )
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
        ) = Bhalf_fit(Bs, mary[i, :, wl])

    plot_bhalf_time(
        time[5:],
        bhalf_time[5:],
        fit_error_time[:, 5:] / np.sqrt(num_samples),
    )
    path = __file__[:-3] + f"_{20}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    bhalf_time = np.zeros((len(mary_rp)))
    fit_time = np.zeros((len(Bs), len(mary_rp)))
    fit_error_time = np.zeros((2, len(mary_rp)))
    R2_time = np.zeros((len(mary_rp)))

    for i in range(2, len(mary_rp)):
        (
            bhalf_time[i],
            fit_time[:, i],
            fit_error_time[:, i],
            R2_time[i],
        ) = Bhalf_fit(Bs, mary_rp[i, :, wl])

    plot_bhalf_time(
        time[5:],
        bhalf_time[5:],
        fit_error_time[:, 5:] / np.sqrt(num_samples),
    )
    path = __file__[:-3] + f"_{21}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    bhalf_time = np.zeros((len(mary_trip)))
    fit_time = np.zeros((len(Bs), len(mary_trip)))
    fit_error_time = np.zeros((2, len(mary_trip)))
    R2_time = np.zeros((len(mary_trip)))

    for i in range(2, len(mary_trip)):
        (
            bhalf_time[i],
            fit_time[:, i],
            fit_error_time[:, i],
            R2_time[i],
        ) = Bhalf_fit(Bs, mary_trip[i, :, wl])

    plot_bhalf_time(
        time[5:],
        bhalf_time[5:],
        fit_error_time[:, 5:] / np.sqrt(num_samples),
    )
    path = __file__[:-3] + f"_{22}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3D plots
    xlabel = r"Wavelength / nm"
    ylabel = r"Time / $\mu s$"
    zlabel = r"$\Delta \Delta A$"
    plot_3d_results(
        groundstate_wavelength,
        time,
        mary_groundstate[:, -1, :],
        xlabel,
        ylabel,
        zlabel,
        azim,
        dist,
        elev,
    )
    path = __file__[:-3] + f"_{12}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"$B_0$ / mT"
    zlabel = r"$\Delta \Delta A$"
    plot_3d_results(
        groundstate_wavelength,
        Bs,
        mary_groundstate[250, :, :],
        xlabel,
        ylabel,
        zlabel,
        azim,
        dist,
        elev,
        factor=1,
    )
    path = __file__[:-3] + f"_{13}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"Time / $\mu s$"
    zlabel = r"$\Delta I_F$"
    plot_3d_results(
        emission_wavelength,
        time,
        mary_emission[:, -1, :],
        xlabel,
        ylabel,
        zlabel,
        azim,
        dist,
        elev,
    )
    path = __file__[:-3] + f"_{14}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"$B_0$ / mT"
    zlabel = r"$\Delta I_F$"
    plot_3d_results(
        emission_wavelength,
        Bs,
        mary_emission[250, :, :],
        xlabel,
        ylabel,
        zlabel,
        azim,
        dist,
        elev,
        factor=1,
    )
    path = __file__[:-3] + f"_{15}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"Time / $\mu s$"
    zlabel = r"$\Delta \Delta A$"
    plot_3d_results(
        wavelength,
        time,
        mary[:, -1, :],
        xlabel,
        ylabel,
        zlabel,
        azim,
        dist,
        elev,
    )
    path = __file__[:-3] + f"_{16}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    xlabel = r"Wavelength / nm"
    ylabel = r"$B_0$ / mT"
    zlabel = r"$\Delta \Delta A$"
    plot_3d_results(
        wavelength,
        Bs,
        mary[250, :, :],
        xlabel,
        ylabel,
        zlabel,
        # azim,
        # dist,
        # elev,
        factor=1,
    )
    path = __file__[:-3] + f"_{17}.png"
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
