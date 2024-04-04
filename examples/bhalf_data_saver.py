#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from radicalpy.plot import plot_3d_results, plot_bhalf_time, plot_general
from radicalpy.utils import Bhalf_fit


def main():

    # Parameters
    scale_factor = 2.5e-2  # 4e-1

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

    result = np.load(
        "./examples/data/fad_06_kd7e6/results.npy", allow_pickle=True
    ).item()

    time = result["ts"]
    Bs = result["Bs"]
    results = result["yield"]

    total_yield = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    zero_field = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)
    mary = np.zeros((len(time), len(Bs), len(wavelength)), dtype=complex)

    radical_pair_yield = (
        results[:, 5, :] + results[:, 10, :] + results[:, 15, :] + results[:, 20, :]
    )
    triplet_yield = results[:, 2, :] + results[:, 3, :] + results[:, 4, :]

    for i, r in enumerate(radical_spectrum):
        for j, t in enumerate(triplet_spectrum):
            total_yield[:, :, j + 1] = (
                (r * radical_pair_yield) + (t * (2 * triplet_yield))
            ) * scale_factor

    for i in range(0, len(wavelength)):
        for j in range(0, len(Bs)):
            zero_field[:, j, i] = total_yield[:, 0, i]

    mary = np.real(total_yield - zero_field)

    wl = 17

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

    np.savetxt("./examples/data/fad_06_kd7e6/time.txt", time)
    np.savetxt("./examples/data/fad_06_kd7e6/bhalf.txt", bhalf_time)
    np.savetxt("./examples/data/fad_06_kd7e6/bhalf_error.txt", fit_error_time)


if __name__ == "__main__":
    main()
