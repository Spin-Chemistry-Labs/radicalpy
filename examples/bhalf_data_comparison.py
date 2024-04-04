#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():

    # Load B1/2 spectra, kd = 3e5
    path = "./examples/data/fad_03_kd3e5"
    time_3e5 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("time.txt")]
    )
    bhalf_3e5 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("bhalf.txt")]
    )
    error_3e5 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("bhalf_error.txt")]
    )

    # Load B1/2 spectra, kd = 3e6
    path = "./examples/data/fad_04_kd3e6"
    time_3e6 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("time.txt")]
    )
    bhalf_3e6 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("bhalf.txt")]
    )
    error_3e6 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("bhalf_error.txt")]
    )

    # Load B1/2 spectra, kd = 7e6
    path = "./examples/data/fad_06_kd7e6"
    time_7e6 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("time.txt")]
    )
    bhalf_7e6 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("bhalf.txt")]
    )
    error_7e6 = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("bhalf_error.txt")]
    )

    time_3e5 = time_3e5[0, :]
    time_3e6 = time_3e6[0, :]
    time_7e6 = time_7e6[0, :]

    bhalf_3e5 = bhalf_3e5[0, :] / bhalf_3e5[0, :].max()
    bhalf_3e6 = bhalf_3e6[0, :] / bhalf_3e6[0, :].max()
    bhalf_7e6 = bhalf_7e6[0, :] / bhalf_7e6[0, :].max()

    num_samples = 200

    error_3e5 = error_3e5[0, :] / np.sqrt(num_samples)
    error_3e6 = error_3e6[0, :] / np.sqrt(num_samples)
    error_7e6 = error_7e6[0, :] / np.sqrt(num_samples)

    factor = 1e6
    cutoff = 5

    plt.figure(1)
    for i in range(2, len(time_3e5), 50):
        plt.plot(time_3e5[cutoff:i] * factor, bhalf_3e5[cutoff:i], "r", linewidth=3)
        # plt.errorbar(
        #     time_3e5[cutoff:i] * factor,
        #     bhalf_3e5[cutoff:i],
        #     error_3e5[1, cutoff:i],
        #     color="r",
        #     linewidth=2,
        # )
        plt.plot(time_3e6[cutoff:i] * factor, bhalf_3e6[cutoff:i], "b", linewidth=3)
        # plt.errorbar(
        #     time_3e6[cutoff:i] * factor,
        #     bhalf_3e6[cutoff:i],
        #     error_3e6[1, i],
        #     color="b",
        #     linewidth=2,
        # )
        plt.plot(time_7e6[cutoff:i] * factor, bhalf_7e6[cutoff:i], "g", linewidth=3)
        # plt.errorbar(
        #     time_7e6[cutoff:i] * factor,
        #     bhalf_7e6[cutoff:i],
        #     error_7e6[1, i],
        #     color="g",
        #     linewidth=2,
        # )
    plt.xlabel("Time / $\mu s$", size=24)
    plt.ylabel("Normalised $B_{1/2}$ / a.u.", size=24)
    plt.legend(
        [
            r"$k_d = 3 \times 10^5 s^{-1}$",
            r"$k_d = 3 \times 10^6 s^{-1}$",
            r"$k_d = 7 \times 10^6 s^{-1}$",
        ],
        fontsize=16,
    )
    plt.tick_params(labelsize=18)
    plt.gcf().set_size_inches(10, 5)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
