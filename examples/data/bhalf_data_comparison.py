#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():

    # Load B1/2 spectra, kd = 3e5
    path = Path("./examples/data/bhalf_analysis/fad_kd3e5")
    time_3e5 = np.genfromtxt(path / "time.txt")

    bhalf_3e5 = np.genfromtxt(path / "bhalf.txt")
    error_3e5 = np.genfromtxt(path / "bhalf_error.txt")

    # Load B1/2 spectra, kd = 3e6
    path = Path("./examples/data/bhalf_analysis/fad_kd3e6")
    time_3e6 = np.genfromtxt(path / "time.txt")
    bhalf_3e6 = np.genfromtxt(path / "bhalf.txt")
    error_3e6 = np.genfromtxt(path / "bhalf_error.txt")

    # Load B1/2 spectra, kd = 7e6
    path = Path("./examples/data/bhalf_analysis/fad_kd7e6")
    time_7e6 = np.genfromtxt(path / "time.txt")
    bhalf_7e6 = np.genfromtxt(path / "bhalf.txt")
    error_7e6 = np.genfromtxt(path / "bhalf_error.txt")

    bhalf_3e5 = bhalf_3e5 / bhalf_3e5.max()
    bhalf_3e6 = bhalf_3e6 / bhalf_3e6.max()
    bhalf_7e6 = bhalf_7e6 / bhalf_7e6.max()

    num_samples = 200

    error_3e5 = error_3e5 / np.sqrt(num_samples)
    error_3e6 = error_3e6 / np.sqrt(num_samples)
    error_7e6 = error_7e6 / np.sqrt(num_samples)

    factor = 1e6
    cutoff = 5

    plt.figure(1)
    for i in range(2, len(time_3e5), 50):
        plt.plot(time_3e5[cutoff:i] * factor, bhalf_3e5[cutoff:i], "r", linewidth=3)
        plt.plot(time_3e6[cutoff:i] * factor, bhalf_3e6[cutoff:i], "b", linewidth=3)
        plt.plot(time_7e6[cutoff:i] * factor, bhalf_7e6[cutoff:i], "g", linewidth=3)
    plt.xlabel(r"Time / $\mu s$", size=24)
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
