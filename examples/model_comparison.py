#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import zoom


def mse(raw, raw_time, data):
    N = len(data) - 1
    indices = N * raw_time / raw_time[-1]
    return sum((data[indices.astype(int)] - raw) ** 2) / len(raw)


def main(start=19):

    path = Path("./examples/data/fad_kinetics")
    raw_data = np.genfromtxt(path / "FADpH21_data.txt")[start:]
    raw_data /= raw_data.max()
    raw_data_time = np.genfromtxt(path / "FADpH21_data_time.txt")[start:]
    raw_data_time -= raw_data_time[0]
    kinetics_data = np.genfromtxt(path / "FAD_pH21_kinetics.txt")
    kinetics_data /= kinetics_data.max()
    kinetics_time = np.genfromtxt(path / "FAD_pH21_kinetics_time.txt")
    semiclassical_data = np.genfromtxt(path / "semiclassical.txt")
    semiclassical_data /= semiclassical_data.max()
    semiclassical_time = np.genfromtxt(path / "semiclassical_time.txt")
    semiclassical_kinetics_data = np.genfromtxt(path / "semiclassical_kinetics.txt")
    semiclassical_kinetics_data /= semiclassical_kinetics_data.max()
    semiclassical_kinetics_time = np.genfromtxt(
        path / "semiclassical_kinetics_time.txt"
    )
    mse_semiclassical = mse(raw_data, raw_data_time, semiclassical_data)
    mse_kinetics = mse(raw_data, raw_data_time, kinetics_data)
    mse_new_method = mse(raw_data, raw_data_time, semiclassical_kinetics_data)
    print(f"{mse_semiclassical=}")
    print(f"{mse_kinetics=}")
    print(f"{mse_new_method=}")

    plt.figure(1)
    # plt.plot(raw_data_time)
    plt.plot(raw_data_time, raw_data, "k", linewidth=3)
    plt.plot(
        kinetics_time * 1e6,
        kinetics_data,
        "b",
        linewidth=3,
    )
    plt.plot(
        semiclassical_time,
        semiclassical_data,
        "g",
        linewidth=3,
    )
    plt.plot(
        semiclassical_kinetics_time * 1e6,
        semiclassical_kinetics_data,
        "r",
        linewidth=3,
    )
    plt.xlabel("Time / $\mu s$", size=24)
    plt.ylabel("Normalised $\Delta \Delta A$ / a.u.", size=24)
    plt.ylim([-0.1, 1.1])
    # plt.xlim([-0.1, 6])
    plt.legend(
        ["Data", "Kinetic Model", "Semiclassical", "Semiclassical + Kinetic Model"],
        fontsize=16,
    )
    plt.tick_params(labelsize=18)
    plt.gcf().set_size_inches(10, 5)
    # plt.show()

    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    return mse_semiclassical, mse_kinetics, mse_new_method


if __name__ == "__main__":
    results = []
    for start in [19]:
        # for start in range(0, 30):
        print(f"raw data cut off: {start=}")
        results.append(main(start))
    results = np.array(results)
    print(f"{results.min(axis=0)=}")
