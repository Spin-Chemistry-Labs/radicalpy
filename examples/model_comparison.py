#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import zoom


def mse(x, y):
    n = len(x)
    assert n == len(y)
    return 1 / n * sum((x - y) ** 2)


def main():

    path = Path("./examples/data/fad_kinetics")
    start = 19
    raw_data = np.genfromtxt(path / "FADpH21_data.txt")[start:]
    raw_data_time = np.genfromtxt(path / "FADpH21_data_time.txt")[start:]
    kinetics_data = np.genfromtxt(path / "FAD_pH21_kinetics.txt")
    kinetics_time = np.genfromtxt(path / "FAD_pH21_kinetics_time.txt")
    semiclassical_data = np.genfromtxt(path / "semiclassical.txt")
    semiclassical_time = np.genfromtxt(path / "semiclassical_time.txt")
    semiclassical_kinetics_data = np.genfromtxt(path / "semiclassical_kinetics.txt")
    semiclassical_kinetics_time = np.genfromtxt(
        path / "semiclassical_kinetics_time.txt"
    )
    print(f"{len(kinetics_data)=}")  # 500
    print(f"{len(semiclassical_data)=}")  # 600
    print(f"{len(semiclassical_kinetics_data)=}")  # 600

    raw_size = len(raw_data)
    raw_kinetics_zoom = zoom(raw_data, len(kinetics_data) / raw_size)
    raw_semiclassical_zoom = zoom(raw_data, len(semiclassical_data) / raw_size)
    raw_semiclassical_kinetics_zoom = zoom(
        raw_data, len(semiclassical_kinetics_data) / raw_size
    )
    print(f"{mse(kinetics_data, raw_kinetics_zoom)=}")
    print(f"{mse(semiclassical_data, raw_semiclassical_zoom)=}")
    print(f"{mse(semiclassical_kinetics_data, raw_semiclassical_kinetics_zoom)=}")

    plt.figure(1)
    plt.plot(
        raw_data_time - raw_data_time[0],
        raw_data / raw_data.max(),
        "k",
        linewidth=3,
    )
    plt.plot(
        kinetics_time * 1e6,
        kinetics_data / kinetics_data.max(),
        "b",
        linewidth=3,
    )
    plt.plot(
        semiclassical_time,
        semiclassical_data / semiclassical_data.max(),
        "g",
        linewidth=3,
    )
    plt.plot(
        semiclassical_kinetics_time * 1e6,
        semiclassical_kinetics_data / semiclassical_kinetics_data.max(),
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


if __name__ == "__main__":
    main()
