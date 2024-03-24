#! /usr/bin/env python

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():

    path = "./examples/data/fad_kinetics"
    raw_data = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("FADpH21_data.txt")]
    )
    raw_data_time = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("FADpH21_data_time.txt")
        ]
    )
    kinetics_data = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("FAD_pH21_kinetics.txt")
        ]
    )
    kinetics_time = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("FAD_pH21_kinetics_time.txt")
        ]
    )
    semiclassical_data = np.array(
        [np.genfromtxt(file_path) for file_path in Path(path).glob("semiclassical.txt")]
    )
    semiclassical_time = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("semiclassical_time.txt")
        ]
    )
    semiclassical_kinetics_data = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("semiclassical_kinetics.txt")
        ]
    )
    semiclassical_kinetics_time = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("semiclassical_kinetics_time.txt")
        ]
    )

    plt.figure(1)
    plt.plot(
        raw_data_time[0, :] - raw_data_time[0, 0],
        raw_data[0, :] / raw_data[0, :].max(),
        "k",
        linewidth=3,
    )
    plt.plot(
        kinetics_time[0, :] * 1e6,
        kinetics_data[0, :] / kinetics_data[0, :].max(),
        "b",
        linewidth=3,
    )
    plt.plot(
        semiclassical_time[0, :],
        semiclassical_data[0, :] / semiclassical_data[0, :].max(),
        "g",
        linewidth=3,
    )
    plt.plot(
        semiclassical_kinetics_time[0, :] * 1e6,
        semiclassical_kinetics_data[0, :] / semiclassical_kinetics_data[0, :].max(),
        "r",
        linewidth=3,
    )
    plt.xlabel("Time / $\mu s$", size=18)
    plt.ylabel("Normalised $\Delta \Delta A$ / a.u.", size=18)
    plt.ylim([-0.1, 1.1])
    plt.legend(
        ["Data", "Kinetic Model", "Semiclassical", "Semiclassical + Kinetic Model"]
    )
    plt.tick_params(labelsize=14)
    plt.gcf().set_size_inches(10, 5)
    plt.show()


if __name__ == "__main__":
    main()
