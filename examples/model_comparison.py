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

    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="black")
    plt.plot(
        raw_data_time[0, :] - raw_data_time[0, 0],
        raw_data[0, :] / raw_data[0, :].max(),
    )
    plt.plot(
        kinetics_time[0, :] * 1e6,
        kinetics_data[0, :] / kinetics_data[0, :].max(),
    )
    plt.plot(
        semiclassical_time[0, :],
        semiclassical_data[0, :] / semiclassical_data[0, :].max(),
    )
    plt.plot(
        semiclassical_kinetics_time[0, :] * 1e6,
        semiclassical_kinetics_data[0, :] / semiclassical_kinetics_data[0, :].max(),
    )
    ax.set_xlabel("Time / $\mu s$", size=24)
    ax.set_ylabel("Normalised $\Delta \Delta A / a.u.$", size=24)
    plt.ylim([-0.1, 1.1])
    # plt.xlim([0, 6])
    plt.legend(
        ["Data", "Kinetic Model", "Semiclassical", "Semiclassical + Kinetic Model"]
    )
    plt.tick_params(labelsize=14)
    plt.gcf().set_size_inches(10, 5)
    plt.show()


# figure(1)
# hold on
# plot(FADpH21_data(:, 1) - FADpH21_data(1, 1), FADpH21_data(:, 2) / max(FADpH21_data(:, 2)))
# plot(FAD_pH21_kinetics_time *1e6, FAD_pH21_kinetics / max(FAD_pH21_kinetics))
# plot(semiclassical_kinetics_time, semiclassical_kinetics / max(semiclassical_kinetics))
# plot(time, real(result3(:,1)) / max(real(result3(:,1))))
# xlim([0, 6]); ylim([-0.1, 1]);
# title("FAD pH 2.1")
# xlabel("Time / \mus"); ylabel("Normalised \Delta\DeltaA / a.u.")
# legend("Data", "Kinetic Model", "Semiclassical", "Semiclassical + Kinetic model")


if __name__ == "__main__":
    main()
