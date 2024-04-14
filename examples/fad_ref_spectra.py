#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():

    path = "./examples/data/fad_kinetics"
    radical_spectrum = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_radical_spectrum.txt")
        ]
    )
    radical_wavelength = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_radical_wavelength.txt")
        ]
    )
    triplet_spectrum = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_triplet_spectrum.txt")
        ]
    )
    triplet_wavelength = np.array(
        [
            np.genfromtxt(file_path)
            for file_path in Path(path).glob("fad_triplet_wavelength.txt")
        ]
    )

    plt.figure(1)
    plt.plot(
        radical_wavelength[0, :],
        radical_spectrum[0, :] * 1e3,
        "ro",
        linewidth=3,
    )
    plt.plot(
        triplet_wavelength[0, :],
        triplet_spectrum[0, :] * 1e3,
        "bo",
        linewidth=3,
    )
    plt.xlabel("Wavelength / nm", size=24)
    plt.ylabel("$\epsilon$ / $M^{-1} cm^{-1}$", size=24)
    plt.legend(["Radical", "Triplet"], fontsize=16)
    plt.tick_params(labelsize=18)
    plt.gcf().set_size_inches(10, 5)
    # plt.show()

    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
