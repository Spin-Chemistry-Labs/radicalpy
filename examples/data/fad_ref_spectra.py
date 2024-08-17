#! /usr/bin/env python

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():

    path = Path("./examples/data/fad_kinetics")
    radical_spectrum = np.genfromtxt(path / "fad_radical_spectrum.txt")
    radical_wavelength = np.genfromtxt(path / "fad_radical_wavelength.txt")
    triplet_spectrum = np.genfromtxt(path / "fad_triplet_spectrum.txt")
    triplet_wavelength = np.genfromtxt(path / "fad_triplet_wavelength.txt")

    plt.figure(1)
    plt.plot(
        radical_wavelength,
        radical_spectrum * 1e3,
        "ro",
        linewidth=3,
    )
    plt.plot(
        triplet_wavelength,
        triplet_spectrum * 1e3,
        "bo",
        linewidth=3,
    )
    plt.xlabel(r"Wavelength / nm", size=24)
    plt.ylabel(r"$\epsilon$ / $M^{-1} cm^{-1}$", size=24)
    plt.legend(["Radical", "Triplet"], fontsize=16)
    plt.tick_params(labelsize=18)
    plt.gcf().set_size_inches(10, 5)
    # plt.show()

    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
