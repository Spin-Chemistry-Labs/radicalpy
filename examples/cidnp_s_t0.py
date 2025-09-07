#! /usr/bin/env python

# Simulation of CIDNP

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.experiments import cidnp
from radicalpy.utils import is_fast_run


def main(Bmax=20.0, dB=100):
    deltag = 2.0041 - 2.0036
    Bs = np.linspace(0.0, Bmax, dB)
    ks = 5e8  # s^-1
    alpha = 1.5
    model = "c"
    donor_hfcs_spinhalf = [0.15, -0.65, -0.65, 0.15, 0.77]
    acceptor_hfcs_spinhalf = [0.390, -0.769, -0.17, 0.24]
    donor_hfcs_spin1 = []
    acceptor_hfcs_spin1 = []

    for i in range(1, 6):
        B0, p = cidnp(
            B0=Bs,
            deltag=deltag,
            cidnp_model=model,
            ks=ks,
            alpha=alpha,
            nucleus_of_interest=i,
            donor_hfc_spinhalf=donor_hfcs_spinhalf,
            acceptor_hfc_spinhalf=acceptor_hfcs_spinhalf,
            donor_hfc_spin1=donor_hfcs_spin1,
            acceptor_hfc_spin1=acceptor_hfcs_spin1,
        )
        plt.plot(B0, p)

    plt.xlabel("$B_0$ /T")
    plt.ylabel("CIDNP Intensity")
    plt.show()

    # path = __file__[:-3] + f"_{0}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(Bmax=20.0, dB=10)
    else:
        main()
