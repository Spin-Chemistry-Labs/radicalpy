#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.experiments import oop_eseem
from radicalpy.utils import is_fast_run

# OOP-ESEEM simulation for ClCry4a WT and W369F
# Gravell et al., JACS, 2025, 147, 28, 24286–24298


def main(tmin=200e-9, tmax=2e-6, timesteps=800):
    tau = np.linspace(tmin, tmax, timesteps)

    # ClCry4a WT
    J = 0.03e6 * 2 * np.pi
    D = -8.37e6 * 2 * np.pi
    T1 = 0.6e-6

    S1 = oop_eseem(tau, J=J, D=D, T1=T1, n_quad=300)
    # Normalise for display (remove arbitrary proportionality)
    S_plot = S1 / (np.max(np.abs(S1)) + 1e-15)

    # ClCry4a W369F
    J = 0.31e6 * 2 * np.pi
    D = -13.95e6 * 2 * np.pi
    T1 = 0.25e-6

    S2 = oop_eseem(tau, J=J, D=D, T1=T1, n_quad=300)
    # Normalise for display (remove arbitrary proportionality)
    S_plot2 = S2 / (np.max(np.abs(S2)) + 1e-15)

    plt.figure()
    plt.plot(tau * 1e6, S_plot)
    plt.plot(tau * 1e6, S_plot2)
    plt.xlabel(r"$\tau$ / $\mu$s", size=14)
    plt.legend([r"ClCry4a WT", r"ClCry4a W369F"])

    path = __file__[:-3] + f"_{1}.png"
    plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(tmin=200e-9, tmax=2e-6, timesteps=200)
    else:
        main()
