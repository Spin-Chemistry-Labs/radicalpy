#! /usr/bin/env python

# Simulation of CIDNP

import matplotlib.pyplot as plt
import numpy as np

from radicalpy.experiments import nmr
from radicalpy.utils import is_fast_run

# 500 MHz NMR example for tyrosine

def main(Bmax=20.0, dB=100):
    MULTIPLETS = [
    (2.0,  800.0, 2, 0),    # 2 doublets @ 800 Hz
    (2.0,  900.0, 2, 0),    # 2 doublets @ 900 Hz
    (1.0,  2870.0, 2, 20),  # doublet @ 2870 Hz, J=20 Hz
    (1.0,  2770.0, 2, 20),  # doublet @ 2770 Hz, J=20 Hz
    ]
    SW_HZ        = 4000.0   # spectral width (Hz)
    NP           = 8000     # acquired points
    N_FFT        = 16000    # zero-filled FFT length
    TRANS_MHZ    = 500.0    # transmitter frequency (MHz)
    CARRIER_PPM  = 4.7      # carrier position (ppm)
    LINEWIDTH_HZ = 2.0      # Lorentzian linewidth (Hz)
    SCALE        = 1.0      # final y-scale

    ppm, spectrum = nmr(multiplets=MULTIPLETS,
        spectral_width=SW_HZ,
        number_of_points=NP,
        fft_number=N_FFT,
        transmitter_frequency=TRANS_MHZ,
        carrier_position=CARRIER_PPM,
        linewidth=LINEWIDTH_HZ,
        scale=SCALE)

    plt.plot(ppm, spectrum.real)
    plt.gca().invert_xaxis()
    plt.xlabel("ppm", size=8)
    plt.tick_params(labelsize=8)
    plt.gcf().set_size_inches(5, 3)
    plt.show()

    # path = __file__[:-3] + f"_{0}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main(Bmax=20.0, dB=10)
    else:
        main()
