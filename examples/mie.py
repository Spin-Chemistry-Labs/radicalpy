#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import radicalpy as rp
from radicalpy.simulation import State


def main():
    Py_h = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.073])
    DMA_h = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[0.181])

    Py_d = rp.simulation.Molecule.fromisotopes(isotopes=["2H"], hfcs=[0.481])
    DMA_d = rp.simulation.Molecule.fromisotopes(isotopes=["2H"], hfcs=[1.18])

    sim = rp.simulation.HilbertSimulation([Py_h, DMA_h])
    sim2 = rp.simulation.HilbertSimulation([Py_d, DMA_d])

    k = 3e6
    time = np.arange(0, 5e-6, 5e-9)
    Bs = np.arange(0, 30, 0.1)

    results = sim.MARY(
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        kinetics=[
            rp.kinetics.Exponential(k),
        ],
    )
    MARY = results["MARY"]
    HFE = results["HFE"]
    LFE = results["LFE"]

    results2 = sim2.MARY(
        init_state=State.SINGLET,
        obs_state=State.SINGLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        kinetics=[
            rp.kinetics.Exponential(k),
        ],
    )
    MARY2 = results2["MARY"]
    HFE2 = results2["HFE"]
    LFE2 = results2["LFE"]

    plt.plot(Bs, MARY, color="red", linewidth=2)
    plt.plot(Bs, MARY2, color="blue", linewidth=2)

    plt.xlabel("$B_0 (mT)$")
    plt.ylabel("MFE (%)")

    print(f"HFE = {HFE: .2f} %")
    print(f"LFE = {LFE: .2f} %")

    print(f"HFE = {HFE2: .2f} %")
    print(f"LFE = {LFE2: .2f} %")

    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)


if __name__ == "__main__":
    main()
