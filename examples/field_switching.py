#! /usr/bin/env python

import matplotlib.pyplot as plt

from radicalpy.data import Molecule
from radicalpy.experiments import field_switching
from radicalpy.simulation import HilbertSimulation, State
from radicalpy.utils import is_fast_run


def main():
    m1 = Molecule.fromdb("flavin_anion", ["H25"])
    m2 = Molecule.fromdb("tryptophan_cation", [])
    sim = HilbertSimulation([m1, m2])

    res = field_switching(
        sim,
        B_on=2.0,
        B_off=0.0,
        init_state=State.TRIPLET,
        dt=1e-9,
        n_offsets=100,
        offset_step=10,
        pulse_width_steps=800,
        k_rec=50e6,
        k_esc=2e6,
    )

    switch_ns = res["switch_times"] * 1e9
    SEMF = res["SEMF"]

    plt.figure()
    plt.plot(switch_ns, SEMF)
    plt.xlabel("Switching time / ns")
    plt.ylabel("$\Delta \Delta A$")
    plt.show()
    # path = __file__[:-3] + f"_{1}.png"
    # plt.savefig(path)


if __name__ == "__main__":
    if is_fast_run():
        main()
    else:
        main()
