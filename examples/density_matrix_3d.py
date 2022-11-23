#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    # sim = rp.simulation.LiouvilleSimulation([flavin, Z])
    sim = rp.simulation.HilbertSimulation([flavin, Z])
    time = np.arange(0, 15e-6, 5e-9)
    Bs = np.arange(0, 3, 1)
    k = 1e6

    MARY = sim.MARY(
        init_state=State.SINGLET,
        obs_state=State.TRIPLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        # kinetics=[rp.kinetics.Exponential(k)],
        # kinetics=[
        #    rp.kinetics.Haberkorn(k, State.SINGLET),
        #    rp.kinetics.HaberkornFree(k),
        # ],
    )
    rhos = MARY["rhos"]

    axis_labels = rp.plot.spin_state_labels(sim)
    axes_kwargs = {
        "xticks": np.arange(0, len(axis_labels)),
        "xticklabels": axis_labels,
        "yticks": np.arange(0, len(axis_labels)) + 1,
        "yticklabels": axis_labels,
        # "xlabel": "Spin state",
        # "ylabel": "Spin state",
        "zlabel": "Probability",
    }
    bar3d_kwargs = {"alpha": 0.9}
    for Bi, B in enumerate(Bs):
        axes_kwargs["title"] = f"$B = {B} mT$"
        anim = rp.plot.density_matrix_animation(
            rhos,
            Bi=Bi,
            frames=30,
            bar3d_kwargs=bar3d_kwargs,
            axes_kwargs=axes_kwargs,
        )
        path = __file__[:-3] + f"_{B}.gif"
        anim.save(path, fps=8)


if __name__ == "__main__":
    main()
