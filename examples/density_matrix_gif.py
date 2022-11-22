#! /usr/bin/env python

import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.LiouvilleSimulation([flavin, Z])
    time = np.arange(0, 15e-6, 5e-9)
    Bs = np.arange(0, 3, 1)
    k = 1e6

    MARY = sim.MARY(
        init_state=State.TRIPLET,
        obs_state=State.TRIPLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        # kinetics=[rp.kinetics.Exponential(k)],
        kinetics=[
            rp.kinetics.Haberkorn(k, State.SINGLET),
            rp.kinetics.Haberkorn(k, State.TRIPLET),
        ],
    )

    bar3d_kwargs = {"alpha": 0.9}
    axes_kwargs = rp.plot.density_matrix_axes_kwargs(sim)
    # axes_kwargs["xlabel"] = "Spin state"
    # axes_kwargs["ylabel"] = "Spin state"
    axes_kwargs["zlabel"] = "Probability"
    rhos = MARY["rhos"]
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
