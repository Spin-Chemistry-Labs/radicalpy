#! /usr/bin/env python


import matplotlib.pylab as pylab
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State

axis_labels = [
    r"T+$\alpha$",
    r"T+$\beta$",
    r"T0$\alpha$",
    r"T0$\beta$",
    r"S$\alpha$",
    r"S$\beta$",
    r"T-$\alpha$",
    r"T-$\beta$",
]

ticksx = np.arange(0.5, len(axis_labels), 1)
ticksy = np.arange(0.5, len(axis_labels), 1)


def details():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])
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
        kinetics=[rp.kinetics.Exponential(k)],
    )

    params = {
        "figure.figsize": [10, 10],
        # "axes3d.grid": False,
        # "axes.facecolor": "none",
        # "axes.labelsize": 12,
        # "axes.spines.bottom": True,
        # "axes.spines.left": True,
        # "axes.spines.right": True,
        # "axes.spines.top": True,
        # "xtick.labelsize": 12,
        # "ytick.labelsize": 12,
    }
    pylab.rcParams.update(params)
    bar3d_kwargs = {
        # "alpha": 0.9,
    }
    axes_kwargs = {
        "xlabel": "Spin state",
        "ylabel": "Spin state",
        "zlabel": "Probability",
        "xticks": ticksx,
        "xticklabels": axis_labels,
        "yticks": ticksy,
        "yticklabels": axis_labels,
    }
    for Bi, B in enumerate(Bs):
        axes_kwargs["title"] = f"$B = {B} mT$"
        anim = rp.plot.density_matrix_animation(
            MARY["rhos"],
            Bi=Bi,
            frames=30,
            bar3d_kwargs=bar3d_kwargs,
            axes_kwargs=axes_kwargs,
        )
        path = __file__[:-3] + f"_{B}.gif"
        anim.save(path, fps=8)


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])
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
        kinetics=[rp.kinetics.Exponential(k)],
    )

    params = {"figure.figsize": [10, 10]}
    pylab.rcParams.update(params)
    bar3d_kwargs = {}

    # axes_kwargs = rp.plot.density_matrix_axis_kwargs()
    axes_kwargs = {
        "xticks": ticksx,
        "xticklabels": axis_labels,
        "yticks": ticksy,
        "yticklabels": axis_labels,
    }
    axes_kwargs["xlabel"] = "Spin state"
    axes_kwargs["ylabel"] = "Spin state"
    axes_kwargs["ylabel"] = "Spin state"

    for Bi, B in enumerate(Bs):
        axes_kwargs["title"] = f"$B = {B} mT$"
        anim = rp.plot.density_matrix_animation(
            MARY["rhos"],
            Bi=Bi,
            frames=30,
            bar3d_kwargs=bar3d_kwargs,
            axes_kwargs=axes_kwargs,
        )
        path = __file__[:-3] + f"_{B}.gif"
        anim.save(path, fps=8)


if __name__ == "__main__":
    main()
