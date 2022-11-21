#! /usr/bin/env python

import os

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from matplotlib.animation import FuncAnimation
from radicalpy.simulation import State

path = __file__[:-2] + "gif"


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


def density_matrix_animation(rhos, B0, frames, bar3d_kwargs, axes_kwargs):
    fig = plt.figure()
    ax = plt.axes(projection="3d", aspect="auto")

    def anim_func(i):
        Z = np.abs(rhos[B0, i])
        X, Y = np.meshgrid(range(len(Z)), range(len(Z)))
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

        fracs = Z.astype(float) / Z.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        color_values = cm.jet(norm(fracs.tolist()))

        ax.cla()
        # ax.axis("off")

        ax.set(**axes_kwargs)
        frame = ax.bar3d(
            X,
            Y,
            np.zeros_like(X),
            np.ones_like(X),
            np.ones_like(Y),
            Z,
            color=color_values,
            **bar3d_kwargs
        )
        return frame

    return FuncAnimation(fig, anim_func, frames=frames)


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])
    time = np.arange(0, 15e-6, 5e-9)
    Bs = np.arange(0, 10, 3)
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

    keys = pylab.rcParams.keys()
    # print(keys)
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
    anim = density_matrix_animation(
        MARY["rhos"],
        B0=0,
        frames=100,
        bar3d_kwargs=bar3d_kwargs,
        axes_kwargs=axes_kwargs,
    )
    anim.save(path, fps=8)


if __name__ == "__main__":
    main()
