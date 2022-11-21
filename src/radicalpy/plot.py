#! /usr/bin/env python

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def density_matrix_animation(rhos, Bi, frames, bar3d_kwargs, axes_kwargs):
    fig = plt.figure()
    ax = plt.axes(projection="3d", aspect="auto")

    def anim_func(t):
        Z = np.abs(rhos[Bi, t])
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
            **bar3d_kwargs,
        )
        return frame

    return FuncAnimation(fig, anim_func, frames=frames)
