#! /usr/bin/env python

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .simulation import HilbertSimulation, State


def _format_label(t):
    return f"$\\vert {t} \\rangle$"


def spin_state_labels(sim: HilbertSimulation):
    if sim.num_electrons != 2:
        raise ValueError(
            "Density matrix plotting make little sense for non-radical pairs!"
        )
    multiplicities = sim.multiplicities[sim.num_electrons :]
    old_labels = [
        State.TRIPLET_PLUS.value,
        State.TRIPLET_ZERO.value,
        State.SINGLET.value,
        State.TRIPLET_MINUS.value,
    ]
    for m in multiplicities:
        labels = []
        for label in old_labels:
            for t in range(m):
                tt = int(2 * ((m - t - 1) - (m - 1) / 2))
                tt = f"{tt}/2" if tt % 2 else str(tt // 2)
                if tt[0] not in {"-", "0"}:
                    tt = f"+{tt}"
                labels.append(f"{label}, {tt}")

        old_labels = labels
    return list(map(_format_label, labels))


def density_matrix_animation(rhos, frames, bar3d_kwargs, axes_kwargs):
    fig = plt.figure()
    ax = plt.axes(projection="3d", aspect="auto")

    def anim_func(t):
        Z = np.abs(rhos[t])
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


def anisotropy_surface(theta, phi, Y):
    PH, TH = np.meshgrid(phi, theta)
    xyz = np.array([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)])

    Yx, Yy, Yz = Y.real * xyz

    # Colour the plotted surface according to the sign of Y
    # cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("Accent_r"))
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    cmap.set_clim(-0.01, 0.01)

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor("none")
    ax.plot_surface(Yx, Yy, Yz, facecolors=cmap.to_rgba(Y.real), rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference
    ax_lim = np.max(Y.real)
    ax.plot([-ax_lim, ax_lim], [0, 0], [0, 0], c="0.5", lw=1, zorder=10)
    ax.plot([0, 0], [-ax_lim, ax_lim], [0, 0], c="0.5", lw=1, zorder=10)
    ax.plot([0, 0], [0, 0], [-ax_lim, ax_lim], c="0.5", lw=1, zorder=10)
    # Set the Axes limits alpha and title, turn off the Axes frame.
    ax_lim = np.max(Y.real)
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis("off")
    fig.set_size_inches(20, 10)
    # plt.savefig('Y.png')
