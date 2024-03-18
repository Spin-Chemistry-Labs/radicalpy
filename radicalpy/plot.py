#! /usr/bin/env python

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .simulation import HilbertSimulation, State
from .utils import spherical_to_cartesian


def anisotropy_surface(theta, phi, Y):
    # TODO(vatai): clean up
    PH, TH = np.meshgrid(phi, theta)
    xyz = np.array([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)])

    Yx, Yy, Yz = Y.real * spherical_to_cartesian(TH, PH)

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


def linear_energy_levels(H, B, linecolour, title):
    # todo(vatai): clean up
    eigval = np.linalg.eigh(H)  # try eig(H)
    E = np.real(eigval[0])  # 0 = eigenvalues, 1 = eigenvectors

    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.eventplot(E, orientation="vertical", color=linecolour, linewidth=3)
    ax.set_title(title, size=18)
    ax.set_ylabel("Spin state energy (J)", size=14)
    plt.tick_params(labelsize=14)


def energy_levels(sim: HilbertSimulation, B: np.ndarray, J=0, D=0):
    # TODO(VATAI): DO THIS PROPERLY
    # TODO(VATAI): use tick labels
    assert (
        type(sim) == HilbertSimulation
    ), "plot.energy_levels assumes Hilbert space simulation"
    H_base = sim.total_hamiltonian(0, J, D)
    H_zee = sim.zeeman_hamiltonian(1)

    E = np.zeros([len(B), len(H_base)], dtype=np.complex_)

    for i, B0 in enumerate(B):
        H = H_base + B0 * H_zee
        eigval = np.linalg.eig(H)
        E[i] = eigval[0]  # 0 = eigenvalues, 1 = eigenvectors

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(B, np.real(E[:, ::-1]), linewidth=2)
    # ax.set_title(title, size=18)
    ax.set_xlabel("$B_0 (T)$", size=14)
    ax.set_ylabel("Spin state energy (J)", size=14)
    plt.tick_params(labelsize=14)


def monte_carlo_free(pos):
    f = 1e9
    pos *= f

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "auto"})
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    ax.plot(*pos.T, alpha=0.9, color="cyan")
    ax.plot(*pos[0], "bo", markersize=15)
    ax.plot(0, 0, 0, "mo", markersize=15)
    ax.set_title(
        "3D Monte Carlo random walk simulation for a radical pair in water", size=16
    )
    ax.set_xlabel("$X$ (nm)", size=14)
    ax.set_ylabel("$Y$ (nm)", size=14)
    ax.set_zlabel("$Z$ (nm)", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)


def monte_carlo_caged(pos, r_max):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x_frame = r_max * np.outer(np.sin(theta), np.cos(phi))
    y_frame = r_max * np.outer(np.sin(theta), np.sin(phi))
    z_frame = r_max * np.outer(np.cos(theta), np.ones_like(phi))

    f = 1e9

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d", "aspect": "auto"})
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    ax.plot_wireframe(
        x_frame * f,
        y_frame * f,
        z_frame * f,
        color="k",
        alpha=0.1,
        rstride=1,
        cstride=1,
    )
    pos = f * pos
    ax.plot(*pos.T, alpha=0.9, color="cyan")
    ax.plot(*pos[0], "bo", markersize=15)
    ax.plot(0, 0, 0, "ro", markersize=15)
    #     ax.set_title("3D Monte Carlo random walk simulation for an encapsulated radical pair", size=16)
    ax.set_xlabel("$X$ (nm)", size=14)
    ax.set_ylabel("$Y$ (nm)", size=14)
    ax.set_zlabel("$Z$ (nm)", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)


def plot_3d_results(results, factor=1e6):
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    ax.set_facecolor("none")
    ax.grid(False)
    X, Y = np.meshgrid(results["Bs"], results["ts"])
    ax.plot_surface(
        X,
        Y * factor,
        results["MARY"],
        facecolors=cmap.to_rgba(results["MARY"].real),
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel("$B_0$ (mT)", size=18)
    ax.set_ylabel("Time ($\mu s$)", size=18)
    ax.set_zlabel("$\Delta \Delta A$", size=18)
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 5)
    plt.show()


def plot_autocorrelation_fit(t_j, acf_j, acf_j_fit, zero_point_crossing_j):
    fig = plt.figure(2)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.xscale("log")
    # .rc("axes", edgecolor="black")
    plt.plot(t_j, acf_j[0:zero_point_crossing_j], color="tab:blue", linewidth=3)
    plt.plot(t_j, acf_j_fit["fit"], color="black", linestyle="dashed", linewidth=2)
    ax.set_xlabel(r"$\tau$ (s)", size=24)
    ax.set_ylabel(r"$g_J(\tau)$", size=24)
    plt.tick_params(labelsize=18)
    fig.set_size_inches(7, 5)
    plt.show()


def plot_bhalf_time(ts, bhalf_time, fit_error_time, factor=1e6):
    plt.figure(3)
    for i in range(2, len(ts), 35):
        plt.plot(ts[i] * factor, bhalf_time[i], "ro", linewidth=3)
        plt.errorbar(
            ts[i] * factor,
            bhalf_time[i],
            fit_error_time[1, i],
            color="k",
            linewidth=2,
        )
    plt.xlabel("Time ($\mu s$)", size=18)
    plt.ylabel("$B_{1/2}$ (mT)", size=18)
    plt.tick_params(labelsize=14)
    plt.gcf().set_size_inches(10, 5)
    plt.show()


def plot_exchange_interaction_in_solution(ts, trajectory_data, j):
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="black")
    color = "tab:red"
    plt.plot(ts, trajectory_data[:, 1] * 1e9, color=color)
    ax2 = ax.twinx()
    color2 = "tab:blue"
    plt.plot(ts, -j, color=color2)
    ax.set_xlabel("Time (ns)", size=24)
    ax.set_ylabel("Radical pair separation (nm)", size=24, color=color)
    ax2.set_ylabel("Exchange interaction (mT)", size=24, color=color2)
    ax.tick_params(axis="y", labelsize=18, labelcolor=color)
    ax.tick_params(axis="x", labelsize=18, labelcolor="k")
    ax2.tick_params(labelsize=18, labelcolor=color2)
    fig.set_size_inches(7, 5)
    plt.show()


def spin_state_labels(sim: HilbertSimulation):
    if len(sim.radicals) != 2:
        raise ValueError(
            "Density matrix plotting make little sense for non-radical pairs!"
        )
    # multiplicities = sim.multiplicities[len(sim.radicals) :]
    multiplicities = [r.multiplicity for r in sim.radicals]
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


def _format_label(t):
    return f"$\\vert {t} \\rangle$"
