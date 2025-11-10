#! /usr/bin/env python
r"""Plotting utilities for spin dynamics, tensors, and Monte Carlo trajectories.

This module collects convenience routines for visualising common
objects in spin chemistry and molecular simulations, including:
energy-level diagrams, anisotropy/tensor surfaces on the unit sphere,
3D random-walk trajectories (free and caged), density-matrix bar
animations, and general helper plots for molecules and time–series.

Functions in this module build on NumPy and Matplotlib’s 3D toolkit and
are intended for exploratory analysis and figure generation inside
notebooks or scripts. Most functions draw directly to the active
Matplotlib figure and return `None`, except where noted.

Contents:

        - `anisotropy_surface`: 3D surface whose radius/color encodes Re(Y(θ,φ)).

        - `density_matrix_animation`: Animated 3D bar plot of :math:`\lvert\rho\rvert` over time.

        - `linear_energy_levels`: Event-plot (vertical lines) of Hamiltonian eigenvalues.

        - `energy_levels`: Eigen-energies vs magnetic field for a `HilbertSimulation`.

        - `monte_carlo_free`: 3D trajectory of a free random walk (nm scaling).

        - `monte_carlo_caged`: 3D trajectory within a spherical cage of radius `r_max`.

        - `plot_3d_results`: Colored 3D surface z(x, y) with adjustable camera.

        - `plot_autocorrelation_fit`: Autocorrelation (log-x) with fitted curve.

        - `plot_bhalf_time`: Discrete B_{1/2} estimates vs time with error bars.

        - `plot_exchange_interaction_in_solution`: Separation and exchange (twin y-axes).

        - `plot_general`: Simple x–y line plot with labels/styles.

        - `plot_molecule`: Minimal 3D stick model from atom coords and bonds.

        - `set_equal_aspect`: Equalizes 3D axis scales to data range.

        - `spin_state_labels`: LaTeX-formatted ket labels for radical pairs.

        - `_format_label`: Helper that wraps text in a Dirac ket for LaTeX.

        - `visualise_tensor`: Rank-2 tensor surface on the sphere after rotation/shift.

Notes:

        - Many plotting functions assume Matplotlib 3D axes; some create and
          manage figures/axes internally. To integrate into existing layouts,
          adapt the code to accept/use a provided `Axes3D` where appropriate.

        - Units are noted in axis labels (e.g., nm, mT, s). Some helpers apply
          scaling factors (e.g., 1e9) for visualization; inspect the source if
          absolute coordinates are required.

        - `energy_levels` assumes a `HilbertSimulation` API exposing
          `total_hamiltonian` and `zeeman_hamiltonian`. Replace or wrap as needed.

Dependencies:

        - `numpy` for array operations.

        - `matplotlib` (including `mpl_toolkits.mplot3d`, `cm`, and `colors`) for plotting.

        - `matplotlib.animation.FuncAnimation` for animated density-matrix plots.

Raises:

        ValueError: `spin_state_labels` raises if `sim` does not contain exactly
            two radicals (labels are defined for radical pairs).

See also:

        `matplotlib.pyplot`, `mpl_toolkits.mplot3d`, and
        domain-specific simulation objects (e.g., `HilbertSimulation`)
        used by the energy-level routines.

"""

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .simulation import HilbertSimulation, State
from .utils import spherical_to_cartesian

ELEMENT_COLORS = {
    "H": "lightcoral",
    "C": "black",
    "N": "blue",
    "O": "red",
    "F": "green",
    "Cl": "green",
    "Br": "brown",
    "I": "purple",
    "S": "gold",
    "P": "orange",
}


def anisotropy_surface(theta, phi, Y):
    """Anisotropy surface plot.

    Renders a 3D surface on the unit sphere whose radius is proportional
    to the real part of a field `Y(θ,φ)`, and colors the surface by `Re(Y)`.

    Args:
            theta (np.ndarray): Polar angles θ in radians (shape: (N,) or mesh-ready).
            phi (np.ndarray): Azimuthal angles φ in radians (shape: (M,) or mesh-ready).
            Y (np.ndarray): Complex field values sampled on (θ, φ); only the real
                component is used for coloring and scaling.

    Returns:
            None: Displays a Matplotlib 3D surface plot.
    """
    # TODO(vatai): clean up
    PH, TH = np.meshgrid(phi, theta)
    xyz = np.array([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)])

    Yx, Yy, Yz = Y.real * spherical_to_cartesian(TH, PH).T

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
    """Density matrix bar-plot animation.

    Builds a 3D bar plot over time from a sequence of density matrices,
    coloring bars by magnitude and returning a `FuncAnimation`.

    Args:
            rhos (Sequence[np.ndarray]): Time-ordered density matrices ρ_t
                (each shape: (N, N)).
            frames (int): Number of frames to render in the animation.
            bar3d_kwargs (dict): Keyword arguments forwarded to `Axes3D.bar3d`.
            axes_kwargs (dict): Keyword arguments forwarded to `Axes3D.set`.

    Returns:
            matplotlib.animation.FuncAnimation: The configured animation object.
    """
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
    """Linear spectrum event plot.

    Diagonalizes a Hamiltonian and shows its eigenvalues as vertical events,
    useful for compact level-spectrum visualization.

    Args:
            H (np.ndarray): Hamiltonian matrix (Hermitian).
            B (Any): Unused placeholder (kept for API compatibility).
            linecolour (str or tuple): Color spec for event lines.
            title (str): Plot title.

    Returns:
            None: Displays a vertical event plot of eigenvalues.
    """
    # todo(vatai): clean up
    # eigval = np.linalg.eigh(H)  # try eig(H)
    # E = np.real(eigval[0])  # 0 = eigenvalues, 1 = eigenvectors
    evals, _ = np.linalg.eig(H)
    evals = np.sort(evals)

    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.eventplot(evals, orientation="vertical", color=linecolour, linewidth=3)
    ax.set_title(title, size=18)
    ax.set_ylabel("Spin state energy / J", size=14)
    plt.tick_params(labelsize=14)


def energy_levels(sim: HilbertSimulation, B: np.ndarray, J=0, D=0):
    """Energy levels vs magnetic field.

    Computes eigen-energies of the total Hamiltonian `H = H_base + B0 * H_zee`
    for each applied field in `B` and plots levels as functions of `B0`.

    Args:
            sim (HilbertSimulation): Simulation in Hilbert space.
            B (np.ndarray): Magnetic field values (T).
            J (float): Isotropic exchange parameter (default: 0).
            D (float): Dipolar coupling parameter (default: 0).

    Returns:
            None: Displays a Matplotlib line plot of energy levels vs field.
    """
    # TODO(VATAI): DO THIS PROPERLY
    # TODO(VATAI): use tick labels
    assert (
        type(sim) == HilbertSimulation
    ), "plot.energy_levels assumes Hilbert space simulation"
    H_base = sim.total_hamiltonian(0, J, D)
    data = []

    for i, B0 in enumerate(B):
        temp = H_base + sim.zeeman_hamiltonian(B0)
        evals, _ = np.linalg.eig(temp)
        evals = np.sort(evals)
        data.append(evals)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    for evals in np.array(data).T:
        ax.plot(B, evals, "k-", linewidth=2)
    ax.set_xlabel("$B_0 / T$", size=14)
    ax.set_ylabel("Spin state energy / J", size=14)
    plt.tick_params(labelsize=14)


def monte_carlo_free(pos):
    """3D random-walk trajectory (free solution).

    Plots a radical-pair Monte Carlo path in free solution, scaling coordinates
    to nanometres for visualization and marking start and origin.

    Args:
            pos (np.ndarray): Cartesian positions along a trajectory
                (shape: (T, 3), in metres).

    Returns:
            None: Displays a 3D line plot of the walk.
    """
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
    ax.set_xlabel("$X$ / nm", size=14)
    ax.set_ylabel("$Y$ / nm", size=14)
    ax.set_zlabel("$Z$ / nm", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)


def monte_carlo_caged(pos, r_max):
    """3D random-walk trajectory (caged).

    Plots a Monte Carlo trajectory inside a spherical cage of radius `r_max`,
    rendering the wireframe boundary and the particle path.

    Args:
            pos (np.ndarray): Cartesian positions along a trajectory
                (shape: (T, 3), in metres).
            r_max (float): Cage radius (metres) for the confining sphere.

    Returns:
            None: Displays a 3D wireframe and trajectory plot.
    """
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
    ax.set_xlabel("$X$ / nm", size=14)
    ax.set_ylabel("$Y$ / nm", size=14)
    ax.set_zlabel("$Z$ / nm", size=14)
    # plt.xlim([-1, 1]); plt.ylim([-1, 1])
    plt.tick_params(labelsize=14)
    fig.set_size_inches(10, 10)


def plot_3d_results(
    xdata, ydata, zdata, xlabel, ylabel, zlabel, azim=-135, dist=10, elev=35, factor=1e6
):
    """3D surface plot of results.

    Creates a colored surface `z(x,y)` with adjustable 3D view parameters and
    axis labels—useful for field maps or parameter sweeps.

    Args:
            xdata (np.ndarray): X-axis coordinates.
            ydata (np.ndarray): Y-axis coordinates.
            zdata (np.ndarray): Z values on the meshgrid of (x, y).
            xlabel (str): Label for the X axis.
            ylabel (str): Label for the Y axis.
            zlabel (str): Label for the Z axis.
            azim (float): Azimuthal angle of the 3D view (deg).
            dist (float): Camera distance parameter.
            elev (float): Elevation angle of the 3D view (deg).
            factor (float): Multiplier applied to Y for unit scaling.

    Returns:
            None: Displays a Matplotlib 3D surface plot.
    """
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(projection="3d")
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap("viridis"))
    ax.set_facecolor("none")
    ax.grid(False)
    X, Y = np.meshgrid(xdata, ydata)
    ax.plot_surface(
        X,
        Y * factor,
        zdata,
        facecolors=cmap.to_rgba(zdata.real),
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel(xlabel, size=24, labelpad=15)
    ax.set_ylabel(ylabel, size=24, labelpad=15)
    ax.set_zlabel(zlabel, size=24, labelpad=18)
    # ax.set_proj_type("ortho")
    ax.azim = azim
    ax.dist = dist
    ax.elev = elev
    plt.tick_params(labelsize=18)
    fig.set_size_inches(10, 10)
    # plt.show()


def plot_autocorrelation_fit(t_j, acf_j, acf_j_fit, zero_point_crossing_j):
    """Autocorrelation with fit (log-scale).

    Plots an autocorrelation function up to its first zero crossing together
    with a fitted curve provided in `acf_j_fit["fit"]`.

    Args:
            t_j (np.ndarray): Lag times τ (s).
            acf_j (np.ndarray): Autocorrelation values g(τ).
            acf_j_fit (Mapping[str, np.ndarray]): Fit results containing key "fit".
            zero_point_crossing_j (int): Index of first zero crossing for g(τ).

    Returns:
            None: Displays a Matplotlib line plot on a log-x axis.
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("none")
    ax.grid(False)
    plt.axis("on")
    plt.xscale("log")
    # .rc("axes", edgecolor="black")
    plt.plot(t_j, acf_j[0:zero_point_crossing_j], color="tab:blue", linewidth=3)
    plt.plot(t_j, acf_j_fit["fit"], color="black", linestyle="dashed", linewidth=2)
    ax.set_xlabel(r"$\tau$ / s", size=24)
    ax.set_ylabel(r"$g_J(\tau)$", size=24)
    plt.tick_params(labelsize=18)
    fig.set_size_inches(5, 5)
    # plt.show()


def plot_bhalf_time(ts, bhalf_time, fit_error_time, style="ro", factor=1e6):
    """`B_{1/2}`-field width vs time.

    Plots discrete estimates of the half-field width `B_{1/2}` against time
    with error bars from fitting uncertainties.

    Args:
            ts (np.ndarray): Time points (s).
            bhalf_time (np.ndarray): Estimated `B_{1/2}` (mT) per time point.
            fit_error_time (np.ndarray): Fit errors (e.g., covariance-derived)
                where index [1, i] is used as y-error.
            style (str): Matplotlib line/marker style (default: "ro").
            factor (float): Multiplier applied to x-values for unit scaling.

    Returns:
            None: Displays a scatter plot with error bars.
    """
    plt.figure()
    for i in range(2, len(ts), 10):
        plt.plot(ts[i] * factor, bhalf_time[i], style, linewidth=3)
        plt.errorbar(
            ts[i] * factor,
            bhalf_time[i],
            fit_error_time[1, i],
            color="k",
            linewidth=2,
        )
    plt.xlabel(r"Time / $\mu s$", size=24)
    plt.ylabel("$B_{1/2}$ / mT", size=24)
    plt.tick_params(labelsize=18)
    plt.gcf().set_size_inches(5, 5)
    # plt.show()


def plot_exchange_interaction_in_solution(ts, trajectory_data, j):
    """Exchange interaction and separation vs time.

    Plots radical-pair separation and corresponding exchange interaction
    on twin y-axes against time.

    Args:
            ts (np.ndarray): Time points (s or ns as labeled).
            trajectory_data (np.ndarray): Trajectory array where column 1
                contains separations (m).
            j (np.ndarray): Exchange interaction values (mT), same length as `ts`.

    Returns:
            None: Displays a Matplotlib plot with twin y-axes.
    """
    fig = plt.figure()
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
    ax.set_xlabel("Time / ns", size=24)
    ax.set_ylabel("Radical pair separation / nm", size=24, color=color)
    ax2.set_ylabel("Exchange interaction / mT", size=24, color=color2)
    ax.tick_params(axis="y", labelsize=18, labelcolor=color)
    ax.tick_params(axis="x", labelsize=18, labelcolor="k")
    ax2.tick_params(labelsize=18, labelcolor=color2)
    fig.set_size_inches(5, 5)
    # plt.show()


def plot_general(
    xdata, ydata, xlabel, ylabel, style="-", label=[], colors="r", factor=1
):
    """General line plot.

    Convenience wrapper for a simple x–y line plot with label, style, color,
    and optional scaling of the x-values.

    Args:
            xdata (np.ndarray): X-axis data.
            ydata (np.ndarray): Y-axis data.
            xlabel (str): Label for the X axis.
            ylabel (str): Label for the Y axis.
            style (str): Matplotlib style string (default: "-").
            label (list or str): Legend label(s).
            colors (Any): Matplotlib color spec (default: "r").
            factor (float): Multiplier applied to x-values.

    Returns:
            None: Displays a Matplotlib line plot.
    """
    # plt.figure()
    plt.plot(xdata * factor, ydata, style, linewidth=3, label=label, color=colors)
    plt.xlabel(xlabel, size=24)
    plt.ylabel(ylabel, size=24)
    plt.legend()
    plt.tick_params(labelsize=18)
    plt.gcf().set_size_inches(10, 5)


def plot_molecule(
    ax, labels, elements, coords, bonds, show_labels=True, show_atoms=False
):
    """Molecular stick model.

    Draws a simple 3D molecular diagram from atom coordinates and bond pairs,
    optionally showing atom spheres and/or atom labels.

    Args:
            ax (mpl_toolkits.mplot3d.Axes3D): Target 3D axes.
            labels (Sequence[str]): Per-atom labels (e.g., element symbols).
            elements (Sequence[str]): Per-atom element identifiers.
            coords (np.ndarray): Cartesian coordinates (Å or nm) of shape (N, 3).
            bonds (Sequence[Tuple[int,int]]): Index pairs of bonded atoms.
            show_labels (bool): Whether to render atom labels.
            show_atoms (bool): Whether to render atom scatter points.

    Returns:
            None: Draws onto the provided axes.
    """
    colors = [ELEMENT_COLORS.get(e, "gray") for e in elements]
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
    if show_atoms:
        ax.scatter(X, Y, Z, s=100, c=colors, depthshade=True)
    if show_labels:
        for lab, x, y, z in zip(labels, X, Y, Z):
            ax.text(x + 0.4, y + 0.4, z + 0.4, lab, fontsize=10)
    for i, j in bonds:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], "k-", linewidth=4)
    set_equal_aspect(ax, X, Y, Z)


def plot_sphere(ax, radius=1, color="black", alpha=0.05):
    """
    Plot a wire-framed, lightly shaded sphere on a Matplotlib 3D axis.

    The surface is rendered via ``Axes3D.plot_surface`` using a spherical
    parameterisation, and three faint coordinate axes (x, y, z) are
    overlaid through the sphere’s center for orientation. A thin great-circle
    outline is also drawn.

    Args:

        ax : matplotlib.axes._subplots.Axes3DSubplot or mpl_toolkits.mplot3d.Axes3D
            A 3D Matplotlib axes object on which to draw.

        radius : float, optional
            Sphere radius. Default is ``1``.

        color : str or tuple, optional
            Base color for the sphere surface and outlines. Default is ``'black'``.

        alpha : float, optional
            Surface transparency in ``[0, 1]``. Default is ``0.05``.

    Returns:
        None
            The function draws on ``ax`` in place and returns ``None``.
    """
    # draw sphere
    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

    # great circle (equator)
    u = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(u) * np.sin(-np.pi / 2)
    y = radius * np.sin(u) * np.sin(-np.pi / 2)
    z = radius * np.cos(-np.pi / 2 * np.ones(len(u)))
    ax.plot(x, y, z, lw=1, color=color, alpha=0.2)
    ax.plot(z, y, x, lw=1, color=color, alpha=0.2)

    # coordinate axes through the center
    R, N, N = np.linspace(-radius, radius, 100), np.zeros(100), np.zeros(100)
    ax.plot(R, N, N, lw=1, c="r", alpha=0.2)  # x-axis
    ax.plot(N, R, N, lw=1, c="g", alpha=0.2)  # y-axis
    ax.plot(N, N, R, lw=1, c="b", alpha=0.2)  # z-axis
    return


def set_equal_aspect(ax, X, Y, Z):
    """Equalise 3D aspect by data range.

    Sets axis limits so the plotted 3D data have equal scale along each axis.

    Args:
            ax (mpl_toolkits.mplot3d.Axes3D): Target 3D axes.
            X (np.ndarray): X coordinates.
            Y (np.ndarray): Y coordinates.
            Z (np.ndarray): Z coordinates.

    Returns:
            None: Adjusts axis limits in place.
    """
    rng = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    xb = yb = zb = 0.5 * rng
    ax.set_xlim(X.mean() - xb, X.mean() + xb)
    ax.set_ylim(Y.mean() - yb, Y.mean() + yb)
    ax.set_zlim(Z.mean() - zb, Z.mean() + zb)


def spin_state_labels(sim: HilbertSimulation):
    """Spin-state labels for a radical pair.

    Generates human-readable kets for singlet/triplet sublevels combined
    with radical spin projections, suitable for plot tick labels.

    Args:
            sim (HilbertSimulation): Simulation containing exactly two radicals.

    Returns:
            list[str]: LaTeX-formatted labels like ``"$\\vert T_+, +1/2 \\rangle$"``.
    """
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
    """LaTeX ket formatter.

    Wraps a raw label string inside a Dirac ket for LaTeX rendering.

    Args:
            t (str): Bare label text (e.g., ``"T_+, +1/2"``).

    Returns:
            str: LaTeX string ``"$\\vert {t} \\rangle$"``.
    """
    return f"$\\vert {t} \\rangle$"


def visualise_tensor(ax, tensor, rot_matrix, coords, colour):
    """Rank-2 tensor visualisation on a sphere.

    Projects a symmetric rank-2 tensor onto directions on the unit sphere,
    scales radius by the directional quadratic form, and plots the resulting
    surface in 3D after applying a rotation and translation.

    Args:
            ax (mpl_toolkits.mplot3d.Axes3D): Target 3D axes.
            tensor (np.ndarray): 3×3 tensor (e.g., hyperfine or g-tensor).
            rot_matrix (np.ndarray): 3×3 rotation matrix to orient the tensor.
            coords (np.ndarray): 3-vector translation applied to the surface.
            colour (Any): Matplotlib color spec for the surface.

    Returns:
            np.ndarray: The rendered surface coordinates with shape (Nθ, Nφ, 3).
    """
    resolution = 30
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)

    tensor_vis = np.zeros([len(theta), len(phi), 3])

    for i in range(0, len(theta)):
        for j in range(0, len(phi)):

            xyz = np.array(
                [
                    np.sin(theta[i]) * np.cos(phi[j]),
                    np.sin(theta[i]) * np.sin(phi[j]),
                    np.cos(theta[i]),
                ]
            )

            tensor_vis[i, j] = (
                np.dot(
                    np.dot(
                        xyz.T, np.array(rot_matrix) @ tensor @ np.array(rot_matrix).T
                    ),
                    xyz,
                )
                * xyz.T
                + coords
            )

    ax.plot_surface(
        tensor_vis[:, :, 0],
        tensor_vis[:, :, 1],
        tensor_vis[:, :, 2],
        color=colour,
        edgecolor="none",
    )
    return tensor_vis
