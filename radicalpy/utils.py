#!/usr/bin/env python
"""
Utilities for spin dynamics, magnetic‐resonance and molecular simulations.

This module collects small, reusable helpers used across the codebase:
unit conversions, geometry on spheres, spectrum building blocks, I/O
parsers for molecular files and quantum-chemistry outputs, and a few
domain-specific utilities for MARY/NMR workflows.

Main contents
-------------
Constants
    - ``COVALENT_RADII``: Covalent radii (Å) for common elements; used by
      bonding heuristics and label/element inference.

CLI/testing
    - ``is_fast_run()``: Check ``--fast`` CLI flag to run lighter examples/tests.

Fitting & line shapes
    - ``Bhalf_fit(B, MARY)``: Fit MARY data to a Lorentzian to obtain
      :math:`B_{1/2}` and goodness-of-fit.
    - ``Lorentzian(B, amplitude, Bhalf)``: MARY Lorentzian model.

Unit conversions
    - ``Gauss_to_mT/MHz/…``, ``MHz_to_mT/Gauss/…``,
      ``mT_to_MHz/Gauss/…``,
      ``angular_frequency_in_MHz``, ``*_to_angular_frequency``:
      Consistent conversions between G, mT, MHz and rad·s⁻¹·T⁻¹.

Angular grids & spherical geometry
    - ``anisotropy_check(theta, phi)``: Validate angle grids and parity (θ odd/φ even).
    - ``_check_full_sphere(theta, phi)``: Ensure full-sphere linspace grids.
    - ``spherical_average(...)``: Weighted average over a θ/φ grid.
    - ``cartesian_to_spherical(...)``, ``spherical_to_cartesian(...)``:
      Coordinate transforms.

Autocorrelation
    - ``autocorrelation(data, factor=1)``: FFT-based trajectory autocorrelation.

CIDNP helper models
    - ``cidnp_polarisation_*``: Exponential, truncated diffusion, and full
      diffusion models; see docstrings for details.
    - ``s_t0_omega(deltag, B0, hfc_star, onuc_all)``: ω₊/ω₋ for S–T₀ mixing.

NMR building blocks
    - ``nmr_chemical_shift_real/im...``: cos/sin(2π f t) modulators.
    - ``nmr_scalar_coupling_modulation``: J-modulation term.
    - ``nmr_t2_relaxation``: exp(−t/T₂) envelope.

Lock-in MARY helpers
    - ``modulated_signal(...)``, ``reference_signal(...)``:
      Time-domain modulation and reference waves.
    - ``mary_lorentzian(mod_signal, lfe_magnitude)``: Simple MARY line shape.

Molecular geometry & visualization
    - ``define_xyz(...)``: Construct an orthonormal (x,y,z) basis from
      atom/point triplets.
    - ``get_angle_between_plane(A, B)``: Angle (deg) between plane normals.
    - ``get_rotation_matrix_euler_angles(A, B)``: Rotation matrix and ZXZ
      Euler angles mapping basis A→B.
    - ``rodrigues_rotation(v, k, theta)``: Rotate vectors around axis k by θ.
    - ``rotate_axes(A, x, y, z)``: Apply a rotation matrix to a basis.
    - ``enumerate_spin_states_from_base(base)``: Mixed-radix enumeration of
      spin projections (e.g., 2 for spin-½, 3 for spin-1).
    - ``infer_bonds(elements, coords, ...)``: Heuristic covalent bonding.

File I/O & parsing
    - ``parse_xyz(path)``, ``parse_label_xyz_txt(path)``, ``parse_pdb(path, ...)``:
      Read coordinates and optional bonds/labels for plotting.
    - ``pdb_label(atom, scheme=...)``: Human-readable PDB atom labels.
    - ``write_xyz/mol_to_plot_arrays/write_pdb/write_sdf``: Minimal writers and
      plotting helpers.
    - ``read_trajectory_files(dir, scale)``: Concatenate text trajectories from a folder.

Quantum-chemistry integration (ORCA)
    - ``read_orca_hyperfine(path, version=6)``: Dispatch to ORCA v6 parsers.
    - Internal: ``_hyperfine_from_orca6_out`` / ``_hyperfine_from_orca6_property_txt``.
      Return zero-based nucleus indices, isotope labels, and 3×3 HFC tensors (mT).
    - ``read_lines_utf8_or_utf16le``: Robust text decoding for ORCA outputs.

Chemistry utilities (RDKit)
    - ``smiles_to_3d(smiles, add_h=True, opt='mmff')``: Generate a 3D conformer
      (ETKDG) with optional MMFF/UFF optimisation.
    - ``mol_to_plot_arrays(mol)``: Extract elements, labels, coords and bonds.

Spectral density
    - ``spectral_density(omega, tau_c)``: :math:`J(ω) = τ_c / (1 + ω^2 τ_c^2)`.

Notes
-----
- Many routines are vectorised and expect NumPy arrays; see individual
  docstrings for shapes and units.
- Angle conventions use radians: ``theta ∈ [0, π]``, ``phi ∈ [0, 2π]``.
- Unit conversions rely on physical constants from ``radicalpy.constants`` (``C``).
"""

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.fftpack import fft, ifft, ifftshift
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from .shared import constants as C

# Covalent radii (Å) for many common elements
COVALENT_RADII = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
}


def is_fast_run():
    """Is the `--fast` parameter set at execution.

    This function helps examples to be used as tests.  By running the
    example with the `--fast` option, a faster version of main can be
    called (e.g., by setting fewer number of time steps etc.).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        default=False,
        action="store_true",
        help="If set, the experiment should perform a reduced number of steps.",
    )
    args = parser.parse_args()
    return args.fast


def Bhalf_fit(
    B: np.ndarray, MARY: np.ndarray
) -> Tuple[float, np.ndarray, float, float]:
    """B_1/2 fit for MARY spectra.

    Args:
            B (np.ndarray): Magnetic field values (x-axis).
            MARY (np.ndarray): Magnetic field effect data
                (y-axis). Use the `MARY` entry in the result of
                `radicalpy.simulation.HilbertSimulation.MARY`.

    Returns:
            (float, np.ndarray, float, float):
            - `Bhalf` (float): The magnetic field strength at half the
              saturation magnetic field.
            - `fit_result` (np.ndarray): y-axis from fit.
            - `fit_error` (float): Standard error for the fit.
            - `R2` (float): R-squared value for the fit.
    """
    popt_MARY, pcov_MARY = curve_fit(
        Lorentzian,
        B,
        MARY,
        p0=[MARY[-1], int(len(B) / 2)],
        maxfev=1000000,
    )
    fit_error = np.sqrt(np.diag(pcov_MARY))

    A_opt_MARY, Bhalf_opt_MARY = popt_MARY
    fit_result = Lorentzian(B, *popt_MARY)
    Bhalf = np.abs(Bhalf_opt_MARY)

    y_pred_MARY = Lorentzian(B, *popt_MARY)
    R2 = r2_score(MARY, y_pred_MARY)

    return Bhalf, fit_result, fit_error, R2


def Bhalf_LFEhalf_fit(
    B: np.ndarray, MARY: np.ndarray, LFE_position: float = 1.0
) -> Tuple[float, np.ndarray, float, float]:
    """B_1/2 and LFE_1/2 fitting for MARY spectra.

    Args:
            B (np.ndarray): Magnetic field values (x-axis).
            MARY (np.ndarray): Magnetic field effect data
                (y-axis). Use the `MARY` entry in the result of
                `radicalpy.simulation.HilbertSimulation.MARY`.
            LFE_position (float): Initial guess for the low field effect position.

    Returns:
            (float, np.ndarray, float, float):
            - `Bhalf` (float): The magnetic field strength at half the
              saturation magnetic field.
            - `LFEhalf` (float): The magnetic field strength at half the
              low field effect magnetic field.
            - `fit_result` (np.ndarray): y-axis from fit.
            - `fit_error` (float): Standard error for the fit.
            - `R2` (float): R-squared value for the fit.
    """
    popt_MARY, pcov_MARY = curve_fit(
        double_Lorentzian,
        B,
        MARY,
        p0=[MARY[-1], int(len(B) / 2), MARY[LFE_position], LFE_position],
        maxfev=1000000,
    )
    fit_error = np.sqrt(np.diag(pcov_MARY))

    A_opt_MARY, Bhalf_opt_MARY, A_opt_LFE, LFEhalf_opt_MARY = popt_MARY
    fit_result = double_Lorentzian(B, *popt_MARY)
    Bhalf = np.abs(Bhalf_opt_MARY)
    LFEhalf = np.abs(LFEhalf_opt_MARY)

    y_pred_MARY = double_Lorentzian(B, *popt_MARY)
    R2 = r2_score(MARY, y_pred_MARY)

    return Bhalf, LFEhalf, fit_result, fit_error, R2


def Gauss_to_MHz(Gauss: float) -> float:
    """Convert Gauss to MHz.

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: The magnetic flux density converted to MHz.
    """
    return Gauss / (1e-10 * C.g_e * C.mu_B / C.h)


def Gauss_to_angular_frequency(Gauss: float) -> float:
    """Convert Gauss to angular frequency.

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: The magnetic flux density converted to angular
            frequency (rad/s/T).
    """
    return Gauss * (C.mu_B / C.hbar * C.g_e / 1e10)


def Gauss_to_mT(Gauss: float) -> float:
    """Convert Gauss to millitesla.

    Args:
            Gauss (float): The magnetic flux density in Gauss (G).

    Returns:
            float: The magnetic flux density converted to millitesla
            (mT).
    """
    return Gauss / 10


def Lorentzian(B: np.ndarray, amplitude: float, Bhalf: float) -> np.ndarray:
    """Lorentzian function for MARY spectra.

    More information in `radicalpy.utils.Bhalf_fit` (where this is
    used).

    Args:
            B (np.ndarray): The x-axis magnetic field values.
            amplitude (float): The amplitude of the saturation field value.
            Bhalf (float): The magnetic field strength at half the
                saturation field value.

    Returns:
            np.ndarray: Lorentzian function for MARY spectrum.
    """
    return (amplitude / Bhalf**2) - (amplitude / (B**2 + Bhalf**2))


def double_Lorentzian(
    B: np.ndarray, amplitude: float, Bhalf: float, LFE_amplitude: float, LFEhalf: float
) -> np.ndarray:
    """Double Lorentzian function for MARY spectra.

    More information in `radicalpy.utils.Bhalf_fit` (where this is
    used).

    Args:
            B (np.ndarray): The x-axis magnetic field values.
            amplitude (float): The amplitude of the saturation field value.
            Bhalf (float): The magnetic field strength at half the
                saturation field value.
            LFE_amplitude (float): The amplitude of the LFE component.
            LFEhalf (float): The magnetic field strength at half the
                LFE field value.

    Returns:
            np.ndarray: Double Lorentzian function for MARY spectrum.
    """
    return -amplitude * (B**2 / (B**2 + Bhalf**2)) + LFE_amplitude * (
        B**2 / (B**2 + LFEhalf**2)
    )


def MHz_in_angular_frequency(MHz: float) -> float:
    """Convert MHz into angular frequency.

    Args:
            MHz (float): The angular frequency in MHz/T

    Returns:
            float: The angular frequency converted to rad/s/T.
    """
    return MHz * (2 * np.pi)


def MHz_to_Gauss(MHz: float) -> float:
    """Convert Megahertz to Gauss.

    Args:
            MHz (float): The frequency in Megahertz (MHz).

    Returns:
            float: Megahertz (MHz) converted to Gauss (G).
    """
    return MHz / (1e-10 * C.g_e * C.mu_B / C.h)


def MHz_to_mT(MHz: float) -> float:
    """Convert Megahertz to milltesla.

    Args:
            MHz (float): The frequency in Megahertz (MHz).

    Returns:
            float: Megahertz (MHz) converted to millitesla (mT).
    """
    return MHz / (1e-9 * C.g_e * C.mu_B / C.h)


def angular_frequency_in_MHz(ang_freq: float) -> float:
    """Convert angular frequency into MHz.

    Args:
            ang_freq (float): The angular frequency in rad/s/T.

    Returns:
            float: The angular frequency converted to MHz/T.
    """
    return ang_freq / (2 * np.pi)


def angular_frequency_to_Gauss(ang_freq: float) -> float:
    """Convert angular frequency to Gauss.

    Args:
            ang_freq (float): The angular frequency in rad/s/T.

    Returns:
            float: The angular frequency converted to Gauss (G).
    """
    return ang_freq / (C.mu_B / C.hbar * C.g_e / 1e10)


def angular_frequency_to_mT(ang_freq: float) -> float:
    """Convert angular frequency to millitesla.

    Args:
            ang_freq (float): The angular frequency in rad/s/T.

    Returns:
            float: The angular frequency converted to millitesla (mT).
    """
    return ang_freq / (C.mu_B / C.hbar * C.g_e / 1e9)


def anisotropy_check(
    theta: float | np.ndarray, phi: float | np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and normalise anisotropy angle grids.

    Accepts scalar or array inputs for the polar (``theta``) and azimuthal
    (``phi``) angles, promotes scalars to 1-element arrays, and enforces
    domain and parity constraints that downstream integrators rely on.

    Args:
            theta (float or np.ndarray): Polar angle(s) in radians.
                Must lie in ``[0, π]``. If an array is provided and
                ``len(theta) > 1`` while ``len(phi) > 1``, then
                ``len(theta)`` must be **odd**.
            phi (float or np.ndarray): Azimuthal angle(s) in radians.
                Must lie in ``[0, 2π]``. If an array is provided and
                ``len(theta) > 1`` while ``len(phi) > 1``, then
                ``len(phi)`` must be **even**.

    Returns:
            (np.ndarray, np.ndarray): The validated angle arrays
            ``(theta, phi)`` (possibly promoted from scalars).

    Raises:
            ValueError: If angles are out of range or the grid parity
                constraints are violated.
    """
    if isinstance(theta, float):
        theta = np.array([theta])
    if isinstance(phi, float):
        phi = np.array([phi])
    if theta.min() < 0 or np.pi < theta.max():
        raise ValueError("Value of `theta` needs to be between 0 and pi!")
    if min(phi) < 0 or 2 * np.pi < max(phi):
        raise ValueError("Value of `phi` needs to be between 0 and 2*pi!")
    lt, lp = len(theta), len(phi)
    if lt > 1 and lp > 1:
        # theta odd, phi even
        if lt % 2 == 0:
            raise ValueError("Number of `len(theta)` needs to be odd!")
        if lp % 2 == 1:
            raise ValueError("Number of `len(phi)` needs to be even!")
    return theta, phi


def autocorrelation(data: np.ndarray, factor: int = 1) -> np.ndarray:
    """Calculate the autocorrelation of a trajectory.

    An FFT-based implementation of the autocorrelation for Monte Carlo
    or molecular dynamics trajectories (or any other time dependent
    value).

    Args:
            data (np.ndarray): The time dependent trajectory.
            factor (int): Data length reduction factor.

    Returns:
            np.ndarray: The autocorrelation of the trajectory.
    """
    datap = ifftshift((data - np.average(data)) / np.std(data))
    n = datap.shape[0]
    datap = np.r_[datap[: n // factor], np.zeros_like(datap), datap[n // factor :]]
    f = fft(datap)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    result = np.real(pi)[: n // factor] / np.arange(n, 0, -1)[: n // factor]
    result = np.delete(result, 0)
    return result


def cartesian_to_spherical(
    x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Convert Cartesian coordinates to spherical coordinates.

    Args:
            x (float or np.ndarray): Coordinate(s) in the x plane.
            y (float or np.ndarray): Coordinate(s) in the y plane.
            z (float or np.ndarray): Coordinate(s) in the z plane.

    Returns:
            r (float or np.ndarray): The radial distance(s).
            theta (float or np.ndarray): The polar angle(s).
            phi (float or np.ndarray): The azimuthal angle(s).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def cidnp_polarisation_diffusion_model(
    omega_plus: np.ndarray, omega_minus: np.ndarray, alpha: float = 1.5
) -> float:
    """
    Compute the CIDNP polarisation using the full diffusion model.

    Parameters
    ----------
    omega_plus : np.ndarray
        Array of omega+ frequencies (rad/s).
    omega_minus : np.ndarray
        Array of omega- frequencies (rad/s).
    alpha : float
        Dimensionless parameter for the Adrian diffusion model 2p/m.

    Returns
    -------
    p : float
        CIDNP polarisation.
    """
    T_to_angular_frequency = 2.8e10 * 2.0 * np.pi
    op_T = omega_plus / T_to_angular_frequency
    om_T = omega_minus / T_to_angular_frequency
    r_op = np.sqrt(np.abs(op_T))
    r_om = np.sqrt(np.abs(om_T))
    a = float(alpha)
    f = lambda r: 1.0 - np.exp(-a * r) * np.cos(a * r)
    return np.sum(f(r_op) - f(r_om), dtype=np.float64)


def cidnp_polarisation_exponential_model(
    ks: float, omega_plus: np.ndarray, omega_minus: np.ndarray
) -> float:
    """
    Compute the CIDNP polarisation using the exponential model.

    Parameters
    ----------
    ks : float
        Singlet recombination rate (s^-1).
    omega_plus : np.ndarray
        Array of omega+ frequencies (rad/s).
    omega_minus : np.ndarray
        Array of omega- frequencies (rad/s).

    Returns
    -------
    p : float
        CIDNP polarisation.
    """
    ks2 = float(ks) ** 2
    term = (omega_plus**2) / (ks2 + 4.0 * omega_plus**2) - (omega_minus**2) / (
        ks2 + 4.0 * omega_minus**2
    )
    return np.sum(term, dtype=np.float64)


def cidnp_polarisation_truncated_diffusion_model(
    omega_plus: np.ndarray, omega_minus: np.ndarray
) -> float:
    """
    Compute the CIDNP polarisation using the truncated t^{-3/2} diffusion model.

    Parameters
    ----------
    omega_plus : np.ndarray
        Array of omega+ frequencies (rad/s).
    omega_minus : np.ndarray
        Array of omega- frequencies (rad/s).

    Returns
    -------
    p : float
        CIDNP polarisation.
    """
    return np.sum(
        np.sqrt(np.abs(omega_plus)) - np.sqrt(np.abs(omega_minus)), dtype=np.float64
    )


def define_xyz(x1, x2, z1, z2, z3, z4):
    """Construct a right-handed orthonormal basis from atom/point triplets.

    The z-axis is defined as the normal to the plane spanned by the
    vectors ``(z1 − z2)`` and ``(z3 − z4)`` (right-hand rule).
    The x-axis is the normalised direction from ``x2`` to ``x1``.
    The y-axis completes the right-handed triad via ``y = z × x``,
    and x is re-orthogonalised by ``x = y × z``.

    Args:
            x1, x2 (array-like): Points defining the x-axis direction.
            z1, z2, z3, z4 (array-like): Points defining two in-plane
                vectors whose cross product yields the z-axis.

    Returns:
            (np.ndarray, np.ndarray, np.ndarray): Unit vectors
            ``(x, y, z)`` forming a right-handed orthonormal basis.
    """
    a = np.array(z1) - np.array(z2)
    b = np.array(z3) - np.array(z4)
    z = np.cross(a, b)
    z = z / np.linalg.norm(z)
    x = (np.array(x1) - np.array(x2)) / np.linalg.norm((np.array(x1) - np.array(x2)))
    y = np.cross(z, x)
    x = np.cross(y, z)
    return x, y, z


def eigensorter(H, threshold=1e-12):
    """
    Diagonalise a square matrix and return eigenvalues in ascending order
    together with the corresponding (row-stacked) eigenvectors.

    This helper wraps :func:`numpy.linalg.eig`, sorts the eigenpairs by
    ascending eigenvalue, and returns the eigenvectors **transposed** so that
    each eigenvector appears as a row in the returned array. A simple
    correctness check is performed by evaluating the residual norm
    ``|| λ_k - v_k† H v_k ||`` over all eigenpairs. If this residual exceeds
    ``threshold``, the function prints the error size and returns the scalar
    error instead of the eigenpairs.

    Parameters
    ----------
    H : ndarray of shape (N, N), complex or float
        Matrix to diagonalise. For Hermitian inputs consider using
        :func:`numpy.linalg.eigh` elsewhere for improved numerical stability,
        but this routine intentionally uses ``eig`` to support non-Hermitian
        cases.
    threshold : float, optional
        Maximum allowed residual norm before reporting failure. Default is
        ``1e-12``.

    Returns
    -------
    evals : ndarray of shape (N,)
        Eigenvalues sorted in ascending order.
    evecs : ndarray of shape (N, N)
        Row-stacked eigenvectors corresponding to ``evals``. Each row ``k``
        is the eigenvector ``v_k`` satisfying (approximately)
        ``H @ v_k = evals[k] * v_k``.
    error : float
        If the residual norm exceeds ``threshold``, the function returns this
        scalar error value instead of ``(evals, evecs)``.

    Notes
    -----
    - Sorting is performed on the (possibly complex) eigenvalues using
      :func:`numpy.argsort`, which orders by real part then imaginary part.
    - The residual check computes
      ``error = || [ λ_k − v_k† H v_k ]_k ||_2`` as a single aggregate
      measure over all eigenpairs.
    - For Hermitian ``H``, eigenvectors are expected to be orthonormal up to
      numerical precision; for general ``H`` they need not be.
    """
    # calculate eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(H)
    # get the list of sorted indices from the eigenvalues
    ids = np.argsort(evals)
    # sort the eigenvalues
    evals = evals[ids]
    # sort the eigenvectors
    evecs = evecs[:, ids]
    # transpose the vectors (row-stack)
    evecs = evecs.T
    # test the result
    error = np.linalg.norm(
        [evals[k] - vec.conj().T @ H @ vec for k, vec in enumerate(evecs)]
    )
    if error > threshold:
        print("Error size: ", error)
        return error
    else:
        return evals, evecs


def enumerate_spin_states_from_base(base: int) -> np.ndarray:
    """
    Return all spin-state patterns for a mixed-radix 'base' (e.g. [2,2,3,...]).
    Each row corresponds to one configuration. For base[i]=b, the digit d in [0..b-1]
    maps to spin projection m = (b-1)/2 - d.
    """
    base = np.asarray(base, dtype=int)
    size = len(base)
    total = int(np.prod(base))

    # Build digits for all states at once via mixed-radix division
    # states: 0..total-1
    n = np.arange(total, dtype=np.int64)[:, None]  # (total, 1)
    digits = np.empty((total, size), dtype=np.int64)

    # Least-significant position first
    for i in range(size):
        b = base[i]
        digits[:, i] = n[:, 0] % b
        n //= b

    # Map digits -> spin projections m_i = (b-1)/2 - digit
    m = (base.astype(np.float64) - 1.0) / 2.0
    patterns = m[None, :] - digits.astype(np.float64)
    return patterns  # shape: (total, size)


def get_angle_between_plane(A, B):
    """Angle between two plane normals in degrees.

    Computes the angle between vectors ``A`` and ``B`` (interpreted as
    plane normals). The result is within ``[0, 180]`` degrees.

    Args:
            A (array-like): First normal vector.
            B (array-like): Second normal vector.

    Returns:
            float: Angle between ``A`` and ``B`` in degrees.
    """
    tmp = np.linalg.norm(A) * np.linalg.norm(B)
    angle = np.arccos(np.dot(A, B) / tmp)
    return np.degrees(angle)


def get_rotation_matrix_euler_angles(A, B):
    """Rotation matrix and ZXZ Euler angles mapping basis A → basis B.

    Forms the rotation matrix ``R = Aᵀ B`` that maps coordinates expressed
    in the (column) basis ``A`` to the basis ``B``, and extracts the ZXZ
    Euler angles ``(α, β, γ)`` in radians.

    Args:
            A (np.ndarray): 3×3 column-basis matrix of frame A.
            B (np.ndarray): 3×3 column-basis matrix of frame B.

    Returns:
            (np.ndarray, float, float, float):
            - ``R``: 3×3 rotation matrix.
            - ``alpha`` (rad): First ZXZ angle.
            - ``beta`` (rad): Second ZXZ angle.
            - ``gamma`` (rad): Third ZXZ angle.

    Notes:
            Handles the gimbal-lock case ``|R[2,2]| = 1`` explicitly.
    """
    R = np.dot(np.array(A).T, np.array(B))
    if np.abs(R[2, 2]) != 1:
        beta = np.arccos(R[2, 2])
        if R[2, 1] >= 0:
            alpha = np.arccos(R[2, 0] / np.sin(beta))
        else:
            alpha = 2 * np.pi - np.arccos(R[2, 0] / np.sin(beta))
        if R[1, 2] >= 0:
            gamma = np.arccos(-R[0, 2] / np.sin(beta))
        else:
            gamma = 2 * np.pi - np.arccos(-R[0, 2] / np.sin(beta))
    else:
        beta = 0
        gamma = 0
        if R[0, 1] / R[2, 2] >= 0:
            alpha = np.arccos(R[0, 0] / R[2, 2])
        else:
            alpha = 2 * np.pi - np.arccos(R[0, 0] / R[2, 2])
    return R, alpha, beta, gamma


def infer_bonds(elements, coords, scale=1.20, max_dist=2.0):
    """Heuristic covalent-bond detection from interatomic distances.

    Bonds are inferred when the interatomic distance is less than
    ``scale × (r_i + r_j)`` using covalent radii (fallback 0.77 Å) and
    also less than ``max_dist`` to avoid spurious long bonds.

    Args:
            elements (list[str]): Atomic symbols per atom.
            coords (np.ndarray): Cartesian coordinates, shape ``(N, 3)`` (Å).
            scale (float): Multiplicative tolerance on summed covalent radii.
            max_dist (float): Hard cutoff distance (Å).

    Returns:
            list[tuple[int, int]]: Pairs of atom indices ``(i, j)`` indicating bonds.
    """
    bonds = []
    n = len(elements)
    for i in range(n):
        ri = COVALENT_RADII.get(elements[i], 0.77)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(elements[j], 0.77)
            cutoff = scale * (ri + rj)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= min(cutoff, max_dist):
                bonds.append((i, j))
    return bonds


def mary_lorentzian(mod_signal: np.ndarray, lfe_magnitude: float):
    """Lorentzian MARY spectral shape.

    Used for brute force modulated MARY simulations.

    Args:
            mod_signal (np.ndarray): The modulated signal.
            lfe_magnitude (float): The magnitude of your low field effect (G).

    Returns:
            np.ndarray: The modulated MARY signal.
    """
    return 1 / (1 + mod_signal**2) - lfe_magnitude / (0.1 + mod_signal**2)


def modulated_signal(timeconstant: np.ndarray, theta: float, frequency: float):
    """Modulated MARY signal.

    Used for brute force modulated MARY simulations.

    Args:
            timeconstant (np.ndarray): The modulation time constant (s).
            theta (float): The modulation phase (rad).
            frequency (float): The modulation frequency (Hz).

    Returns:
            np.ndarray: The modulated signal.
    """
    return np.cos(frequency * timeconstant * (2 * np.pi) + theta)


def mol_to_plot_arrays(mol):
    """Return labels, elements, coords[N,3], bonds[(i,j)] from an RDKit Mol with a conformer."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=float)
    elements, labels = [], []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
        el = atom.GetSymbol()
        elements.append(el)
        labels.append(f"{el}{i:02d}")
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return labels, elements, coords, bonds


def mT_to_Gauss(mT: float) -> float:
    """Convert millitesla to Gauss.

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: The magnetic flux density converted to Gauss (G).
    """
    return mT * 10


def mT_to_MHz(mT: float) -> float:
    """Convert millitesla to Megahertz.

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: The magnetic flux density converted to Megahertz (MHz).
    """
    return mT * (1e-9 * C.g_e * C.mu_B / C.h)


def mT_to_angular_frequency(mT: float) -> float:
    """Convert millitesla to angular frequency.

    Args:
            mT (float): The magnetic flux density in millitesla (mT).

    Returns:
            float: The magnetic flux density converted to angular frequency (rad/s/T).
    """
    return mT * (C.mu_B / C.hbar * C.g_e / 1e9)


def nmr_chemical_shift_imaginary_modulation(
    freq_hz: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Chemical-shift modulation (imag): sin(2π f t)."""
    return np.sin(2.0 * np.pi * np.multiply.outer(freq_hz, t))


def nmr_chemical_shift_real_modulation(
    freq_hz: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Chemical-shift modulation (real): cos(2π f t)."""
    return np.cos(2.0 * np.pi * np.multiply.outer(freq_hz, t))


def nmr_scalar_coupling_modulation(
    j_hz: np.ndarray, t: np.ndarray, mult_minus_one: np.ndarray
) -> np.ndarray:
    """J-modulation: cos(π J t) ** (mult - 1)."""
    base = np.cos(np.pi * np.multiply.outer(j_hz, t))
    return np.where(mult_minus_one[:, None] > 0, base ** mult_minus_one[:, None], 1.0)


def nmr_t2_relaxation(t: np.ndarray, t2: float) -> np.ndarray:
    """T2 relaxation decay: exp(-t / T2)."""
    return np.exp(-t / t2)


def parse_pdb(path, use_rdkit_bonds=False, label_scheme="atom"):
    """Load a PDB and return plotting arrays (labels, elements, coords, bonds).

    Builds coordinates from an existing conformer or embeds one if missing.
    Optionally uses RDKit’s bond topology.

    Args:
            path (str | Path): Path to a ``.pdb`` file.
            use_rdkit_bonds (bool): If True, bonds are taken from RDKit
                connectivity; otherwise an empty list is returned.
            label_scheme (str): One of ``{"atom","res_atom","atom_serial","chain_res_atom"}``
                passed to :func:`pdb_label`.

    Returns:
            (list[str], list[str], np.ndarray, list[tuple[int,int]]):
            - ``labels``: Per-atom labels.
            - ``elements``: Atomic symbols.
            - ``coords``: Cartesian coordinates, shape ``(N, 3)`` (Å).
            - ``bonds``: Pairs of atom indices (optional if ``use_rdkit_bonds=False``).
    """
    mol = Chem.MolFromPDBFile(path, removeHs=False)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=float)
    elements, labels = [], []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
        elements.append(atom.GetSymbol())
        labels.append(pdb_label(atom, scheme=label_scheme))
    bonds = []
    if use_rdkit_bonds:
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return labels, elements, coords, bonds


def parse_label_xyz_txt(path):
    """Parse a simple labeled XYZ-like text file.

    Expected format per non-comment line: ``<label> <x> <y> <z>`` with
    whitespace separation. Lines starting with ``#`` are ignored.

    Args:
            path (str | Path): Path to the labeled coordinate file.

    Returns:
            (list[str], list[str], np.ndarray):
            - ``labels``: Raw labels from file.
            - ``elements``: Guessed elements from label alphabetic prefix.
            - ``coords``: Cartesian coordinates, shape ``(N, 3)`` (Å).

    Raises:
            ValueError: If a line cannot be parsed into 4 fields.
    """
    labels, coords = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Bad line in {path!r}: {line}")
            labels.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords, dtype=float)
    elements = []
    for lab in labels:
        prefix = "".join(ch for ch in lab if ch.isalpha()) or lab[0]
        elements.append(prefix if prefix in COVALENT_RADII else prefix[0])
    return labels, elements, coords


def parse_xyz(path):
    """Parse a standard XYZ file (single frame).

    The file must contain at least the header line with atom count, a comment
    line, and ``N`` subsequent atom lines.

    Args:
            path (str | Path): Path to the ``.xyz`` file.

    Returns:
            (list[str], list[str], np.ndarray):
            - ``labels``: Auto-generated labels ``<element><index>``.
            - ``elements``: Atomic symbols.
            - ``coords``: Cartesian coordinates, shape ``(N, 3)`` (Å).

    Raises:
            ValueError: If the file is too short or the atom count does not match.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]
    if len(lines) < 3:
        raise ValueError("XYZ file too short.")
    n = int(lines[0].split()[0])
    atom_lines = lines[2 : 2 + n]
    if len(atom_lines) != n:
        raise ValueError(f"Expected {n} atom lines, got {len(atom_lines)}.")
    elements, coords = [], []
    for ln in atom_lines:
        parts = ln.split()
        el, x, y, z = parts[0], *map(float, parts[1:4])
        elements.append(el)
        coords.append([x, y, z])
    coords = np.array(coords, dtype=float)
    labels = [f"{el}{i:02d}" for i, el in enumerate(elements)]
    return labels, elements, coords


def pdb_label(atom, scheme="chain_res_atom"):
    """Create a human-readable atom label from PDB residue info.

    Args:
            atom (rdkit.Chem.Atom): RDKit atom with PDB residue info.
            scheme (str): Labeling scheme:
                - ``"atom"`` → atom name or element symbol.
                - ``"res_atom"`` → ``RESN<resid>:ATOM``.
                - ``"atom_serial"`` → ``ATOM(serial)``.
                - ``"chain_res_atom"`` (default) → ``CHAIN:RESN<resid>:ATOM``.

    Returns:
            str: Formatted label; falls back to element symbol if fields are missing.
    """
    info = atom.GetPDBResidueInfo()
    symbol = atom.GetSymbol()
    if info is None:
        return symbol
    name = (info.GetName() or "").strip()
    resn = (info.GetResidueName() or "").strip()
    resi = info.GetResidueNumber()
    chain = (info.GetChainId() or "").strip()
    serial = info.GetSerialNumber()
    res_tag = f"{resn}{resi}" if resn or resi else ""
    if scheme == "atom":
        return name or symbol
    elif scheme == "res_atom":
        return f"{res_tag}:{name}" if res_tag else (name or symbol)
    elif scheme == "atom_serial":
        return f"{name}({serial})" if name else f"{symbol}({serial})"
    else:  # chain_res_atom
        left = f"{chain}:{res_tag}" if chain and res_tag else (res_tag or chain)
        return f"{left}:{name}" if left else (name or symbol)


def purity(rho):
    """
    Compute the purity of a quantum state.

    The purity is defined as

    .. math::
        P(\rho) = \mathrm{Tr}(\rho^2),

    where :math:`\rho` is a density matrix. It measures the mixedness of
    a quantum state: pure states satisfy :math:`P = 1`, while maximally
    mixed states have :math:`P = 1/d`, where :math:`d` is the Hilbert-space
    dimension.

    Parameters
    ----------
    rho : ndarray of shape (N, N)
        Density matrix of the quantum state. It should be Hermitian and
        normalised (``Tr(rho) = 1``), although the function does not
        explicitly enforce these conditions.

    Returns
    -------
    float
        The purity :math:`P(\rho)`, computed as ``Re(Tr(rho @ rho))``.

    Notes
    -----
    - The result is real for any Hermitian ``rho``; the real part is
      taken to suppress small imaginary numerical noise.
    - Purity values range from ``1/d`` (maximally mixed) to ``1`` (pure).
    """
    return np.real(np.trace(rho @ rho))


def read_orca_hyperfine(
    path: Path | str, version: int = 6
) -> tuple[list[int], list[str], list[np.ndarray]]:
    """Read ORCA output file.

    Args:
        path (Path): The path to the ORCA output file. Both .out and .property.txt are supported.

    Returns:
        tuple[list[int], list[str], list[np.ndarray]]: The list of nucleus indices, isotopes, and HFC matrices. Note that indices starts from 0 (same as ORCA).

    Examples:
       >>> import radicalpy as rp
       >>> indices, isotopes, hfc_matrices = rp.utils.read_orca("orca.out") # or rp.utils.read_orca("orca.property.txt")
       >>> nuclei = [rp.data.Nucleus.fromisotope(isotope, hfc_matrix.tolist()) for isotope, hfc_matrix in zip(isotopes, hfc_matrices)]
       >>> molecule = rp.simulation.Molecule("mol1", nuclei)
       Molecule: mol1
       Nuclei:
       14N(19337792.0, 3, 1.017 <anisotropic available>)
       1H(267522187.44, 2, -2.199 <anisotropic available>)
       1H(267522187.44, 2, -2.198 <anisotropic available>)
       Radical: E(-176085963023.0, 2, 0.0 <anisotropic not available>)

    """
    if isinstance(path, str):
        path = Path(path)
    if version == 6:
        if path.name.endswith(".property.txt"):
            indices, isotopes, hfc_matrices = _hyperfine_from_orca6_property_txt(path)
        else:
            indices, isotopes, hfc_matrices = _hyperfine_from_orca6_out(path)
    else:
        raise NotImplementedError(f"Version {version} is not supported")

    for index, isotope, hfc_matrix in zip(indices, isotopes, hfc_matrices, strict=True):
        print(
            f"Nucleus {index} (starts from 0) isotope {isotope} HFC matrix [mT]:\n {hfc_matrix}\n"
        )

    return indices, isotopes, hfc_matrices


def read_lines_utf8_or_utf16le(path: Path) -> list[str]:
    """Read text file as UTF-8 (with BOM) or UTF-16LE, returning lines.

    Tries UTF-8 (BOM-aware) first, then UTF-16LE, which is common in
    ORCA outputs on Windows.

    Args:
            path (Path): File path.

    Returns:
            list[str]: Lines of the decoded file (including newlines).

    Raises:
            UnicodeError: If the file is neither valid UTF-8 nor UTF-16LE.
    """
    # Try UTF-8 first (with BOM support via -sig)
    # ORCA in Windows may have utf-16le
    for enc in ("utf-8-sig", "utf-16le"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeError("File is neither valid UTF-8 nor UTF-16LE.")


def _hyperfine_from_orca6_out(
    path: Path,
) -> tuple[list[int], list[str], list[np.ndarray]]:
    """Parse hyperfine data from an ORCA 6 ``.out`` file.

    Extracts nucleus indices, isotopes, and total hyperfine coupling
    (HFC) matrices (in mT) from the EPR/HFC sections.

    Args:
            path (Path): Path to the ORCA ``.out`` file.

    Returns:
            tuple[list[int], list[str], list[np.ndarray]]:
            - ``indices``: Zero-based nucleus indices (as in ORCA).
            - ``isotopes``: Isotope strings, e.g. ``"14N"``.
            - ``hfc_matrices``: 3×3 HFC matrices (mT) per nucleus.

    Raises:
            ValueError: If the number of parsed nuclei, isotopes, or
                matrices is inconsistent with the header.
    """
    lines = read_lines_utf8_or_utf16le(path)

    is_epr_block = False
    is_hfc_matrix_block = False
    n_nuclei = -1
    indices = []
    isotopes = []
    hfc_matrices = []
    # Extract 3 from "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE (3 nuclei)"
    NUCLEI_REGEX = re.compile(
        r"""\(          # opening parenthesis
            \s*         # optional spaces
            (\d+)       # ← capture the integer
            \s*         # optional spaces
            nucle(?:i|us)  # 'nuclei' or 'nucleus', case‑insensitive
            \s*         # optional spaces
            \)          # closing parenthesis
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    # Extract 0 from "Nucleus   0N"
    INDEX_REGEX = re.compile(
        r"Nucleus\s+(\d+)\w+",
        flags=re.IGNORECASE,
    )
    # Extract N from  "Nucleus   0N"
    ELEM_REGEX = re.compile(
        r"Nucleus\s+\d+(\w+)",
        flags=re.IGNORECASE | re.VERBOSE,
    )
    ISOTOPE_REGEX = re.compile(
        r"Isotope=(\s*\d+)",
        flags=re.IGNORECASE | re.VERBOSE,
    )
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE"):
            is_epr_block = True
            # line is like "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE (3 nuclei)"
            # we need to extract the number of nuclei by re.search
            n_nuclei = int(NUCLEI_REGEX.search(line).group(1))
            i += 1
            continue
        if not is_epr_block:
            i += 1
            continue
        if line.startswith(" Nucleus"):
            # line is like "Nucleus   0N : A  : Isotope=   14 I=  1.0 P= 38.5677 MHz/au**3"
            # we need to extract the nucleus name, isotope.
            split_items = line.split(":")
            # split_items[0] is "Nucleus   0N"
            indices.append(int(INDEX_REGEX.search(split_items[0]).group(1)))
            element = ELEM_REGEX.search(split_items[0]).group(1)
            # split_items[1] is "A"
            if split_items[1].strip() != "A":
                raise ValueError(f"Unsupported format: {line}")
            # split_items[2] is "Isotope=   14 I=  1.0 P= 38.5677 MHz/au**3"
            isotope = int(ISOTOPE_REGEX.search(split_items[2]).group(1))
            isotopes.append(f"{isotope}{element}")
        elif line.startswith(" Total HFC matrix (all values in MHz)"):
            is_hfc_matrix_block = True
            i += 2
            continue
        if is_hfc_matrix_block:
            AxxAxyAxz = [MHz_to_mT(float(x)) for x in line.split()]
            i += 1
            AyxAyyAyz = [MHz_to_mT(float(x)) for x in lines[i].split()]
            i += 1
            AzxAzyAzz = [MHz_to_mT(float(x)) for x in lines[i].split()]
            i += 1
            A = np.array([AxxAxyAxz, AyxAyyAyz, AzxAzyAzz])
            # When SOC is included, A is not always symmetric.
            # np.testing.assert_allclose(A, A.T, atol=1e-6)
            hfc_matrices.append(A)
            is_hfc_matrix_block = False
            if len(hfc_matrices) == n_nuclei:
                break
        i += 1
    if len(indices) != n_nuclei:
        raise ValueError(f"Number of nuclei mismatch: {len(indices)} != {n_nuclei}")
    if len(isotopes) != n_nuclei:
        raise ValueError(f"Number of isotopes mismatch: {len(isotopes)} != {n_nuclei}")
    if len(hfc_matrices) != n_nuclei:
        raise ValueError(
            f"Number of HFC matrices mismatch: {len(hfc_matrices)} != {n_nuclei}"
        )
    return indices, isotopes, hfc_matrices


def _hyperfine_from_orca6_property_txt(
    path: Path,
) -> tuple[list[int], list[str], list[np.ndarray]]:
    """Parse hyperfine data from an ORCA 6 ``.property.txt`` file.

    Reads the ``$SCF_A_Tensor`` block to obtain per-nucleus indices,
    isotopes, and anisotropic A-tensors, converted to mT.

    Args:
            path (Path): Path to the ORCA ``.property.txt`` file.

    Returns:
            tuple[list[int], list[str], list[np.ndarray]]:
            - ``indices``: Zero-based nucleus indices.
            - ``isotopes``: Isotope strings, e.g. ``"14N"``.
            - ``hfc_matrices``: 3×3 HFC matrices (mT) per nucleus.

    Raises:
            ValueError: If counts of nuclei/isotopes/matrices are inconsistent.
    """
    lines = read_lines_utf8_or_utf16le(path)
    is_scf_a_tensor_block = False
    indices = []
    isotopes = []
    hfc_matrices = []
    # Extract 3 from  "   &NUMOFNUCS [&Type "Integer"] 3 "Number of active nuclei"
    NUCLEI_REGEX = re.compile(
        r"""
        &NUMOFNUCS\s*\[&Type\s*"Integer"\]\s*(\d+)\s*"Number\s*of\s*active\s*nucle(?:i|us)"
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    # Extract 0 from "   &NUC [&Type "Integer"] 0 "Index of the nuclei"
    NUCLEI_INDEX_REGEX = re.compile(
        r"""
        &NUC\s*\[&Type\s*"Integer"\]\s*(\d+)\s*"Index\s*of\s*the\s*nuclei"
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    # Extract 7 from "   &ELEM [&Type "Integer"] 7 "Atomic number of the nuclei"
    ELEM_REGEX = re.compile(
        r"""
        &ELEM\s*\[&Type\s*"Integer"\]\s*(\d+)\s*"Atomic\s*number\s*of\s*the\s*nucle(?:i|us)"
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    # Extract 14 from "   &ISOTOPE [&Type "Double"]       1.4000000000000000e+01  "Atomic mass"
    ISOTOPE_REGEX = re.compile(
        r"""
        &ISOTOPE\s*\[&Type\s*"Double"\]\s*([0-9.e+-]+)\s*"Atomic\s*mass"
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    # This dictionary may be better to be a shared constant.
    elem_dict = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
        87: "Fr",
        88: "Ra",
        89: "Ac",
        90: "Th",
        91: "Pa",
        92: "U",
        93: "Np",
        94: "Pu",
        95: "Am",
        96: "Cm",
        97: "Bk",
        98: "Cf",
        99: "Es",
        100: "Fm",
        101: "Md",
        102: "No",
        103: "Lr",
        104: "Rf",
        105: "Db",
        106: "Sg",
        107: "Bh",
        108: "Hs",
        109: "Mt",
        110: "Ds",
        111: "Rg",
        112: "Cn",
        113: "Nh",
        114: "Fl",
        115: "Mc",
        116: "Lv",
        117: "Ts",
        118: "Og",
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("$SCF_A_Tensor"):
            is_scf_a_tensor_block = True
            i += 1
            continue
        if not is_scf_a_tensor_block:
            i += 1
            continue
        if is_scf_a_tensor_block and line.startswith("$End"):
            is_scf_a_tensor_block = False
            break

        if line.strip().startswith('&NUMOFNUCS [&Type "Integer"]'):
            # line is like    &NUMOFNUCS [&Type "Integer"] 3 "Number of active nuclei"
            n_nuclei = int(NUCLEI_REGEX.search(line).group(1))
            i += 1
            continue

        if line.strip().startswith('&NUC [&Type "Integer"]'):
            # line is like    &NUC [&Type "Integer"] 0 "Index of the nuclei"
            nucleus_index = int(NUCLEI_INDEX_REGEX.search(line).group(1))
            indices.append(nucleus_index)
            i += 1
            continue

        if line.strip().startswith('&ELEM [&Type "Integer"]'):
            # line is like    &ELEM [&Type "Integer"] 7 "Atomic number of the nuclei"
            element = int(ELEM_REGEX.search(line).group(1))
            i += 1
            continue

        if line.strip().startswith('&ISOTOPE [&Type "Double"]'):
            #    &ISOTOPE [&Type "Double"]       1.4000000000000000e+01  "Atomic mass"
            isotope = int(float(ISOTOPE_REGEX.search(line).group(1)))
            isotopes.append(f"{isotope}{elem_dict[element]}")
            i += 1
            del nucleus_index
            del element
            del isotope
            continue

        if line.strip().startswith("&ARAW"):
            i += 3
            AxxAxyAxz = [MHz_to_mT(float(x)) for x in lines[i].split()[1:]]
            i += 1
            AyxAyyAyz = [MHz_to_mT(float(x)) for x in lines[i].split()[1:]]
            i += 1
            AzxAzyAzz = [MHz_to_mT(float(x)) for x in lines[i].split()[1:]]
            A = np.array([AxxAxyAxz, AyxAyyAyz, AzxAzyAzz])
            hfc_matrices.append(A)
            i += 1
            continue
        i += 1

    if len(indices) != n_nuclei:
        raise ValueError(f"Number of nuclei mismatch: {len(indices)} != {n_nuclei}")
    if len(isotopes) != n_nuclei:
        raise ValueError(f"Number of isotopes mismatch: {len(isotopes)} != {n_nuclei}")
    if len(hfc_matrices) != n_nuclei:
        raise ValueError(
            f"Number of HFC matrices mismatch: {len(hfc_matrices)} != {n_nuclei}"
        )
    return indices, isotopes, hfc_matrices


def read_trajectory_files(path: Path, scale=1e-10):
    """Load and concatenate text trajectory files in a directory.

    Each file under ``path`` is read with ``numpy.genfromtxt`` and
    concatenated along the first axis; the result is multiplied by
    ``scale`` (default converts Å → m if files are in Å).

    Args:
            path (Path): Directory containing trajectory fragments.
            scale (float): Multiplicative scaling factor applied post-concat.

    Returns:
            np.ndarray: Concatenated (and scaled) trajectory array.
    """
    data = [np.genfromtxt(file_path) for file_path in Path(path).glob("*")]
    return scale * np.concatenate(data, axis=0)


def reference_signal(
    timeconstant: np.ndarray, harmonic: float, theta: float, frequency: float
):
    """Modulated MARY reference signal.

    Used for brute force modulated MARY simulations.

    Args:
            timeconstant (np.ndarray): The modulation time constant (s).
            harmonic (float): The harmonic of the modulation.
            theta (float): The modulation phase (rad).
            frequency (float): The modulation frequency (Hz).

    Returns:
            np.ndarray: The modulated reference signal.
    """
    return np.cos(harmonic * frequency * timeconstant * (2 * np.pi) + harmonic * theta)


def rodrigues_rotation(v, k, theta):
    """Rotate vector(s) ``v`` about axis ``k`` by angle ``theta`` (Rodrigues’ formula).

    Supports arrays of stacked row vectors ``v`` with shape ``(N, 3)`` or
    column-major layout ``(3, N)``. The axis ``k`` is normalised internally.

    Args:
            v (np.ndarray): Vectors to rotate, shape ``(N, 3)`` or ``(3, N)``.
            k (array-like): Rotation axis (length-3).
            theta (float): Rotation angle (rad).

    Returns:
            np.ndarray: Rotated vectors with the same shape as ``v``.
    """
    m, n = v.shape

    # Normalise rotation axis
    k = k / np.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)
    No = np.size(v) // 3  # Number of vectors in array
    v_rot = np.copy(v)  # Initialise rotated vector array

    if n == 3:
        crosskv = np.zeros(3)  # Initialise cross product k and v with right dimensions
        for i in range(No):
            crosskv[0] = k[1] * v[i, 2] - k[2] * v[i, 1]
            crosskv[1] = k[2] * v[i, 0] - k[0] * v[i, 2]
            crosskv[2] = k[0] * v[i, 1] - k[1] * v[i, 0]
            v_rot[i, :] = (
                np.cos(theta) * v[i, :]
                + crosskv * np.sin(theta)
                + k * np.dot(k, v[i, :]) * (1 - np.cos(theta))
            )
    else:  # if m == 3 and n != 3
        crosskv = np.zeros(3)  # Initialise cross product k and v with right dimensions
        for i in range(No):
            crosskv[0] = k[1] * v[2, i] - k[2] * v[1, i]
            crosskv[1] = k[2] * v[0, i] - k[0] * v[2, i]
            crosskv[2] = k[0] * v[1, i] - k[1] * v[0, i]
            v_rot[:, i] = (
                np.cos(theta) * v[:, i]
                + crosskv * np.sin(theta)
                + k * np.dot(k, v[:, i]) * (1 - np.cos(theta))
            )

    return v_rot


def rotate_axes(A, x, y, z):
    """Apply rotation matrix ``A`` to a basis ``(x, y, z)``.

    Args:
            A (np.ndarray): 3×3 rotation matrix.
            x, y, z (np.ndarray): Input basis unit vectors (length-3).

    Returns:
            (np.ndarray, np.ndarray, np.ndarray): Rotated basis vectors
            ``(x', y', z')``.
    """
    x = np.dot(A, x)
    y = np.dot(A, y)
    z = np.dot(A, z)
    return x, y, z


def smiles_to_3d(smiles: str, add_h: bool = True, opt: str = "mmff"):
    """Create a 3D-embedded RDKit Mol from SMILES (ETKDG + optional MMFF/UFF)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_h:
        mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Retry with more attempts
        params.maxAttempts = 2000
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            return None
    if opt.lower() == "mmff":
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    elif opt.lower() == "uff":
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass
    return mol


def spectral_density(omega: float, tau_c: float) -> float:
    """Frequency at which the motion of the particle exists.

    Args:
            omega (float): The Larmor frequency of the electron (1/s).
            tau_c (float): The rotational correlation time (s).

    Returns:
            float: Spectral density frequency (1/s).
    """
    return tau_c / (1 + omega**2 * tau_c**2)


def _check_full_sphere(theta: np.ndarray, phi: np.ndarray) -> Tuple[int, int]:
    """Verify that ``theta``/``phi`` form a full-sphere tensor grid.

    Checks that ``theta`` equals ``linspace(0, π, len(theta))`` and
    ``phi`` equals ``linspace(0, 2π, len(phi))`` (within numerical tolerance).

    Args:
            theta (np.ndarray): Polar grid (radians).
            phi (np.ndarray): Azimuthal grid (radians).

    Returns:
            (int, int): ``(len(theta), len(phi))`` for convenience.

    Raises:
            ValueError: If either grid does not match the expected full-sphere linspace.
    """
    nth, nph = len(theta), len(phi)
    if not np.all(np.isclose(theta, np.linspace(0, np.pi, nth))):
        raise ValueError(
            "Not a full sphere: `theta` should be `linspace(0, np.pi, ntheta)`"
        )
    if not np.all(np.isclose(phi, np.linspace(0, 2 * np.pi, nph))):
        raise ValueError(
            "Not a full sphere: `phi` should be `linspace(0, np.pi, nphi)`"
        )
    return nth, nph


def spherical_average(
    product_yield: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> float:
    """Spherical average of anisotropic product yields.

    Args:
            product_yield (np.ndarray): The anisotropic product
                yields.
            theta (np.ndarray): The angles theta by which the
                anisotropic product yields were calculated.
            phi (np.ndarray): The angles phi by which the anisotropic
                product yields were calculated.

    Returns:
            float: The spherical average of the anisotropic product
                yields.
    """
    theta, phi = anisotropy_check(theta, phi)
    nth, nph = _check_full_sphere(theta, phi)

    wt = 4 * np.ones(nth)
    wt[2:-2:2] = 2
    wt[0] = wt[-1] = 1

    wp = 4 * np.ones(nph)
    wp[0:-1:2] = 2
    sintheta = np.sin(np.linspace(0, np.pi, nth))

    spherical_avg = sum(
        product_yield[i, j] * sintheta[i] * wt[i] * wp[j]
        for i in range(nth)
        for j in range(nph)
    )

    return spherical_avg * theta[1] * phi[1] / (4 * np.pi) / 9


def spherical_to_cartesian(
    theta: float | np.ndarray, phi: float | np.ndarray
) -> np.ndarray:
    """Spherical coordinates to Cartesian coordinates.

    Args:
            theta (float or np.ndarray): The polar angle(s).
            phi (float or np.ndarray): The azimuthal angle(s).

    Returns:
            np.ndarray: The Cartesian coordinates.
    """
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    ).T


def s_t0_omega(
    deltag: float, B0: float, hfc_star: float, onuc_all: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the two radical pair frequencies ω+ and ω- (in rad/s) for S-T0 mixing.

    Parameters
    ----------
    deltag : float
        Difference in g-factors of the two radicals.
    b0 : float
        External magnetic field strength (Tesla).
    hfc_star : float
        Hyperfine coupling constant (rad/s) of the nucleus of interest.
    onuc_all : np.ndarray
        Array of total hyperfine contributions from all other nuclei (rad/s).

    Returns
    -------
    omega_plus : float
        The ω+ frequency (rad/s).
    omega_minus : float
        The ω- frequency (rad/s).
    """
    base_omega = (deltag * C.mu_B * B0) / C.hbar  # Δg μB B0 / ħ
    omega_plus = base_omega + 0.5 * hfc_star + onuc_all
    omega_minus = base_omega - 0.5 * hfc_star + onuc_all
    return omega_plus, omega_minus


def von_neumann_entropy(rho):
    """
    Compute the von Neumann entropy of a quantum state.

    The von Neumann entropy is defined as
    :math:`S(\\rho) = -\\mathrm{Tr}\\,(\\rho\\,\\log \\rho)`,
    which, in the eigenbasis of :math:`\\rho`, reduces to
    :math:`S = -\\sum_i \\lambda_i \\log \\lambda_i`,
    where :math:`\\{\\lambda_i\\}` are the eigenvalues of :math:`\\rho`.
    This implementation uses the natural logarithm, so the entropy is
    returned in **nats**.

    Parameters
    ----------
    rho : ndarray of shape (N, N)
        Density matrix. For physical states, ``rho`` should be Hermitian,
        positive semidefinite, and trace-normalised (``Tr(rho)=1``), though
        the function does not explicitly enforce these constraints.

    Returns
    -------
    float
        The von Neumann entropy :math:`S(\\rho)` in nats.

    Notes
    -----
    - Eigenvalues that are numerically non-positive are **ignored** in the
      sum to avoid evaluating ``log(0)``.
    - To obtain entropy in **bits**, divide the returned value by ``np.log(2)``.
    """
    evals, evecs = np.linalg.eig(rho)
    ids = evals.argsort()
    evals = evals[ids]
    evecs = evecs[:, ids]
    return -np.real(np.sum([val * np.log(val) for val in evals if val > 0]))


def write_pdb(mol, path):
    """Write the current conformer to a PDB file.

    Args:
            mol (rdkit.Chem.Mol): Molecule to serialise.
            path (str | Path): Destination ``.pdb`` filepath.

    Side effects:
            Prints a confirmation line with the saved path.
    """
    block = Chem.MolToPDBBlock(mol)
    with open(path, "w", encoding="utf-8") as f:
        f.write(block)
    print(f"Saved PDB to {path}")


def write_sdf(mol, path):
    """Write the molecule’s current conformer to an SDF file (one record).

    Saves the active conformer of an RDKit `Mol` as a single SDF record,
    preserving topology (bonds, charges, stereochemistry) and 3D coordinates.

    Args:
        mol: RDKit `Chem.Mol` instance containing the conformer to export.
        path: Destination file path for the `.sdf` file.

    Notes:
        - Creates the file or overwrites it if it already exists.
        - Prints a short confirmation message on success.
    """
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.flush()
    writer.close()
    print(f"Saved SDF to {path}")


def write_xyz(mol, path):
    """Write the current conformer to an XYZ file (single frame).

    Args:
            mol (rdkit.Chem.Mol): Molecule to serialise.
            path (str | Path): Destination ``.xyz`` filepath.

    Notes:
            Writes a minimal XYZ with a generated comment line.
    """
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\nGenerated by plot_smiles.py\n")
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    print(f"Saved XYZ to {path}")


def yield_anisotropy(
    product_yield: np.ndarray, theta: np.ndarray, phi: np.ndarray
) -> Tuple[float, float]:
    """Calculate the yield anisotropy.

    Args:
            product_yield (np.ndarray): The anisotropic product yields.
            theta (np.ndarray): The angles theta by which the
                anisotropic product yields were calculated.
            phi (np.ndarray): The angles phi by which the anisotropic
                product yields were calculated.

    Returns:
            (float, float):
            - delta_phi (float): Maximum yield - minimum yield.
            - gamma (float): delta_phi / spherical average.
    """
    delta_phi = product_yield.max() - product_yield.min()
    yield_av = spherical_average(product_yield, theta, phi)
    gamma = delta_phi / yield_av
    return delta_phi, gamma
