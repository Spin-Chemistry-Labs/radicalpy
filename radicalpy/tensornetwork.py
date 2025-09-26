"""Tensor network simulations for radical pair dynamics.

This module provides tensor network-based simulations for radical pair
dynamics.

The simulations support quantum mechanical treatments of nuclear spins, allowing
for accurate modeling of hyperfine interactions, exchange coupling, and dipolar
interactions in radical pairs.

Example:
    Basic usage examples are available in:

    - :file:`examples/lpmps.py` - Linear product MPS simulations
    - :file:`examples/smps.py` - Stochastic MPS simulations
    - :file:`examples/vmpdo.py` - Vectorised MPDO simulations

Note:
    Install the tensor network simulation library:

    .. code-block:: bash

        pip install git+https://github.com/QCLovers/PyTDSCF

References:
    - `PyTDSCF <https://github.com/QCLovers/PyTDSCF>`_
    - `JAX installation <https://docs.jax.dev/en/latest/installation.html>`_
"""

import concurrent.futures
import multiprocessing as mp
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from math import isqrt
from typing import Any, Literal, Optional

import numpy as np
from scipy.linalg import expm
from tqdm.auto import tqdm

from . import utils
from .data import Molecule
from .simulation import Basis, HilbertSimulation, State

try:
    import netCDF4 as nc
    from pytdscf import BasInfo, Exciton, Model, Simulator, units
    from pytdscf.dvr_operator_cls import TensorOperator
    from pytdscf.hamiltonian_cls import TensorHamiltonian
    from pytdscf.util import read_nc

    IS_PYTDSCF_AVAILABLE = True
except ModuleNotFoundError:
    IS_PYTDSCF_AVAILABLE = False

try:
    from pympo import (
        AssignManager,
        OpSite,
        SumOfProducts,
        get_eye_site,
    )
    from sympy import Symbol

    IS_PYMPO_AVAILABLE = True
except ModuleNotFoundError:
    IS_PYMPO_AVAILABLE = False
    SumOfProducts = Any  # type: ignore[assignment]

MSG_PYTDSCF_NOT_INSTALLED = """
pytdscf is not installed. 
Please install it with `pip install git+https://github.com/QCLovers/PyTDSCF`.
For more information, see https://github.com/QCLovers/PyTDSCF.
For GPU support, see https://docs.jax.dev/en/latest/installation.html.
"""

MSG_PYMPO_NOT_INSTALLED = """
pympo is not installed. 
Please install it with `pip install git+https://github.com/KenHino/pympo`.
For more information, see https://github.com/KenHino/pympo.
Cargo, package manager for Rust, is required to install pympo.
"""

SCALE = 1e-09


def _get_vecB(
    B0: float,
    B_axis: str = "z",
    theta: Optional[float] = None,
    phi: Optional[float] = None,
) -> np.ndarray:
    if theta is None and phi is None:
        match B_axis:
            case "x":
                B = np.array([B0, 0, 0])
            case "y":
                B = np.array([0, B0, 0])
            case "z":
                B = np.array([0, 0, B0])
            case _:
                raise ValueError(f"Invalid axis: {B_axis}")
    else:
        B = utils.spherical_to_cartesian(theta, phi) * B0
    return B


class BaseMPSSimulation(HilbertSimulation, ABC):
    """
    Abstract base class for matrix product state simulations.
    See following class:

    - :class:`StochasticMPSSimulation`
    - :class:`LocallyPurifiedMPSSimulation`
    - :class:`VectorisedMPDOSimulation`.

    """

    def __init__(
        self,
        molecules: list[Molecule],
        custom_gfactors: bool = False,
        basis: Basis = Basis.ST,
        bond_dimension: int = 16,
        backend: Literal["numpy", "jax", "auto"] = "numpy",
        integrator: Literal["arnoldi", "lanczos"] = "arnoldi",
        jobname: str = "tensornetwork",
    ):
        if not IS_PYTDSCF_AVAILABLE:
            raise ModuleNotFoundError(MSG_PYTDSCF_NOT_INSTALLED)
        if not IS_PYMPO_AVAILABLE:
            raise ModuleNotFoundError(MSG_PYMPO_NOT_INSTALLED)

        if backend == "auto":
            if bond_dimension > 32:
                backend = "jax"
            else:
                backend = "numpy"

        if basis != Basis.ST:
            raise NotImplementedError(
                "Only ST basis is supported for tensor network method."
            )
        super().__init__(molecules, custom_gfactors, basis)
        # To reduce entanglement, sort nuclei by the absolute value of the HFC
        self._sort_nuclei()
        self.subs, self.g_ele_sym, self.g_nuc_sym = self._get_gyromagnetic_subs()
        self.bond_dimension = bond_dimension
        self.backend = backend
        self.integrator = integrator
        self.jobname = jobname

    def _sort_nuclei(self):
        hf_abs = []
        nucs = self.molecules[0].nuclei
        if len(nucs) > 0:
            for nuc in nucs:
                if nuc.hfc._anisotropic is None:
                    hf_abs.append(abs(nuc.hfc.isotropic))
                else:
                    eigvals = np.linalg.eigvals(nuc.hfc.anisotropic)
                    hf_abs.append(np.mean(np.abs(eigvals)))
            nuclei = [nucs[i] for i in np.argsort(hf_abs)]
            self.molecules[0].nuclei = nuclei
        hf_abs = []
        nucs = self.molecules[1].nuclei
        if len(nucs) > 0:
            for nuc in nucs:
                if nuc.hfc._anisotropic is None:
                    hf_abs.append(abs(nuc.hfc.isotropic))
                else:
                    eigvals = np.linalg.eigvals(nuc.hfc.anisotropic)
                    hf_abs.append(np.mean(np.abs(eigvals)))
            print(nucs)
            nuclei = [nucs[i] for i in np.argsort(hf_abs)]
            self.molecules[1].nuclei = nuclei

    def _get_gyromagnetic_subs(self):
        g_ele_sym = [
            Symbol(r"\gamma_e^{(" + f"{i + 1}" + ")}")
            for i in range(len(self.radicals))
        ]
        g_nuc_sym = {}
        for i in range(len(self.radicals)):
            for j in range(len(self.molecules[i].nuclei)):
                g_nuc_sym[(i, j)] = Symbol(r"\gamma_n^{" + f"{(i + 1, j + 1)}" + "}")

        subs = {}
        for i, ge in enumerate(g_ele_sym):
            subs[ge] = self.radicals[i].gamma_mT
        for (i, j), gn in g_nuc_sym.items():
            subs[gn] = self.molecules[i].nuclei[j].gamma_mT
        return subs, g_ele_sym, g_nuc_sym

    def _get_electron_ops(self):
        sx_1 = self.spin_operator(0, "x", kron_eye=False)
        sy_1 = self.spin_operator(0, "y", kron_eye=False)
        sz_1 = self.spin_operator(0, "z", kron_eye=False)
        sx_2 = self.spin_operator(1, "x", kron_eye=False)
        sy_2 = self.spin_operator(1, "y", kron_eye=False)
        sz_2 = self.spin_operator(1, "z", kron_eye=False)

        Qs = self.projection_operator(State.SINGLET, kron_eye=False)
        Qt = self.projection_operator(State.TRIPLET, kron_eye=False)
        return sx_1, sy_1, sz_1, sx_2, sy_2, sz_2, Qs, Qt

    def zeeman_hamiltonian(
        self,
        B0: float,
        B_axis: str = "z",
        theta: Optional[float] = None,
        phi: Optional[float] = None,
    ) -> SumOfProducts:
        """
        Construct the Zeeman Hamiltonian as a SumOfProducts.

        Args:
           B0 (float): Magnetic field strength in mT.
           B_axis (str): Axis of the magnetic field ('x', 'y', or 'z'). Default is 'z'.
           theta (float, optional): Polar angle of the magnetic field in radians.
           phi (float, optional): Azimuthal angle of the magnetic field in radians.

        """
        zeeman = SumOfProducts()
        xyz = "xyz"
        B = _get_vecB(B0, B_axis, theta, phi)
        for a, (Sr_op, Ir_op) in enumerate(zip(self.Sr_ops, self.Ir_ops, strict=True)):
            if B[a] == 0.0:
                continue
            r = xyz[a]
            Br = Symbol(f"B_{r}")
            self.subs[Br] = B[a] * SCALE
            for i in range(len(self.radicals)):
                zeeman += -Br * self.g_ele_sym[i] * Sr_op[i]
                for j in range(len(self.molecules[i].nuclei)):
                    zeeman += -Br * self.g_nuc_sym[(i, j)] * Ir_op[(i, j)]
        zeeman = zeeman.simplify()
        return zeeman

    def hyperfine_hamiltonian(self, hfc_anisotropy: bool = False) -> SumOfProducts:
        """
        Construct the hyperfine Hamiltonian as a SumOfProducts.

        Args:
           hfc_anisotropy (bool): Whether to include anisotropic hyperfine coupling. Default is False.

        """
        hyperfine = SumOfProducts()
        xyz = "xyz"
        for i in range(len(self.radicals)):
            for j, nuc in enumerate(self.molecules[i].nuclei):
                if hfc_anisotropy:
                    A_ij = nuc.hfc.anisotropic
                else:
                    A_ij = np.eye(3) * nuc.hfc.isotropic
                for a, Sr_op in enumerate(self.Sr_ops):
                    for b, Ir_op in enumerate(self.Ir_ops):
                        if A_ij[a, b] == 0.0:
                            continue
                        Asym = Symbol(
                            "A^{"
                            + f"{(i + 1, j + 1)}"
                            + "}_{"
                            + f"{xyz[a]}"
                            + f"{xyz[b]}"
                            + "}"
                        )
                        self.subs[Asym] = A_ij[a, b].item() * SCALE
                        hyperfine += (
                            Asym * abs(self.g_ele_sym[0]) * Sr_op[i] * Ir_op[(i, j)]
                        )
        hyperfine = hyperfine.simplify()
        return hyperfine

    def exchange_hamiltonian(self, J: float, prod_coeff: float = 2) -> SumOfProducts:
        """
        Construct the exchange Hamiltonian as a SumOfProducts.

        Args:
           J (float): Exchange coupling in mT.
           prod_coeff (float): Coefficient for S1.S2 term. Default is 2 for -2J S1.S2 convention.

        """
        exchange = SumOfProducts()
        Jsym = Symbol("J")
        self.subs[Jsym] = J * SCALE
        exchange += (
            -Jsym * abs(self.g_ele_sym[0]) * (2 * self.S1S2_op + 0.5 * self.E_op)
        )
        exchange = exchange.simplify()
        return exchange

    def dipolar_hamiltonian(self, D: float | np.ndarray) -> SumOfProducts:
        """
        Construct the dipolar Hamiltonian as a SumOfProducts.

        Args:
           D (float | np.ndarray): Dipolar coupling in mT.

        """
        if isinstance(D, float):
            if D > 0.0:
                print(
                    f"WARNING: D is {D} mT, which is positive. In point dipole approximation, D should be negative."
                )
            D = 2 / 3 * np.diag((-1.0, -1.0, 2.0)) * D
        # Define Dipolar Hamiltonian
        dipolar = SumOfProducts()
        xyz = "xyz"
        for a in range(3):
            for b in range(3):
                if D[a, b] == 0.0:
                    continue
                else:
                    Dsym = Symbol("D_{" + f"{xyz[a]}" + f"{xyz[b]}" + "}")
                    self.subs[Dsym] = D[a, b] * SCALE
                    dipolar += (
                        Dsym
                        * abs(self.g_ele_sym[0])
                        * self.Sr_ops[a][0]
                        * self.Sr_ops[b][1]
                    )
        dipolar = dipolar.simplify()
        return dipolar

    def haberkorn_hamiltonian(self, kS: float, kT: float) -> SumOfProducts:
        """
        Construct the Haberkorn recombination term as a SumOfProducts.

        Args:
           kS (float): Singlet decay rate in s-1.
           kT (float): Triplet decay rate in s-1.

        """
        haberkorn = SumOfProducts()
        if kS != 0.0:
            kSsym = Symbol("k_S")
            self.subs[kSsym] = kS * SCALE
            haberkorn -= 1.0j * kSsym / 2 * self.Qs_op
        if kT != 0.0:
            kTsym = Symbol("k_T")
            self.subs[kTsym] = kT * SCALE
            haberkorn -= 1.0j * kTsym / 2 * self.Qt_op
        haberkorn = haberkorn.simplify()
        return haberkorn

    def zero_field_splitting_hamiltonian(
        self, D, E, kron_eye: bool = True
    ) -> np.ndarray:
        """
        Zero-field splitting Hamiltonian is not implemented.
        """
        raise NotImplementedError("Zero-field splitting Hamiltonian is not implemented")

    def total_hamiltonian(
        self,
        B0: float,
        J: float,
        D: float | np.ndarray,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        hfc_anisotropy: bool = False,
        kS: float = 0.0,
        kT: float = 0.0,
    ) -> list[np.ndarray]:
        """
        Construct the total Hamiltonian as a matrix product operator.

        Args:
           B0 (float): Magnetic field strength in mT.
           J (float): Exchange coupling in mT. Employing -2J S1.S2 convention.
           D (float | np.ndarray): Dipolar coupling in mT.
           theta (float, optional): Polar angle of the magnetic field in radians.
           phi (float, optional): Azimuthal angle of the magnetic field in radians.
           hfc_anisotropy (bool): Whether to include anisotropic hyperfine coupling. Default is False.
           kS (float): Singlet recombination rate in s-1 for Haberkorn term. Default is 0.0.
           kT (float): Triplet recombination rate in s-1 for Haberkorn term. Default is 0.0.

        Returns:
           list[np.ndarray]: Total Hamiltonian as a matrix product operator.

        """
        zeeman = self.zeeman_hamiltonian(B0=B0, theta=theta, phi=phi)
        hyperfine = self.hyperfine_hamiltonian(hfc_anisotropy)
        exchange = self.exchange_hamiltonian(J)
        dipolar = self.dipolar_hamiltonian(D)
        haberkorn = self.haberkorn_hamiltonian(kS, kT)
        total = zeeman + hyperfine + exchange + dipolar + haberkorn
        total = total.simplify()
        am = AssignManager(total)
        _ = am.assign()
        mpo = am.numerical_mpo(subs=self.subs)
        return mpo

    @abstractmethod
    def _get_basis(self):
        pass

    @abstractmethod
    def time_evolution(
        self, init_state: State, time: np.ndarray, H: list[np.ndarray]
    ) -> np.ndarray:
        pass

    def product_probability(self, obs: State, rhos: np.ndarray) -> np.ndarray:
        if obs == State.EQUILIBRIUM:
            raise ValueError("Observable state should not be EQUILIBRIUM")
        Q = self.observable_projection_operator(obs)
        if rhos.ndim == 2:
            if obs not in [
                State.SINGLET,
                State.TRIPLET,
                State.TRIPLET_PLUS,
                State.TRIPLET_ZERO,
                State.TRIPLET_MINUS,
            ]:
                raise NotImplementedError(
                    f"Observable state {obs} for diagonal matrix is not implemented for BaseMPSSimulation."
                )
            # Convert batch of diagonal elements to batch of diagonal matrices
            # Shape: (batch_size, n) -> (batch_size, n, n)
            batch_size, n = rhos.shape
            rhos_diag = np.zeros((batch_size, n, n), dtype=rhos.dtype)
            # Use advanced indexing to set diagonal elements for each batch
            batch_indices = np.arange(batch_size)[:, None]
            diag_indices = np.arange(n)
            rhos_diag[batch_indices, diag_indices, diag_indices] = rhos
            rhos = rhos_diag
        return np.real(np.trace(Q @ rhos, axis1=-2, axis2=-1))

    def apply_liouville_hamiltonian_modifiers(self, H, modifiers):
        """
        apply_liouville_hamiltonian_modifiers is not used for BaseMPSSimulation.
        For Haberkorn term, include kS and kT for `total_hamiltonian()`.
        """
        raise NotImplementedError(
            "apply_liouville_hamiltonian_modifiers is not used for BaseMPSSimulation."
        )

    def initial_density_matrix(self, state: State, H: np.ndarray) -> np.ndarray:
        """
        Initial density matrix is not used for BaseMPSSimulation.
        """
        raise NotImplementedError(
            "initial_density_matrix is not used for BaseMPSSimulation."
        )

    @staticmethod
    def unitary_propagator(H: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Unitary propagator is not used for BaseMPSSimulation. Use `time_evolution()` instead.
        """
        raise ValueError("unitary_propagator is not used for BaseMPSSimulation.")

    def propagate(self, propagator: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """
        Propagate is not used for BaseMPSSimulation. Use `time_evolution()` instead.
        """
        raise ValueError(
            "propagate is not used for StochasticMPSSimulation. Use `time_evolution()` instead."
        )

    def observable_projection_operator(self, state: State) -> np.ndarray:
        return self.projection_operator(state, kron_eye=False)

    def _set_ele_opsites(self):
        sx_1, sy_1, sz_1, sx_2, sy_2, sz_2, Qs, Qt = self._get_electron_ops()
        S1S2_op = OpSite(
            r"\hat{S}_1\cdot\hat{S}_2",
            self.ele_site,
            value=(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2).real,
        )
        E_op = OpSite(r"\hat{E}", self.ele_site, value=np.eye(*sx_1.shape))

        Qs_op = OpSite(r"\hat{Q}_S", self.ele_site, value=Qs)
        Qt_op = OpSite(r"\hat{Q}_T", self.ele_site, value=Qt)

        Sx_ops = []
        Sy_ops = []
        Sz_ops = []

        Sx_ops.append(OpSite(r"\hat{S}_x^{(1)}", self.ele_site, value=sx_1))
        Sy_ops.append(OpSite(r"\hat{S}_y^{(1)}", self.ele_site, value=sy_1))
        Sz_ops.append(OpSite(r"\hat{S}_z^{(1)}", self.ele_site, value=sz_1))
        Sx_ops.append(OpSite(r"\hat{S}_x^{(2)}", self.ele_site, value=sx_2))
        Sy_ops.append(OpSite(r"\hat{S}_y^{(2)}", self.ele_site, value=sy_2))
        Sz_ops.append(OpSite(r"\hat{S}_z^{(2)}", self.ele_site, value=sz_2))

        Sr_ops = [Sx_ops, Sy_ops, Sz_ops]
        self.Sr_ops = Sr_ops
        self.Qs_op = Qs_op
        self.Qt_op = Qt_op
        self.E_op = E_op
        self.S1S2_op = S1S2_op

    def _set_nuc_opsites(self, isites: list[int]):
        # Define nuclear spin operators
        Ix_ops = {}
        Iy_ops = {}
        Iz_ops = {}
        i = 0
        for j, nuc in enumerate(self.molecules[0].nuclei):
            isite = isites[i]
            i += 1
            Ix_ops[(0, j)] = OpSite(
                r"\hat{I}_x^{" + f"{(1, j + 1)}" + "}",
                isite=isite,
                value=nuc.pauli["x"],
            )
            Iy_ops[(0, j)] = OpSite(
                r"\hat{I}_y^{" + f"{(1, j + 1)}" + "}",
                isite=isite,
                value=nuc.pauli["y"],
            )
            Iz_ops[(0, j)] = OpSite(
                r"\hat{I}_z^{" + f"{(1, j + 1)}" + "}",
                isite=isite,
                value=nuc.pauli["z"],
            )

        for j, nuc in enumerate(self.molecules[1].nuclei):
            isite = isites[i]
            i += 1
            Ix_ops[(1, j)] = OpSite(
                r"\hat{I}_x^{" + f"{(2, j + 1)}" + "}",
                isite=isite,
                value=nuc.pauli["x"],
            )
            Iy_ops[(1, j)] = OpSite(
                r"\hat{I}_y^{" + f"{(2, j + 1)}" + "}",
                isite=isite,
                value=nuc.pauli["y"],
            )
            Iz_ops[(1, j)] = OpSite(
                r"\hat{I}_z^{" + f"{(2, j + 1)}" + "}",
                isite=isite,
                value=nuc.pauli["z"],
            )

        Ir_ops = [Ix_ops, Iy_ops, Iz_ops]
        self.Ir_ops = Ir_ops

    def _clean_up(self, jobname: str | None = None):
        if jobname is None:
            jobname = self.jobname
        try:
            os.remove(f"wf_{jobname}.pkl")
            # rm jobname_prop/*.dat but keep jobname_prop/reduced_density.nc
            for file in os.listdir(f"{jobname}_prop"):
                if file.endswith(".dat"):
                    os.remove(f"{jobname}_prop/{file}")
        except Exception:
            # Some OS does not support removing open files
            pass


# To pickle function, define here
def _spin_coherent_state(pair, basis, nsite, ele_site, nuclei):
    """
    Source: `Fay et al. J. Chem. Phys. 154, 084121 (2021)`_.

    Sample from spin coherent state
    |Ω⁽ᴵ⁾⟩ = cos(θ/2)²ᴵ exp(tan(θ/2)exp(iϕ)Î₋) |I,I⟩

    _Fay et al. J. Chem. Phys. 154, 084121 (2021):
    https://doi.org/10.1063/5.0040519
    """
    hp = []
    for isite in range(nsite):
        if isite == ele_site:
            hp.append([0, 0, 1, 0])  # Singlet
        else:
            mult = basis[isite].nstate
            I = (mult - 1) / 2
            nind = isite - int(ele_site <= isite)

            theta = np.arccos(pair[2 * nind] * 2 - 1.0)
            # same as
            # theta = np.arcsin(pair[2*nind] * 2 - 1.0)
            # if theta < 0:
            #    theta += np.pi
            phi = pair[2 * nind + 1] * 2 * np.pi
            weights = np.zeros((mult, 1))
            weights[0, 0] = 1.0
            weights = (
                (np.cos(theta / 2) ** (2 * I))
                * expm(np.tan(theta / 2) * np.exp(1.0j * phi) * nuclei[nind].pauli["m"])
                @ weights
            )
            assert abs((weights.conjugate().T @ weights).real[0, 0] - 1.0) < 1.0e-14, (
                weights.conjugate().T @ weights
            )[0, 0]
            hp.append(weights.reshape(-1).tolist())
    return hp


def _process_pair(
    *,
    pair,
    i,
    H,
    basis,
    nsteps,
    dt,
    bond_dimension,
    ele_site,
    backend,
    integrator,
    nsite,
    nuclei,
    jobname,
):
    operators = {"hamiltonian": H}
    basinfo = BasInfo([basis], spf_info=None)
    model = Model(basinfo=basinfo, operators=operators)
    model.m_aux_max = bond_dimension
    # Get initial Hartree product (rank-1) state
    hp = _spin_coherent_state(pair, basis, nsite, ele_site, nuclei)
    model.init_HartreeProduct = [hp]
    simulator = Simulator(jobname=jobname, model=model, backend=backend, verbose=0)
    # Save diagonal element of reduced density matrix every 1 steps
    ener, wf = simulator.propagate(
        reduced_density=(
            [(ele_site,)],
            1,
        ),
        maxstep=nsteps,
        stepsize=dt,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        observables=False,
        conserve_norm=False,  # Because of Haberkorn term
        integrator=integrator,  # or Lanczos if Hamiltonian is (skew-) Hermitian
    )

    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({ele_site},)_0"][:]["real"]
        time_data = file.variables["time"][:]

    # remove propagation directory
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    density_data = np.array(density_data_real)
    time_data = np.array(time_data)

    return density_data, time_data


def _get_nsteps_dt(time: np.ndarray):
    dt = (time[1] - time[0]) / SCALE * units.au_in_fs
    nsteps = len(time)
    return nsteps, dt


class StochasticMPSSimulation(BaseMPSSimulation):
    """Stochastic matrix product state simulation for radical pair dynamics.

    This class implements stochastic matrix product state (SMPS).
    It uses Monte Carlo sampling to generate trajectories of radical pair states.

    The simulation samples from spin coherent states and propagates them
    using tensor network methods, providing a stochastic approach to
    quantum dynamics that can handle systems with high-dimensional
    Hilbert spaces.

    Args:
        nsamples (int): Number of Monte Carlo samples to generate.
        max_workers (int): Maximum number of parallel workers for sampling.
        bond_dimension (int): Bond dimension for the matrix product state.
        backend (str): Computational backend 'numpy' only.
        integrator (str): Integration method for time evolution.
            For Hermitian Hamiltonian, use 'lanczos' for fast convergence.
            For non-Hermitian Hamiltonian, use 'arnoldi' for accurate results.
        jobname (str): Job name for output files.

    """

    def __init__(
        self,
        molecules: list[Molecule],
        custom_gfactors: bool = False,
        basis: Basis = Basis.ST,
        bond_dimension: int = 16,
        integrator: Literal["arnoldi", "lanczos"] = "arnoldi",
        nsamples: int = 128,
        max_workers: int = 8,
        jobname: str = "smps",
    ):
        nthreads = None
        env_vars = [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]
        for var in env_vars:
            val = os.environ.get(var)
            if val is not None:
                nthreads = val
                break

        if nthreads is not None:
            print(f"nthreads: {nthreads}")
        else:
            raise RuntimeError(
                "Thread number is not set. "
                + "Before importing all library, please set the following environment variables: "
                + ", ".join(env_vars)
            )

        mp.set_start_method("fork", force=True)

        super().__init__(
            molecules,
            custom_gfactors,
            basis,
            bond_dimension=bond_dimension,
            backend="numpy",
            integrator=integrator,
            jobname=jobname,
        )
        self.ele_site = len(self.molecules[0].nuclei)
        self.nsite = len(self.molecules[0].nuclei) + len(self.molecules[1].nuclei) + 1
        self.nsamples = nsamples
        self.max_workers = max_workers
        # Coefficient of the Hamiltonian
        self._set_ele_opsites()
        isites = [i for i in range(len(self.molecules[0].nuclei))] + [
            i + len(self.molecules[0].nuclei) + 1
            for i in range(len(self.molecules[1].nuclei))
        ]
        self._set_nuc_opsites(isites=isites)

    def _get_basis(self):
        basis = []
        for nuc in self.molecules[0].nuclei:
            basis.append(Exciton(nstate=nuc.multiplicity))
        basis.append(Exciton(nstate=4))
        for nuc in self.molecules[1].nuclei:
            basis.append(Exciton(nstate=nuc.multiplicity))
        return basis

    def time_evolution(
        self, init_state: State, time: np.ndarray, H: list[np.ndarray]
    ) -> np.ndarray:
        """Evolve the system through time.

        Args:

            init_state (State): Initial `State` of the density matrix
                (see `projection_operator`).

            time (np.ndarray): An sequence of (uniform) time points,
                usually created using `np.arange` or `np.linspace`.

            H (list[np.ndarray]): Hamiltonian matrix product operator.

        Returns:
            np.ndarray: Diagonal elements of the reduced density matrix.

        """
        nsteps, dt = _get_nsteps_dt(time)
        if init_state != State.SINGLET:
            raise NotImplementedError(
                "Only singlet state is supported for StochasticMPSSimulation."
            )
        basis = self._get_basis()
        op_dict = {
            tuple([(isite, isite) for isite in range(self.nsite)]): TensorOperator(
                mpo=H
            )
        }
        H = TensorHamiltonian(
            self.nsite, potential=[[op_dict]], kinetic=None, backend=self.backend
        )

        density_sum = None
        engine = np.random.default_rng(0)
        # sample from [0, 1]^N hypercubic
        pairs = engine.random((self.nsamples, 2 * (self.nsite - 1)))
        density_sums = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                density_sum = None
                active_futures = []
                i = 0  # number of submitted jobs
                j = 0  # number of finished jobs

                pbar = tqdm(total=len(pairs), desc="Processing pairs")
                while i < len(pairs) or active_futures:
                    # Submit new jobs up to max_active
                    while len(active_futures) < self.max_workers and i < len(pairs):
                        future = executor.submit(
                            _process_pair,
                            pair=pairs[i],
                            i=i,
                            H=H,
                            basis=basis,
                            nsteps=nsteps,
                            dt=dt,
                            bond_dimension=self.bond_dimension,
                            ele_site=self.ele_site,
                            backend=self.backend,
                            integrator=self.integrator,
                            nsite=self.nsite,
                            nuclei=self.nuclei,
                            jobname=f"{self.jobname}_{i}",
                        )
                        active_futures.append((future, i))
                        i += 1

                    # Wait for at least one job to complete
                    done, not_done = concurrent.futures.wait(
                        [f for f, _ in active_futures],
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    # Process completed jobs
                    remaining_futures = []
                    for future, job_i in active_futures:
                        if future in done:
                            density_data, time_data = future.result()
                            if density_sum is None:
                                density_sum = density_data
                            else:
                                density_sum += density_data
                            j += 1
                            if j == self.nsamples or (j >= 32 and j.bit_count() == 1):
                                # when j in [32, 64, 128, ...] record result to estimate convergence of Monte Carlo
                                density_sums.append(density_sum / j)
                                # Save intermediate result as npz
                                np.savez(
                                    f"radicalpair_sse_{j}samples_{self.bond_dimension}m.npz",
                                    density=density_sum / j,
                                    time=time_data * SCALE * 1e06 / units.au_in_fs,
                                )
                            pbar.update(1)
                        else:
                            remaining_futures.append((future, job_i))
                    active_futures = remaining_futures

            except KeyboardInterrupt:
                print("\nCancelling active tasks...")
                for future, _ in active_futures:
                    future.cancel()
                executor.shutdown(wait=False)
                pbar.close()
                raise

            pbar.close()

        density_data = density_sum / len(pairs)
        return np.array(density_data)  # shape: (nsteps, 4)


class LocallyPurifiedMPSSimulation(BaseMPSSimulation):
    """Locally purified matrix product state simulation for radical pair dynamics.

    This class implements locally purified matrix product state (LPMPS) simulations.
    LPMPS expresses mixed quantum states through purification.

    Args:
        molecules (list[Molecule]): List of molecules in the radical pair.
        custom_gfactors (bool): Whether to use custom g-factors.
        basis (str): Basis set for the simulation.
        bond_dimension (int): Bond dimension for the matrix product state.
        backend (str): Computational backend 'numpy' or 'jax'.
            For large dimension or GPU acceleration, use 'jax'.
        integrator (str): Integration method for time evolution.
            For Hermitian Hamiltonian, use 'lanczos' for fast convergence.
            For non-Hermitian Hamiltonian, use 'arnoldi' for accurate results.
        jobname (str): Job name for output files.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        custom_gfactors: bool = False,
        basis: Basis = Basis.ST,
        bond_dimension: int = 16,
        integrator: Literal["arnoldi", "lanczos"] = "arnoldi",
        backend: Literal["numpy", "jax", "auto"] = "auto",
        jobname: str = "lpmps",
    ):
        super().__init__(
            molecules,
            custom_gfactors,
            basis,
            bond_dimension=bond_dimension,
            backend=backend,
            integrator=integrator,
            jobname=jobname,
        )
        self.ele_site = len(self.molecules[0].nuclei) * 2
        self.nsite = (
            len(self.molecules[0].nuclei) * 2 + len(self.molecules[1].nuclei) * 2 + 1
        )
        self._set_ele_opsites()
        isites = [2 * i + 1 for i in range(len(self.molecules[0].nuclei))] + [
            self.ele_site - 1 + (i + 1) * 2
            for i in range(len(self.molecules[1].nuclei))
        ]
        print(isites)
        self._set_nuc_opsites(isites=isites)

    def total_hamiltonian(
        self,
        B0: float,
        J: float,
        D: float | np.ndarray,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        hfc_anisotropy: bool = False,
        kS: float = 0.0,
        kT: float = 0.0,
    ) -> list[np.ndarray]:
        zeeman = self.zeeman_hamiltonian(B0=B0, theta=theta, phi=phi)
        hyperfine = self.hyperfine_hamiltonian(hfc_anisotropy)
        exchange = self.exchange_hamiltonian(J)
        dipolar = self.dipolar_hamiltonian(D)
        haberkorn = self.haberkorn_hamiltonian(kS, kT)
        dummy = self._dummy_term()
        total = zeeman + hyperfine + exchange + dipolar + haberkorn + 0.0 * dummy
        total = total.simplify()
        am = AssignManager(total)
        _ = am.assign()
        mpo = am.numerical_mpo(subs=self.subs)
        return mpo

    def _dummy_term(self) -> SumOfProducts:
        eye_sites = []
        for nuc in self.molecules[0].nuclei:
            eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))
            eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))

        eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=4))

        for nuc in self.molecules[1].nuclei:
            eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))
            eye_sites.append(get_eye_site(i=len(eye_sites), n_basis=nuc.multiplicity))
        dammy_op = 1
        for op in eye_sites:
            dammy_op *= op
        return dammy_op

    def _get_basis(self):
        # Define basis for LPMPS
        basis = []
        for nuc in self.molecules[0].nuclei:
            # To add anicilla, we need to add two same basis
            for _ in range(2):
                basis.append(Exciton(nstate=nuc.multiplicity))
        # Electron basis
        basis.append(Exciton(nstate=4))
        for nuc in self.molecules[1].nuclei:
            for _ in range(2):
                basis.append(Exciton(nstate=nuc.multiplicity))
        return basis

    def _purified_state(self) -> list[np.ndarray]:
        hp = []
        for nuc in self.molecules[0].nuclei:
            mult = nuc.multiplicity
            core_anci = np.zeros((1, mult, mult))
            core_phys = np.zeros((mult, mult, 1))
            core_anci[0, :, :] = np.eye(mult)
            core_phys[:, :, 0] = np.eye(mult)
            core_phys /= np.sqrt(mult)
            hp.append(core_anci)
            hp.append(core_phys)
        # electron site = singlet
        core_phys = np.zeros((1, 4, 1))
        core_phys[0, 2, 0] = 1.0
        hp.append(core_phys)
        # hp.append(core_anci)
        for nuc in self.molecules[1].nuclei:
            mult = nuc.multiplicity
            core_phys = np.zeros((1, mult, mult))
            core_anci = np.zeros((mult, mult, 1))
            core_phys[0, :, :] = np.eye(mult)
            core_anci[:, :, 0] = np.eye(mult)
            core_phys /= np.sqrt(mult)
            hp.append(core_phys)
            hp.append(core_anci)
        assert len(hp) == self.nsite
        return hp

    def time_evolution(
        self, init_state: State, time: np.ndarray, H: list[np.ndarray]
    ) -> np.ndarray:
        """Evolve the system through time.

        Args:

            init_state (State): Initial `State` of the density matrix
                (see `projection_operator`).

            time (np.ndarray): An sequence of (uniform) time points,
                usually created using `np.arange` or `np.linspace`.

            H (list[np.ndarray]): Hamiltonian matrix product operator.

        Returns:
            np.ndarray: Diagonal elements of the reduced density matrix.

        """
        nsteps, dt = _get_nsteps_dt(time)
        if init_state != State.SINGLET:
            raise NotImplementedError(
                "Only singlet state is supported for StochasticMPSSimulation."
            )
        basis = self._get_basis()
        # Define where MPO has "legs"
        # key = ((0, 0), (1,), (2, 2), (3,), (4, 4), (5,), (6, 6))
        # while
        # leg is (0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6)
        key = []
        for i in range(self.nsite):
            if i == self.ele_site:
                act_site = (i, i)
            else:
                if i % 2 == 1:
                    act_site = (i, i)
                else:
                    act_site = (i,)
            key.append(act_site)
        key = tuple(key)  # list is not hashable
        op_dict = {key: TensorOperator(mpo=H, legs=tuple(chain.from_iterable(key)))}
        H = TensorHamiltonian(
            self.nsite, potential=[[op_dict]], kinetic=None, backend=self.backend
        )
        operators = {"hamiltonian": H}
        basinfo = BasInfo([basis], spf_info=None)
        model = Model(basinfo=basinfo, operators=operators, space="Hilbert")
        model.m_aux_max = self.bond_dimension
        hp = self._purified_state()
        model.init_HartreeProduct = [hp]
        jobname = self.jobname
        simulator = Simulator(
            jobname=jobname, model=model, backend=self.backend, verbose=1
        )
        # Save diagonal element of reduced density matrix every 1 steps
        ener, wf = simulator.propagate(
            reduced_density=(
                [(self.ele_site,)],
                1,
            ),
            maxstep=nsteps,
            stepsize=dt,
            autocorr=False,
            energy=False,
            norm=False,
            populations=False,
            observables=False,
            conserve_norm=False,  # Since Haberkorn term is included
            integrator=self.integrator,  # or "Lanczos" if Hamiltonian is (skew-) Hermitian
        )

        with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
            density_data_real = file.variables[f"rho_({self.ele_site},)_0"][:]["real"]
            time_data = file.variables["time"][:]

        density_data = np.array(density_data_real)
        time_data = np.array(time_data)
        self._clean_up(jobname)

        return density_data


# Define linear operation
def _get_OE(op):
    """
    OT ⊗ 𝟙
    """
    return np.kron(op.T, np.eye(op.shape[0], dtype=op.dtype))


def _get_EO(op):
    """
    𝟙 ⊗ O
    """
    return np.kron(np.eye(op.shape[0], dtype=op.dtype), op)


class MPDOSimulation(BaseMPSSimulation):
    """Vectorised matrix product density operator simulation for radical pair dynamics.

    This class implements vectorised matrix product density operator (VMPDO).
    It uses density operators to describe mixed quantum states,
    allowing for the simulation of decoherence and relaxation processes.

    Args:
        molecules (list[Molecule]): List of molecules in the radical pair.
        custom_gfactors (bool): Whether to use custom g-factors.
        basis (str): Basis set for the simulation.
        bond_dimension (int): Bond dimension for the matrix product operator.
        backend (str): Computational backend 'numpy' or 'jax'.
            For large dimension or GPU acceleration, use 'jax'.
        integrator (str): Integration method for time evolution.
            For Hermitian Hamiltonian, use 'lanczos' for fast convergence.
            For non-Hermitian Hamiltonian, use 'arnoldi' for accurate results.
        jobname (str): Job name for output files.

    """

    def __init__(
        self,
        molecules: list[Molecule],
        custom_gfactors: bool = False,
        basis: Basis = Basis.ST,
        bond_dimension: int = 16,
        integrator: Literal["arnoldi", "lanczos"] = "arnoldi",
        backend: Literal["numpy", "jax", "auto"] = "auto",
        jobname: str = "vmpdo",
    ):
        super().__init__(
            molecules,
            custom_gfactors,
            basis,
            bond_dimension=bond_dimension,
            backend=backend,
            integrator=integrator,
            jobname=jobname,
        )
        self.ele_site = len(self.molecules[0].nuclei)
        self.nsite = len(self.molecules[0].nuclei) + len(self.molecules[1].nuclei) + 1
        self._set_ele_opsites()
        self._set_nuc_opsites()

    def _set_ele_opsites(self):
        sx_1, sy_1, sz_1, sx_2, sy_2, sz_2, Qs, Qt = self._get_electron_ops()
        SxE_ops = []
        SyE_ops = []
        SzE_ops = []
        ESx_ops = []
        ESy_ops = []
        ESz_ops = []

        S1S2E_op = OpSite(
            r"(\hat{S}_1\cdot\hat{S}_2)^\ast ⊗ 𝟙",
            self.ele_site,
            value=_get_OE(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2),
        )
        ES1S2_op = OpSite(
            r"𝟙 ⊗ (\hat{S}_1\cdot\hat{S}_2)",
            self.ele_site,
            value=_get_EO(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2),
        )
        EE_op = OpSite(
            r"\hat{E} ⊗ \hat{E}", self.ele_site, value=_get_OE(np.eye(sx_1.shape[0]))
        )

        QsE_op = OpSite(r"\hat{Q}_S ⊗ 𝟙", self.ele_site, value=_get_OE(Qs))
        EQs_op = OpSite(r"𝟙 ⊗ \hat{Q}_S", self.ele_site, value=_get_EO(Qs))
        QtE_op = OpSite(r"\hat{Q}_T ⊗ 𝟙", self.ele_site, value=_get_OE(Qt))
        EQt_op = OpSite(r"𝟙 ⊗ \hat{Q}_T", self.ele_site, value=_get_EO(Qt))

        SxE_ops.append(
            OpSite(r"\hat{S}_x^{(1)\ast} ⊗ 𝟙", self.ele_site, value=_get_OE(sx_1))
        )
        SxE_ops.append(
            OpSite(r"\hat{S}_x^{(2)\ast} ⊗ 𝟙", self.ele_site, value=_get_OE(sx_2))
        )
        SyE_ops.append(
            OpSite(r"\hat{S}_y^{(1)\ast} ⊗ 𝟙", self.ele_site, value=_get_OE(sy_1))
        )
        SyE_ops.append(
            OpSite(r"\hat{S}_y^{(2)\ast} ⊗ 𝟙", self.ele_site, value=_get_OE(sy_2))
        )
        SzE_ops.append(
            OpSite(r"\hat{S}_z^{(1)\ast} ⊗ 𝟙", self.ele_site, value=_get_OE(sz_1))
        )
        SzE_ops.append(
            OpSite(r"\hat{S}_z^{(2)\ast} ⊗ 𝟙", self.ele_site, value=_get_OE(sz_2))
        )

        ESx_ops.append(
            OpSite(r"𝟙 ⊗ \hat{S}_x^{(1)}", self.ele_site, value=_get_EO(sx_1))
        )
        ESx_ops.append(
            OpSite(r"𝟙 ⊗ \hat{S}_x^{(2)}", self.ele_site, value=_get_EO(sx_2))
        )
        ESy_ops.append(
            OpSite(r"𝟙 ⊗ \hat{S}_y^{(1)}", self.ele_site, value=_get_EO(sy_1))
        )
        ESy_ops.append(
            OpSite(r"𝟙 ⊗ \hat{S}_y^{(2)}", self.ele_site, value=_get_EO(sy_2))
        )
        ESz_ops.append(
            OpSite(r"𝟙 ⊗ \hat{S}_z^{(1)}", self.ele_site, value=_get_EO(sz_1))
        )
        ESz_ops.append(
            OpSite(r"𝟙 ⊗ \hat{S}_z^{(2)}", self.ele_site, value=_get_EO(sz_2))
        )

        SrE_ops = [SxE_ops, SyE_ops, SzE_ops]
        ESr_ops = [ESx_ops, ESy_ops, ESz_ops]

        self.SrE_ops = SrE_ops
        self.ESr_ops = ESr_ops
        self.EE_op = EE_op
        self.QsE_op = QsE_op
        self.EQs_op = EQs_op
        self.QtE_op = QtE_op
        self.EQt_op = EQt_op
        self.S1S2E_op = S1S2E_op
        self.ES1S2_op = ES1S2_op

    def _set_nuc_opsites(self):
        # Define nuclear spin operators
        IxE_ops = {}
        IyE_ops = {}
        IzE_ops = {}
        EIx_ops = {}
        EIy_ops = {}
        EIz_ops = {}

        for j, nuc in enumerate(self.molecules[0].nuclei):
            val = nuc.pauli["x"]
            IxE_ops[(0, j)] = OpSite(
                r"\hat{I}_x^{" + f"{(1, j + 1)}" + r"\ast} ⊗ 𝟙",
                j,
                value=_get_OE(val),
            )
            EIx_ops[(0, j)] = OpSite(
                r"𝟙 ⊗ \hat{I}_x^{" + f"{(1, j + 1)}" + "}",
                j,
                value=_get_EO(val),
            )
            val = nuc.pauli["y"]
            IyE_ops[(0, j)] = OpSite(
                r"\hat{I}_y^{" + f"{(1, j + 1)}" + r"\ast} ⊗ 𝟙",
                j,
                value=_get_OE(val),
            )
            EIy_ops[(0, j)] = OpSite(
                r"𝟙 ⊗ \hat{I}_y^{" + f"{(1, j + 1)}" + "}",
                j,
                value=_get_EO(val),
            )
            val = nuc.pauli["z"]
            IzE_ops[(0, j)] = OpSite(
                r"\hat{I}_z^{" + f"{(1, j + 1)}" + r"\ast} ⊗ 𝟙",
                j,
                value=_get_OE(val),
            )
            EIz_ops[(0, j)] = OpSite(
                r"𝟙 ⊗ \hat{I}_z^{" + f"{(1, j + 1)}" + "}",
                j,
                value=_get_EO(val),
            )

        for j, nuc in enumerate(self.molecules[1].nuclei):
            site = self.ele_site + 1 + j
            val = nuc.pauli["x"]
            IxE_ops[(1, j)] = OpSite(
                r"\hat{I}_x^{" + f"{(2, j + 1)}" + r"\ast} ⊗ 𝟙",
                site,
                value=_get_OE(val),
            )
            EIx_ops[(1, j)] = OpSite(
                r"𝟙 ⊗ \hat{I}_x^{" + f"{(2, j + 1)}" + "}",
                site,
                value=_get_EO(val),
            )
            val = nuc.pauli["y"]
            IyE_ops[(1, j)] = OpSite(
                r"\hat{I}_y^{" + f"{(2, j + 1)}" + r"\ast} ⊗ 𝟙",
                site,
                value=_get_OE(val),
            )
            EIy_ops[(1, j)] = OpSite(
                r"𝟙 ⊗ \hat{I}_y^{" + f"{(2, j + 1)}" + "}",
                self.ele_site + 1 + j,
                value=_get_EO(val),
            )
            val = nuc.pauli["z"]
            IzE_ops[(1, j)] = OpSite(
                r"\hat{I}_z^{" + f"{(2, j + 1)}" + r"\ast} ⊗ 𝟙",
                site,
                value=_get_OE(val),
            )
            EIz_ops[(1, j)] = OpSite(
                r"𝟙 ⊗ \hat{I}_z^{" + f"{(2, j + 1)}" + "}",
                site,
                value=_get_EO(val),
            )

        IrE_ops = [IxE_ops, IyE_ops, IzE_ops]
        EIr_ops = [EIx_ops, EIy_ops, EIz_ops]

        self.IrE_ops = IrE_ops
        self.EIr_ops = EIr_ops

    def zeeman_hamiltonian(
        self,
        B0: float,
        B_axis: str = "z",
        theta: Optional[float] = None,
        phi: Optional[float] = None,
    ) -> SumOfProducts:
        zeeman = SumOfProducts()
        xyz = "xyz"
        B = _get_vecB(B0, B_axis, theta, phi)
        for a, (SrE_op, ESr_op, IrE_op, EIr_op) in enumerate(
            zip(self.SrE_ops, self.ESr_ops, self.IrE_ops, self.EIr_ops, strict=True)
        ):
            if B[a] == 0.0:
                continue
            r = xyz[a]
            Br = Symbol(f"B_{r}")
            self.subs[Br] = B[a] * SCALE
            for i in range(len(self.radicals)):
                zeeman += -Br * self.g_ele_sym[i] * ESr_op[i]
                zeeman += Br * self.g_ele_sym[i] * SrE_op[i]
                for j in range(len(self.molecules[i].nuclei)):
                    zeeman += -Br * self.g_nuc_sym[(i, j)] * EIr_op[(i, j)]
                    zeeman -= -Br * self.g_nuc_sym[(i, j)] * IrE_op[(i, j)]
        zeeman = zeeman.simplify()
        return zeeman

    def hyperfine_hamiltonian(self, hfc_anisotropy: bool = False) -> SumOfProducts:
        hyperfine = SumOfProducts()
        xyz = "xyz"
        for i in range(len(self.radicals)):
            for j, nuc in enumerate(self.molecules[i].nuclei):
                if hfc_anisotropy:
                    A_ij = nuc.hfc.anisotropic
                else:
                    A_ij = np.eye(3) * nuc.hfc.isotropic
                for a, (SrE_op, ESr_op) in enumerate(
                    zip(self.SrE_ops, self.ESr_ops, strict=True)
                ):
                    for b, (IrE_op, EIr_op) in enumerate(
                        zip(self.IrE_ops, self.EIr_ops, strict=True)
                    ):
                        if A_ij[a, b] == 0.0:
                            continue
                        Asym = Symbol(
                            "A^{"
                            + f"{(i + 1, j + 1)}"
                            + "}_{"
                            + f"{xyz[a]}"
                            + f"{xyz[b]}"
                            + "}"
                        )
                        self.subs[Asym] = A_ij[a, b].item() * SCALE
                        hyperfine += (
                            Asym * abs(self.g_ele_sym[0]) * ESr_op[i] * EIr_op[(i, j)]
                        )
                        hyperfine -= (
                            Asym * abs(self.g_ele_sym[0]) * SrE_op[i] * IrE_op[(i, j)]
                        )
        hyperfine = hyperfine.simplify()
        return hyperfine

    def exchange_hamiltonian(self, J: float, prod_coeff: float = 2) -> np.ndarray:
        exchange = SumOfProducts()
        Jsym = Symbol("J")
        self.subs[Jsym] = J * SCALE
        exchange += (
            -Jsym * abs(self.g_ele_sym[0]) * (2 * self.ES1S2_op + 0.5 * self.EE_op)
        )
        exchange -= (
            -Jsym * abs(self.g_ele_sym[0]) * (2 * self.S1S2E_op + 0.5 * self.EE_op)
        )
        exchange = exchange.simplify()
        return exchange

    def dipolar_hamiltonian(self, D: float | np.ndarray) -> np.ndarray:
        if isinstance(D, float):
            if D > 0.0:
                print(
                    f"WARNING: D is {D} mT, which is positive. In point dipole approximation, D should be negative."
                )
            D = 2 / 3 * np.diag((-1.0, -1.0, 2.0)) * D
        # Define Dipolar Hamiltonian
        dipolar = SumOfProducts()
        xyz = "xyz"
        for a in range(3):
            for b in range(3):
                if D[a, b] == 0.0:
                    continue
                Dsym = Symbol("D_{" + f"{xyz[a]}" + f"{xyz[b]}" + "}")
                self.subs[Dsym] = D[a, b] * SCALE
                dipolar += (
                    Dsym
                    * abs(self.g_ele_sym[0])
                    * self.ESr_ops[a][0]
                    * self.ESr_ops[b][1]
                )
                dipolar -= (
                    Dsym
                    * abs(self.g_ele_sym[0])
                    * self.SrE_ops[a][0]
                    * self.SrE_ops[b][1]
                )
        dipolar = dipolar.simplify()
        return dipolar

    def haberkorn_hamiltonian(self, kS: float, kT: float) -> SumOfProducts:
        haberkorn = SumOfProducts()
        if kS != 0.0:
            kSsym = Symbol("k_S")
            self.subs[kSsym] = kS * SCALE
            haberkorn -= 1.0j * kSsym / 2 * (self.QsE_op + self.EQs_op)
        if kT != 0.0:
            kTsym = Symbol("k_T")
            self.subs[kTsym] = kT * SCALE
            haberkorn -= 1.0j * kTsym / 2 * (self.QtE_op + self.EQt_op)
        haberkorn = haberkorn.simplify()
        return haberkorn

    def _get_basis(self):
        basis = []
        for nuc in self.molecules[0].nuclei:
            basis.append(Exciton(nstate=nuc.multiplicity**2))
        basis.append(Exciton(nstate=4**2))
        for nuc in self.molecules[1].nuclei:
            basis.append(Exciton(nstate=nuc.multiplicity**2))
        return basis

    def _initial_state(self, basis, state: State):
        hp = []
        for isite in range(self.nsite):
            if isite == self.ele_site:
                match state:
                    case State.SINGLET:
                        op = np.diag([0, 0, 1, 0])
                    case State.TRIPLET_PLUS:
                        op = np.diag([1, 0, 0, 0])
                    case State.TRIPLET_ZERO:
                        op = np.diag([0, 1, 0, 0])
                    case State.TRIPLET_MINUS:
                        op = np.diag([0, 0, 0, 1])
                    case State.TRIPLET_PLUS_MINUS:
                        op = np.diag([1, 0, 0, 1]) / 2
                    case State.TRIPLET:
                        op = np.diag([1, 1, 0, 1]) / 3
                    case _:
                        raise NotImplementedError(f"Invalid state: {state}")
            else:
                # Mixed states !
                op = np.eye(isqrt(basis[isite].nstate))
            # Automatically nomarized so that trace=1 in internal code
            hp.append(op.reshape(-1).tolist())
        return hp

    def time_evolution(
        self, init_state: State, time: np.ndarray, H: list[np.ndarray]
    ) -> np.ndarray:
        """Evolve the system through time.

        Args:

            init_state (State): Initial `State` of the density matrix
                (see `projection_operator`).

            time (np.ndarray): An sequence of (uniform) time points,
                usually created using `np.arange` or `np.linspace`.

            H (list[np.ndarray]): Hamiltonian matrix product operator.

        Returns:
            np.ndarray: Reduced density matrix with shape (nsteps, 4, 4).

        """
        nsteps, dt = _get_nsteps_dt(time)
        basis = self._get_basis()
        basinfo = BasInfo([basis], spf_info=None)
        op_dict = {
            tuple([(isite, isite) for isite in range(self.nsite)]): TensorOperator(
                mpo=H
            )
        }
        H = TensorHamiltonian(
            self.nsite, potential=[[op_dict]], kinetic=None, backend=self.backend
        )
        operators = {"hamiltonian": H}
        # space is "Liouville" rather than "Hilbert"
        model = Model(basinfo=basinfo, operators=operators, space="Liouville")
        model.m_aux_max = self.bond_dimension
        hp = self._initial_state(basis, init_state)
        model.init_HartreeProduct = [hp]
        jobname = self.jobname
        simulator = Simulator(
            jobname=jobname,
            model=model,
            backend=self.backend,
            verbose=0,
        )
        # Save whole reduced density matrix every 1 steps
        ener, wf = simulator.propagate(
            reduced_density=(
                [(self.ele_site, self.ele_site)],
                1,
            ),
            maxstep=nsteps,
            stepsize=dt,
            autocorr=False,
            energy=False,
            norm=False,
            populations=False,
            observables=False,
            integrator=self.integrator,  # or Lanczos if linealised Liouvillian is (skew-) Hermitian
        )
        data = read_nc(
            f"{jobname}_prop/reduced_density.nc", [(self.ele_site, self.ele_site)]
        )
        time_data = data["time"]
        density_data = data[(self.ele_site, self.ele_site)]
        self._clean_up(jobname)
        return density_data
