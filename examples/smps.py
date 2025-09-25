import os

# Set number of thread before importing numpy, scipy, radicalpy, pytdscf, etc
nthreads = 1  # number of threads
max_workers = os.cpu_count() - 1  # number of processes
os.environ["OMP_NUM_THREADS"] = f"{nthreads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{nthreads}"
os.environ["MKL_NUM_THREADS"] = f"{nthreads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{nthreads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{nthreads}"

try:
    import pytdscf
except ModuleNotFoundError:
    # You can install it via `pip install git+https://github.com/QCLovers/PyTDSCF`
    print("pytdscf not found, skipping example")
    import sys

    sys.exit(0)

import numpy as np

import radicalpy as rp
import radicalpy.tensornetwork as tn

B0 = 0.05  # in mT
B = np.array((0.0, 0.0, 1.0)) * B0
J = 0.224  # Fay et al 2020
D = -0.38  # Fay et al 2020
kS = 1.0e06  # in s-1
kT = 1.0e06  # in s-1
kS = 0.0  # in s-1
kT = 0.0  # in s-1
if isinstance(D, float):
    assert D <= 0
    D = 2 / 3 * np.diag((1.0, 1.0, -2.0)) * (-D)

methyl = rp.data.Molecule.fromisotopes(
    name="methyl",
    isotopes=["1H", "1H", "1H"],
    hfcs=[0.5, 1.0, 1.5],
)
sim = tn.StochasticMPSSimulation(
    [methyl, methyl],
    bond_dimension=32,
    nsamples=128,
    max_workers=4,
    integrator="lanczos" if kS == kT == 0 else "arnoldi",
)
ham = sim.total_hamiltonian(B0=B0, J=J, D=D)
time = np.arange(0, 2e-08, 1e-9)  # only 20 steps to save time
reduced_rho_diag = sim.time_evolution(rp.simulation.State.SINGLET, time, ham)
time_evol_mps = sim.product_probability(rp.simulation.State.SINGLET, reduced_rho_diag)

sim = rp.simulation.HilbertSimulation([methyl, methyl])
ham = sim.total_hamiltonian(B0=B0, J=J, D=D)
rhos = sim.time_evolution(rp.simulation.State.SINGLET, time, ham)
time_evol = sim.product_probability(rp.simulation.State.SINGLET, rhos)

import matplotlib.pyplot as plt

plt.plot(time * 1e09, time_evol, label="RP")
plt.plot(time * 1e09, time_evol_mps, label="MPS")
plt.legend()
plt.xlabel("Time (ns)")
plt.ylabel("Singlet Probability")
plt.grid(":")
plt.show()
