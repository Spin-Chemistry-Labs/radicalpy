from itertools import product

import numpy as np
import scipy as sp
from tqdm.auto import tqdm

import radicalpy as rp
from radicalpy.data import FuseNucleus

B0 = 0.5
J = 0.5
D = 0.3
time = np.arange(0, 2e-07, 1e-9)  # only 20 steps to save time
dt = time[1] - time[0]

assert D > 0, "current radicalpy assumes D>0"

# bench mark calculation

n = 20
h_left = rp.data.Molecule.fromisotopes(
    name="h_left",
    isotopes=["1H"] * n,
    hfcs=[0.6] * n,
)
h_right = rp.data.Molecule.fromisotopes(
    name="h_right",
    isotopes=["1H"] * n,
    hfcs=[0.3] * n,
)

import time as _time

fused_h_left = FuseNucleus.from_nuclei(h_left.nuclei)
fused_h_right = FuseNucleus.from_nuclei(h_right.nuclei)
time_evol_true_list = []
items = list(product(fused_h_left, fused_h_right))
for (w1, h1), (w2, h2) in tqdm(items):  #
    if h1.multiplicity == 1:
        mol1 = rp.data.Molecule(name="mol1", nuclei=[])
    else:
        mol1 = rp.data.Molecule(name="mol1", nuclei=[h1])
    if h2.multiplicity == 1:
        mol2 = rp.data.Molecule(name="mol2", nuclei=[])
    else:
        mol2 = rp.data.Molecule(name="mol2", nuclei=[h2])
    sim = rp.simulation.SparseCholeskyHilbertSimulation([mol1, mol2])
    start = _time.time()
    ham = sim.total_hamiltonian(B0=B0, J=J, D=D)
    rhos_c = sim.time_evolution(rp.simulation.State.SINGLET, time, ham)
    singlet_probability_c = sim.product_probability(rp.simulation.State.SINGLET, rhos_c)
    end = _time.time()
    print(f"{ham.size=} {ham.data.nbytes / 1e6:.2f} MB")
    print(f"{sum([X.data.nbytes * 2 for (X, Xt) in rhos_c]) / 1e6:.2f} MB")
    print(f"cholesky time: {end - start:.2f} s")

    sim = rp.simulation.HilbertSimulation([mol1, mol2])
    start = _time.time()
    ham = sim.total_hamiltonian(B0=B0, J=J, D=D)
    rhos = sim.time_evolution(rp.simulation.State.SINGLET, time, ham, cholesky=False)
    singlet_probability = sim.product_probability(rp.simulation.State.SINGLET, rhos)
    end = _time.time()
    print(f"{ham.size=} {ham.data.nbytes / 1e6:.2f} MB")
    print(f"{sum([rho.data.nbytes for rho in rhos]) / 1e6:.2f} MB")
    print(f"direct time: {end - start:.2f} s")
    np.testing.assert_allclose(singlet_probability_c, singlet_probability, atol=1e-12)
    time_evol_true_list.append(
        w1 * w2 * sim.product_probability(rp.simulation.State.SINGLET, rhos)
    )
    print("-" * 10)
