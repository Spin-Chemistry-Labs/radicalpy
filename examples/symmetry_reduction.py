### PARAMETRERS ###

B0 = 0.5  # Magnetic field along z-axis in mT
J = 0.25  # Exchange coupling in mT
D = -0.1  # Point dipole approximation in mT

dt = 1e-9  # Time step in ns
T = 1e-07  # Total propagation time in ns

# Effective isotropic hyperfine coupling for each molecule in mT
A1 = 0.6
A2 = 0.3

# Number of identical nuclei for each molecule
N1 = 10  # up to ~50 for SparseCholeskyHilbertSimulation
N2 = 10  # up to ~50 for SparseCholeskyHilbertSimulation

# If you want to parallelise, set parallel = True (not recommended for testing)
parallel = False
nthreads = 2  # numbr of theread per process (only valid if parallel==True)

###################

if parallel:
    import concurrent.futures
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    mp.set_start_method("fork", force=True)  # <-- Spawn (Mac OS default) does not work.
    # These must be set before importing numpy and scipy.
    os.environ["OMP_NUM_THREADS"] = f"{nthreads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{nthreads}"
    os.environ["MKL_NUM_THREADS"] = f"{nthreads}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{nthreads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{nthreads}"


from itertools import product

import numpy as np
from tqdm.auto import tqdm

import radicalpy as rp
from radicalpy.data import FuseNucleus

time = np.arange(0, T, dt)  # 100 steps to save time
h_left = rp.data.Molecule.fromisotopes(
    name="h_left",
    isotopes=["1H"] * N1,
    hfcs=[A1] * N1,
)
h_right = rp.data.Molecule.fromisotopes(
    name="h_right",
    isotopes=["1H"] * N2,
    hfcs=[A2] * N2,
)
fused_h_left = FuseNucleus.from_nuclei(h_left.nuclei)
fused_h_right = FuseNucleus.from_nuclei(h_right.nuclei)
singlet_probability = None
items = list(product(fused_h_left, fused_h_right))


def each_block_dynamics(w1, h1, w2, h2, B0, J, D, time):
    if h1.multiplicity == 1:
        mol1 = rp.data.Molecule(name="mol1", nuclei=[])
    else:
        mol1 = rp.data.Molecule(name="mol1", nuclei=[h1])
    if h2.multiplicity == 1:
        mol2 = rp.data.Molecule(name="mol2", nuclei=[])
    else:
        mol2 = rp.data.Molecule(name="mol2", nuclei=[h2])
    sim = rp.simulation.SparseCholeskyHilbertSimulation([mol1, mol2])
    ham = sim.total_hamiltonian(B0=B0, J=J, D=D)
    # print(f"{ham.size=} {ham.data.nbytes / 1e6:.2f} MB")
    rhos_c = sim.time_evolution(rp.simulation.State.SINGLET, time, ham)
    # print(f"{sum([X.data.nbytes * 2 for (X, Xt) in rhos_c]) / 1e6:.2f} MB")
    singlet_probability_block = (
        w1 * w2 * sim.product_probability(rp.simulation.State.SINGLET, rhos_c)
    )
    return singlet_probability_block


if parallel:
    # execute `nprocs` * `nthreds` should be smaller than number of CPUs
    singlet_probability = None
    nprocs = (os.cpu_count() - 1) // nthreads
    # 1 core is used for master, and others are used for slave
    print(f"{nprocs=} {nthreads=}")
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        try:
            active_futures = []
            i = 0  # number of submitted jobs
            j = 0  # number of finished jobs
            pbar = tqdm(total=len(items), desc="Processing each block")
            while i < len(items) or active_futures:
                # Submit new jobs up to max_active
                while len(active_futures) < nprocs and i < len(items):
                    (w1, h1), (w2, h2) = items[i]
                    future = executor.submit(
                        each_block_dynamics, w1, h1, w2, h2, B0, J, D, time
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
                        singlet_probability_block = future.result()
                        if singlet_probability is None:
                            singlet_probability = singlet_probability_block
                        else:
                            singlet_probability += singlet_probability_block
                        j += 1
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
else:
    for (w1, h1), (w2, h2) in tqdm(items):
        singlet_probability_block = each_block_dynamics(w1, h1, w2, h2, B0, J, D, time)
        if singlet_probability is None:
            singlet_probability = singlet_probability_block
        else:
            singlet_probability += singlet_probability_block


# If you want to plot, comment out following commands
# import matplotlib.pyplot as plt
# plt.plot(time, singlet_probability)
# plt.show()
