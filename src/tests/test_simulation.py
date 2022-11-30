#! /usr/bin/env python

import doctest
import os
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from src.radicalpy import data as rpdata
from src.radicalpy import estimations, kinetics, relaxation
from src.radicalpy import simulation as rpsim
from src.radicalpy import utils
from src.radicalpy.simulation import Basis

import tests.radpy as radpy

# np.seterr(divide="raise", invalid="raise")

RUN_SLOW_TESTS = "INSIDE_EMACS" not in os.environ  # or True
MEASURE_TIME = False


PARAMS = dict(
    B=np.random.uniform(size=20),
    J=np.random.uniform(),
    D=np.random.uniform(),
)

RADICAL_PAIR = [
    rpsim.Molecule("flavin_anion", ["H29"]),
    rpsim.Molecule("adenine_cation"),
    # rpsim.Molecule("adenine_cation", ["C8-H"]),
]

RADICAL_PAIR_RAW = [
    rpsim.Molecule(
        multiplicities=[2, 2],
        gammas_mT=[rpdata.gamma_mT("E"), rpdata.gamma_mT("E")],
        hfcs=[0, 0],
    ),
    rpsim.Molecule(multiplicities=[2], gammas_mT=[rpdata.gamma_mT("E")], hfcs=[0]),
]


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(rpsim))
    tests.addTests(doctest.DocTestSuite(kinetics))
    return tests


def state2radpy(state: rpsim.State) -> str:
    return (
        str(state.value)
        .replace("+", "p")
        .replace("-", "m")
        .replace("/", "")
        .replace("_", "")
        .replace("\\", "")
    )


class MoleculeTests(unittest.TestCase):
    def test_effective_hyperfine(self):
        flavin = rpsim.Molecule("flavin_anion", ["N5"])
        self.assertAlmostEqual(flavin.effective_hyperfine, 1.4239723207027404)

    def test_manual_effective_hyperfine(self):
        nuclei = ["14N"] * 4 + ["1H"] * 12
        hfcs = [
            0.5233,
            0.1887,
            -0.0035,
            -0.0383,
            0.0565,
            -0.1416,
            -0.1416,
            -0.1416,
            -0.3872,
            0.4399,
            0.4399,
            0.4399,
            0.0099,
            0.407,
            0.407,
            -0.0189,
        ]
        flavin = rpsim.Molecule(nuclei=nuclei, hfcs=hfcs)
        self.assertAlmostEqual(flavin.effective_hyperfine, 1.3981069)


class HilbertTests(unittest.TestCase):
    def setUp(self):
        if MEASURE_TIME:
            self.start_time = time.time()
        self.data = rpsim.MOLECULE_DATA["adenine_cation"]["data"]
        self.sim = rpsim.HilbertSimulation(RADICAL_PAIR, basis=Basis.ZEEMAN)
        self.gamma_mT = rpdata.gamma_mT("E")
        self.dt = 0.01
        self.t_max = 1.0
        self.time = np.arange(0, self.t_max, self.dt)

    def tearDown(self):
        if MEASURE_TIME:
            print(f"Time: {time.time() - self.start_time}")

    @unittest.skip("Maybe bad test")
    def test_molecule_properties(self):
        molecule = rpsim.Molecule("adenine_cation", ["N6-H1", "C8-H"])
        for prop in ["hfc", "element"]:
            for i, h in enumerate(molecule._get_properties(prop)):
                assert h == molecule._get_property(i, prop)

    @unittest.skip("Maybe bad test")
    def test_molecule_name(self):
        molecule = rpsim.Molecule("adenine_cation", ["N6-H1", "C8-H"])
        for i, h in enumerate(molecule.hfcs):
            assert h == self.data[molecule.nuclei[i]]["hfc"]
        for i, g in enumerate(molecule.gammas_mT):
            elem = self.data[molecule.nuclei[i]]["element"]
            assert g == rpdata.gamma_mT(elem)
        for i, m in enumerate(molecule.multiplicities):
            elem = self.data[molecule.nuclei[i]]["element"]
            assert m == rpdata.multiplicity(elem)

    def test_molecule_raw(self):
        hfcs = [0.1, 0.2]
        multiplicities = [2, 3]
        gammas_mT = [3.14, 2.71]

        molecule = rpsim.Molecule(
            hfcs=hfcs, multiplicities=multiplicities, gammas_mT=gammas_mT
        )
        for i in range(2):
            assert hfcs[i] == molecule.hfcs[i]
            assert multiplicities[i] == molecule.multiplicities[i]
            assert gammas_mT[i] == molecule.gammas_mT[i]

    def test_molecule_empty(self):
        """Test empty molecule.

        A silly test which verifies that the "empty" molecule has:
        - shape = (4, 4)
        - exactly two non-zero entries, and
        - those entries have opposite signs.

        """
        mol = rpsim.Molecule()
        sim = rpsim.HilbertSimulation([mol, mol])
        HZ = sim.zeeman_hamiltonian(0.5)
        nz = HZ != 0
        assert HZ.shape == (4, 4)
        assert len(HZ[nz]) == 2
        assert np.sum(HZ) == 0

    def test_HZ_raw(self):
        ################
        # RadicalPy code
        sim = rpsim.HilbertSimulation(RADICAL_PAIR_RAW)
        HZ = sim.zeeman_hamiltonian(PARAMS["B"][0])

        #########################
        # Assume this is correct!
        omega_e = PARAMS["B"][0] * self.gamma_mT
        electrons = sum(
            [radpy.np_spinop(radpy.np_Sz, i, sim.num_particles) for i in range(2)]
        )
        omega_n = PARAMS["B"][0] * rpdata.gamma_mT("E")
        nuclei = sum(
            [
                radpy.np_spinop(radpy.np_Sz, i, sim.num_particles)
                for i in range(2, sim.num_particles)
            ]
        )
        HZ_true = -omega_e * electrons - omega_n * nuclei
        assert np.all(
            np.isclose(HZ, HZ_true)
        ), "Zeeman Hamiltonian not calculated properly."

    def test_HZ(self):
        HZ = self.sim.zeeman_hamiltonian(PARAMS["B"][0])

        #########################
        # Assume this is correct!
        omega_e = PARAMS["B"][0] * self.gamma_mT
        electrons = sum(
            [radpy.np_spinop(radpy.np_Sz, i, self.sim.num_particles) for i in range(2)]
        )
        omega_n = PARAMS["B"][0] * rpdata.gamma_mT("1H")
        nuclei = sum(
            [
                radpy.np_spinop(radpy.np_Sz, i, self.sim.num_particles)
                for i in range(2, self.sim.num_particles)
            ]
        )
        HZ_true = -omega_e * electrons - omega_n * nuclei

        assert np.all(
            np.isclose(HZ, HZ_true)
        ), "Zeeman Hamiltonian not calculated properly."

    def test_HH(self):
        couplings = self.sim.coupling
        hfcs = self.sim.hfcs
        HH_true = sum(
            [
                radpy.HamiltonianHyperfine(
                    self.sim.num_particles,
                    ei,
                    2 + ni,
                    utils.isotropic(hfcs[ni]),
                    self.gamma_mT,
                )
                for ni, ei in enumerate(couplings)
            ]
        )
        assert np.all(
            self.sim.hyperfine_hamiltonian() == HH_true
        ), "Hyperfine Hamiltonian not calculated properly."

    def test_HE(self):
        HE_true = radpy.HamiltonianExchange(
            self.sim.num_particles, PARAMS["J"], gamma=self.gamma_mT
        )
        HE = self.sim.exchange_hamiltonian(PARAMS["J"])
        np.testing.assert_almost_equal(HE, HE_true)

    def test_HD(self):
        HD_true = radpy.HamiltonianDipolar(
            self.sim.num_particles, PARAMS["D"], self.gamma_mT
        )
        HD = self.sim.dipolar_hamiltonian(PARAMS["D"])
        np.testing.assert_almost_equal(HD, HD_true)

    @unittest.skip("This needs to be figured out")
    def test_HH_3D(self):
        # from SingletYield.ipynb do
        ### anisotropic
        # hfcs = [N5, N10, H5, H4]
        # yields = singletYieldsAvgAniso(nucDims, indE, hfcs, b0*mT2angfreq, k0, kS)
        #
        # paper: "Radical triads, not pairs,
        # may explain effects of hypomagnetic fields on neurogenesis"
        mT2angfreq = (
            9.274009994e-24 / 1.0545718e-34 * 2.00231930436256 / 1e9
        )  # Mrad/s/mT; ~28 MHz/mT
        MHz2mT = 1 / (mT2angfreq) * 2 * np.pi * 1e6
        N5 = (
            np.array(
                [
                    [-2.41368, -0.0662465, -0.971492],
                    [-0.0662465, -2.44657, 0.0485258],
                    [-0.971492, 0.0485258, 43.5125],
                ]
            )
        ) * MHz2mT
        N10 = (
            np.array(
                [
                    [0.442319, 0.06085, 1.8016],
                    [0.06085, -0.0133137, -0.338064],
                    [1.8016, -0.338064, 23.1529],
                ]
            )
        ) * MHz2mT
        H5 = (
            np.array(
                [
                    [-2.38856, 1.8683, 0.514044],
                    [1.8683, -40.6401, 0.0339364],
                    [0.514044, 0.0339364, -27.8618],
                ]
            )
        ) * MHz2mT

        flavin = rpsim.Molecule(
            hfcs=[N5, N10, H5],
            multiplicities=[3, 3, 2],
            gammas_mT=[
                rpsim.gamma_mT("14N"),
                rpsim.gamma_mT("14N"),
                rpsim.gamma_mT("1H"),
            ],
        )

        H4 = 0.176 * np.eye(3)
        ascorbic_acid = rpsim.Molecule(
            hfcs=[H4],
            multiplicities=[2],
            gammas_mT=[rpsim.gamma_mT("1H")],
        )

        sim = rpsim.HilbertSimulation([flavin, ascorbic_acid])
        H = sim.hyperfine_hamiltonian()
        # print(H.shape)
        # print(H)
        Htrue = np.load("/tmp/save.npy") * MHz2mT * 1e6
        print(Htrue[:5, :5])
        print(H[:5, :5])
        print(f"\nError: {np.linalg.norm(Htrue - H)}")
        # plt.imshow(np.real(H))
        # plt.show()

    @unittest.skip("Doesn't check anything")
    def test_3d(self):
        H = self.sim.zeeman_hamiltonian_3d(1, 10, 20)

    def test_dipolar_interaction_1d(self):
        approx = estimations.dipolar_interaction_1d(1)
        gold = approx
        self.assertEqual(gold, approx)

    def test_initial_density_matrix(self):
        H = self.sim.total_hamiltonian(PARAMS["B"][0], PARAMS["J"], PARAMS["D"])
        for state in rpsim.State:
            rho0 = self.sim.initial_density_matrix(state, H)
            rpstate = state2radpy(state)
            rho0_true = radpy.Hilbert_initial(rpstate, self.sim.num_particles, H)
            np.testing.assert_almost_equal(rho0, rho0_true)

    def test_unitary_propagator(self):
        dt = np.random.uniform(1e-6)
        H = self.sim.total_hamiltonian(PARAMS["B"][0], PARAMS["J"], PARAMS["D"])
        unitary_true = radpy.UnitaryPropagator(H, dt, "Hilbert")
        unitary = self.sim.unitary_propagator(H, dt)
        np.testing.assert_almost_equal(unitary_true, unitary)

    def test_time_evolution(self):
        k = np.random.uniform()
        H = self.sim.total_hamiltonian(PARAMS["B"][0], PARAMS["J"], PARAMS["D"])
        Kexp = kinetics.Exponential(k)
        for init_state in rpsim.State:
            for obs_state in rpsim.State:
                if obs_state == rpsim.State.EQUILIBRIUM:
                    continue
                evol_true = radpy.TimeEvolution(
                    self.sim.num_particles,
                    state2radpy(init_state),
                    state2radpy(obs_state),
                    self.t_max,
                    self.dt,
                    k,
                    0,
                    H,
                    "Hilbert",
                )
                rhos = self.sim.time_evolution(init_state, self.time, H)
                pprob = self.sim.product_probability(obs_state, rhos)
                pprob = pprob[1:]
                Kexp.adjust_product_probabilities(pprob, self.time[:-1])
                pyield, pyield_sum = self.sim.product_yield(pprob, self.time[:-1], k)
                assert np.all(  # close (not equal)
                    np.isclose(rhos[1:], evol_true[-1][:-1])
                ), "Time evolution (rho) failed."
                assert np.all(  # close (not equal)
                    np.isclose(pprob, evol_true[1][:-1])
                ), "Time evolution (probability or kinetics) failed."
                assert np.all(  # close (not equal)
                    np.isclose(pyield, evol_true[2][:-1])
                ), "Time evolution (product yield) failed."

    # @unittest.skip("Not ready yet")
    def test_mary(self):
        k = np.random.uniform()
        for init_state in rpsim.State:
            for obs_state in rpsim.State:
                if obs_state == rpsim.State.EQUILIBRIUM:
                    continue
                rslt = self.sim.MARY(
                    init_state,
                    obs_state,
                    self.time,
                    B=PARAMS["B"],
                    D=PARAMS["D"],
                    J=PARAMS["J"],
                    kinetics=[kinetics.Exponential(k)],
                )
                # time, MFE, HFE, LFE, MARY, _, _, rho = radpy.MARY(
                #     self.sim.num_particles,
                #     init_state,
                #     obs_state,
                #     self.t_max,
                #     self.dt,
                #     k,
                #     PARAMS["B"],
                #     H,
                # )

    @unittest.skip("Numerical difference")
    def test_ST_vs_Zeeman_basis(self):
        k = np.random.uniform()
        st_sim = rpsim.HilbertSimulation(RADICAL_PAIR, basis=Basis.ST)
        ts = np.arange(0, 5e-6, 5e-9)
        for i, init_state in enumerate(rpsim.State):
            for j, obs_state in enumerate(rpsim.State):
                if obs_state == rpsim.State.EQUILIBRIUM:
                    continue
                kwargs = dict(
                    init_state=init_state,
                    obs_state=obs_state,
                    time=ts,
                    B=PARAMS["B"],
                    D=PARAMS["D"],
                    J=PARAMS["J"],
                    kinetics=[kinetics.Exponential(k)],
                )
                rslt = self.sim.MARY(**kwargs)
                strs = st_sim.MARY(**kwargs)

                key = "time_evolutions"
                Bi = 1
                # print(results.keys())
                # print(results["product_yields"])
                B = PARAMS["B"]
                n = len(rpsim.State)
                idx = i * n + j + 1
                plt.subplot(n, n, idx)

                title = f"{init_state.value}, {obs_state.value}"
                suptitle = f"{key}: B={B[Bi]}"
                plt.title(title)
                plt.suptitle(suptitle)
                plt.plot(rslt["time"], rslt[key][Bi])
                plt.plot(strs["time"], strs[key][Bi])
        # np.testing.assert_almost_equal(rslt[key], strs[key])
        plt.show()

    def test_hyperfine_3d(self):

        results = self.sim.MARY(
            rpsim.State.SINGLET,
            rpsim.State.TRIPLET,
            self.time,
            PARAMS["B"],
            PARAMS["D"],
            PARAMS["J"],
            theta=np.pi / 2,
            phi=0,
            hfc_anisotropy=True,
        )
        self.assertEqual(results["time_evolutions"][0][0], 0)
        # print(results.keys())
        # print(results["product_yields"])
        # plt.plot(results["time"], results["time_evolutions"][1])
        # plt.show()

        N5 = np.array(
            [
                [-2.41368, -0.0662465, -0.971492],
                [-0.0662465, -2.44657, 0.0485258],
                [-0.971492, 0.0485258, 43.5125],
            ]
        )
        N5 /= 28.025
        dipolar_tensor = np.array(
            [
                [5680970.81962565, -65574461.04030437, 34606093.12997659],
                [-65574461.04030436, -34583196.80020875, 47131454.19638795],
                [34606093.12997659, 47131454.19638795, 28902225.98058307],
            ]
        )

        molecules = [
            rpsim.Molecule("flavin3d", nuclei=["14N"], hfcs=[N5]),
            rpsim.Molecule(),
        ]
        B0 = 0.05
        B = np.arange(-10e-3, 10e-3, 1e-3)
        sim = rpsim.HilbertSimulation(molecules)
        HZ = sim.zeeman_hamiltonian_3d(B0=B0, theta=0, phi=0)
        HZt = sim.zeeman_hamiltonian_3d(B0=B0, theta=np.pi / 2, phi=0)
        HZtp = sim.zeeman_hamiltonian_3d(B0=B0, theta=np.pi / 2, phi=np.pi)
        HZp = sim.zeeman_hamiltonian_3d(B0=B0, theta=np.pi, phi=0)
        HH = sim.hyperfine_hamiltonian()
        HD = sim.dipolar_hamiltonian_3d(dipolar_tensor)
        H = HZt + HH + HD
        time = np.arange(0, 15e-6, 5e-9)
        # rhos = sim.time_evolution(rpsim.State.SINGLET, time, H)
        # pp = sim.product_probability(rpsim.State.TRIPLET, rhos)
        # print(sim)
        # print(f"{HZ.shape=}")
        # print(f"{HH.shape=}")
        # print(f"{HD.shape=}")
        # print(f"{rhos.shape=}")
        # plt.plot(time, pp)
        # plt.show()
        # print(HZ)
        # results = sim.MARY(
        #     rpsim.State.SINGLET,
        #     rpsim.State.TRIPLET,
        #     time,
        #     B,
        #     dipolar_tensor,
        #     0,
        #     kinetics=[kinetics.Exponential(3e6)],
        #     theta=np.pi / 2,
        #     phi=0,
        # )
        # idx = 0
        # plt.plot(results["B"], results["MARY"])
        # plt.title(f"B={results['B'][idx]}")
        # plt.show()
        # print("DONE")


class LiouvilleTests(unittest.TestCase):
    def setUp(self):
        self.sim = rpsim.LiouvilleSimulation(RADICAL_PAIR, basis=Basis.ZEEMAN)
        self.dt = 0.01
        self.t_max = 1.0
        self.time = np.arange(0, self.t_max, self.dt)

    def test_initial_density_matrix(self):
        H = self.sim.total_hamiltonian(PARAMS["B"][0], PARAMS["J"], PARAMS["D"])
        for state in rpsim.State:
            rho0 = self.sim.initial_density_matrix(state, H)
            rpstate = state2radpy(state)
            rho0_true = radpy.Liouville_initial(rpstate, self.sim.num_particles, H)
            np.testing.assert_almost_equal(rho0, rho0_true)

    def test_unitary_propagator(self):
        dt = np.random.uniform(0, 1e-6)
        H = self.sim.total_hamiltonian(PARAMS["B"][0], PARAMS["J"], PARAMS["D"])
        unitary_true = radpy.UnitaryPropagator(H, dt, "Liouville")
        unitary = self.sim.unitary_propagator(H, dt)
        np.testing.assert_almost_equal(unitary, unitary_true)

    @unittest.skipUnless(RUN_SLOW_TESTS, "slow")
    def test_kinetics(self):
        kwargs = dict(
            init_state=rpsim.State.SINGLET,
            obs_state=rpsim.State.TRIPLET,
            time=np.arange(0, 15e-6, 5e-9),
            B=np.arange(0, 20, 1),
            D=0,
            J=0,
        )
        k = 1e6
        results_haberkorn = self.sim.MARY(
            kinetics=[
                kinetics.Haberkorn(k, rpsim.State.TRIPLET),
                kinetics.Haberkorn(k, rpsim.State.SINGLET),
                kinetics.Exponential(k),
            ],
            **kwargs,
        )
        results_jones_hore = self.sim.MARY(
            kinetics=[
                kinetics.JonesHore(k, k),
                kinetics.Exponential(k),
            ],
            **kwargs,
        )

        # idx = 0
        # plt.plot(results_haberkorn["time"], results_haberkorn["time_evolutions"][idx])
        # plt.plot(results_jones_hore["time"], results_jones_hore["time_evolutions"][idx])
        # plt.title(f"B={results_haberkorn['B'][idx]}")
        # plt.show()
        # print("DONE")

    def test_relaxation(self):
        kwargs = dict(
            init_state=rpsim.State.TRIPLET,
            obs_state=rpsim.State.TRIPLET,
            time=np.arange(0, 5e-6, 1e-9),
            B=np.arange(0, 4, 1),
            D=0,
            J=0,
        )
        k = 1e6
        results = self.sim.MARY(
            kinetics=[],
            relaxations=[
                # relaxation.SingletTripletDephasing( k),
                # relaxation.TripleTripletDephasing( k),
                relaxation.RandomFields(k),
                # relaxation.DipolarModulation( k),
                # relaxation.TripletTripletRelaxation( k),
            ],
            **kwargs,
        )

        # idx = 0
        # plt.plot(results["time"], results["time_evolutions"][idx])
        # plt.title(f"B={results['B'][idx]}")
        # plt.show()
        # print("DONE")
