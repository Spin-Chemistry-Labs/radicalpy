import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import src.radicalpy as rp

import tests.radpy as radpy

RUN_SLOW_TESTS = False
MEASURE_TIME = False


class QuantumTests(unittest.TestCase):
    def setUp(self):
        if MEASURE_TIME:
            self.start_time = time.time()
        self.data = rp.data.MOLECULE_DATA["adenine"]["data"]
        self.rad_pair = [
            rp.Molecule("adenine", ["N6-H1", "N6-H2"]),
            rp.Molecule("adenine", ["C8-H"]),
        ]
        self.sim = rp.simulation.QuantumSimulation(self.rad_pair)
        self.spins = self.sim.num_particles
        self.gamma_mT = rp.data.SPIN_DATA["E"]["gamma"] * 0.001
        self.states = ["S", "Tm", "T0", "Tp", "Tpm"]
        self.B = np.random.uniform(size=20)
        self.J = np.random.uniform()
        self.D = np.random.uniform()

    def tearDown(self):
        if MEASURE_TIME:
            print(f"Time: {time.time() - self.start_time}")

    def test_molecule_properties(self):
        molecule = rp.Molecule("adenine", ["N6-H1", "C8-H"])
        for prop in ["hfc", "element"]:
            for i, h in enumerate(molecule._get_properties(prop)):
                assert h == molecule._get_property(i, prop)

    def test_molecule_name(self):
        molecule = rp.Molecule("adenine", ["N6-H1", "C8-H"])
        for i, h in enumerate(molecule.hfcs):
            assert h == self.data[molecule.nuclei[i]]["hfc"]
        for i, g in enumerate(molecule.gammas_mT):
            elem = self.data[molecule.nuclei[i]]["element"]
            assert g == rp.data.SPIN_DATA[elem]["gamma"] * 0.001
        for i, m in enumerate(molecule.multiplicities):
            elem = self.data[molecule.nuclei[i]]["element"]
            assert m == rp.data.SPIN_DATA[elem]["multiplicity"]

    def test_molecule_raw(self):
        hfcs = [0.1, 0.2]
        multiplicities = [2, 3]
        gammas_mT = [3.14, 2.71]

        molecule = rp.Molecule(
            hfcs=hfcs, multiplicities=multiplicities, gammas_mT=gammas_mT
        )
        for i in range(2):
            assert hfcs[i] == molecule.hfcs[i]
            assert multiplicities[i] == molecule.multiplicities[i]
            assert gammas_mT[i] == molecule.gammas_mT[i]

    def test_molecule_raw_nohfcs(self):
        multiplicities = [2, 3]
        gammas_mT = [3.14, 2.71]

        molecule = rp.Molecule(multiplicities=multiplicities, gammas_mT=gammas_mT)
        for i in range(2):
            assert multiplicities[i] == molecule.multiplicities[i]
            assert gammas_mT[i] == molecule.gammas_mT[i]

    def test_molecule_empty(self):
        """Test empty molecule.

        A silly test which verifies that the "empty" molecule has:
        - shape = (4, 4)
        - exactly two non-zero entries, and
        - those entries have opposite signs.

        """
        mol = rp.Molecule()
        sim = rp.simulation.QuantumSimulation([mol, mol])
        HZ = sim.zeeman_hamiltonian(0.5)
        nz = HZ != 0
        assert HZ.shape == (4, 4)
        assert len(HZ[nz]) == 2
        assert np.sum(HZ) == 0

    def test_HZ_raw(self):
        ################
        # RadicalPy code
        gamma_mT = 3.14
        rad_pair = [
            rp.Molecule(multiplicities=[2, 2], gammas_mT=[gamma_mT, gamma_mT]),
            rp.Molecule(multiplicities=[2], gammas_mT=[gamma_mT]),
        ]
        sim = rp.simulation.QuantumSimulation(rad_pair)
        HZ = sim.zeeman_hamiltonian(self.B[0])

        #########################
        # Assume this is correct!
        omega_e = self.B[0] * self.gamma_mT
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, self.spins) for i in range(2)])
        omega_n = self.B[0] * gamma_mT
        nuclei = sum(
            [radpy.np_spinop(radpy.np_Sz, i, self.spins) for i in range(2, self.spins)]
        )
        HZ_true = -omega_e * electrons - omega_n * nuclei

        assert np.all(
            np.isclose(HZ, HZ_true)
        ), "Zeeman Hamiltonian not calculated properly."

    def test_HZ(self):
        HZ = self.sim.zeeman_hamiltonian(self.B[0])

        #########################
        # Assume this is correct!
        omega_e = self.B[0] * self.gamma_mT
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, self.spins) for i in range(2)])
        omega_n = self.B[0] * rp.data.SPIN_DATA["1H"]["gamma"] * 0.001
        nuclei = sum(
            [radpy.np_spinop(radpy.np_Sz, i, self.spins) for i in range(2, self.spins)]
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
                    self.spins, ei, 2 + ni, hfcs[ni], self.gamma_mT
                )
                for ni, ei in enumerate(couplings)
            ]
        )
        assert np.all(
            np.isclose(self.sim.hyperfine_hamiltonian(), HH_true)
        ), "Hyperfine Hamiltonian not calculated properly."

    def test_HE(self):
        HE_true = radpy.HamiltonianExchange(self.spins, self.J, gamma=self.gamma_mT)
        assert np.all(
            np.isclose(self.sim.exchange_hamiltonian(self.J), HE_true)
        ), "Exchange (J-coupling) Hamiltonian not calculated properly."

    def test_HD(self):
        HD_true = radpy.HamiltonianDipolar(self.spins, self.D, self.gamma_mT)
        assert np.all(
            np.isclose(self.sim.dipolar_hamiltonian(self.D), HD_true)
        ), "Dipolar Hamiltonian not calculated properly."

    @unittest.skip("This needs to be figured out")
    def test_HH_3D(self):
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

        flavin = rp.Molecule(
            hfcs=[N5, N10, H5],
            multiplicities=[3, 3, 2],
            gammas_mT=[
                rp.data.gamma_mT("14N"),
                rp.data.gamma_mT("14N"),
                rp.data.gamma_mT("1H"),
            ],
        )

        H4 = 0.176 * np.eye(3)
        ascorbic_acid = rp.Molecule(
            hfcs=[H4],
            multiplicities=[2],
            gammas_mT=[rp.data.gamma_mT("1H")],
        )

        sim = rp.simulation.QuantumSimulation([flavin, ascorbic_acid])
        H = sim.hyperfine_hamiltonian()
        # print(H.shape)
        # print(H)
        Htrue = np.load("/tmp/save.npy") * MHz2mT * 1e6
        print(Htrue[:5, :5])
        print(H[:5, :5])
        print(f"\nError: {np.linalg.norm(Htrue - H)}")
        # plt.imshow(np.real(H))
        # plt.show()


class HilbertTests(QuantumTests):
    def setUp(self):
        super().setUp()
        self.sim = rp.simulation.HilbertSimulation(self.rad_pair)
        self.dt = 0.01
        self.t_max = 1.0
        self.time = np.arange(0, self.t_max, self.dt)

    def test_hilbert_initial(self):
        H = self.sim.total_hamiltonian(self.B[0], self.J, self.D)
        for state in self.states:
            rho0 = self.sim.hilbert_initial(state, H)
            rho0_true = radpy.Hilbert_initial(state, self.spins, H)
            assert np.all(
                np.isclose(rho0, rho0_true)
            ), "Initial density not calculated properly."

    def test_hilbert_unitary_propagator(self):
        dt = np.random.uniform(1e-6)
        H = self.sim.total_hamiltonian(self.B[0], self.J, self.D)
        U_true = radpy.UnitaryPropagator(H, dt, "Hilbert")
        Utensor = self.sim.hilbert_unitary_propagator(H, dt)
        for pair in zip(U_true, Utensor):
            assert np.all(np.isclose(*pair))

    def test_hilbert_time_evolution(self):
        k = np.random.uniform()
        H = self.sim.total_hamiltonian(self.B[0], self.J, self.D)
        Kexp = self.sim.kinetics_exponential(k, self.time)
        for init_state in self.states:
            for obs_state in self.states:
                evol_true = radpy.TimeEvolution(
                    self.spins,
                    init_state,
                    obs_state,
                    self.t_max,
                    self.dt,
                    k,
                    0,
                    H,
                    "Hilbert",
                )
                rhos = self.sim.hilbert_time_evolution(init_state, self.time, H)
                pprob = self.sim.product_probability(obs_state, rhos)
                pprob_kinetics = pprob[1:] * Kexp[:-1]
                pyield, pyield_sum = self.sim.product_yield(
                    pprob_kinetics, self.time[:-1], k
                )
                assert np.all(
                    np.isclose(rhos[1:], evol_true[-1][:-1])
                ), "Time evolution (rho) failed."
                assert np.all(
                    np.isclose(pprob_kinetics, evol_true[1][:-1])
                ), "Time evolution (probability or kinetics) failed."
                assert np.all(
                    np.isclose(pyield, evol_true[2][:-1])
                ), "Time evolution (product yield) failed."


class LiouvilleTests(QuantumTests):
    def setUp(self):
        super().setUp()
        self.sim = rp.simulation.LiouvilleSimulation(self.rad_pair)
        self.dt = 0.01
        self.t_max = 1.0
        self.time = np.arange(0, self.t_max, self.dt)

    def test_liouville_initial(self):
        H = self.sim.total_hamiltonian(self.B[0], self.J, self.D)
        for state in self.states:
            rho0 = self.sim.liouville_initial(state, H)
            rho0_true = radpy.Liouville_initial(state, self.spins, H)
            assert np.all(
                np.isclose(rho0, rho0_true)
            ), "Initial density not calculated properly."

    def test_liouville_unitary_propagator(self):
        dt = np.random.uniform(0, 1e-6)
        H = self.sim.total_hamiltonian(self.B[0], self.J, self.D)
        U_true = radpy.UnitaryPropagator(H, dt, "Liouville")
        U_prop = self.sim.liouville_unitary_propagator(H, dt)
        assert np.all(np.isclose(U_true, U_prop))

    # @unittest.skipUnless(RUN_SLOW_TESTS, "slow")
    def test_liouville_time_evolution(self):
        H = self.sim.total_hamiltonian(self.B[0], self.J, self.D)
        HL = self.sim.hilbert_to_liouville(H)
        for init_state in self.states:
            for obs_state in self.states:
                rhos = self.sim.liouville_time_evolution(init_state, self.time, H)[1:]
                evol_true = radpy.TimeEvolution(
                    self.spins,
                    init_state,
                    obs_state,
                    self.t_max,
                    self.dt,
                    k=0,
                    B=0,
                    H=HL,
                    space="Liouville",
                )
                prob = self.sim.product_probability(obs_state, rhos)
                # assert np.all(
                #     np.isclose(rhos, evol_true[-1][:-1])
                # ), "Time evolution (rho) failed)"
                assert np.all(
                    np.isclose(prob, evol_true[1][:-1])
                ), "Time evolution (probability) failed)"

    @unittest.skip("Not ready yet")
    def test_mary(self):
        k = np.random.uniform()
        H = self.sim.total_hamiltonian(0, self.J, self.D)
        for init_state in self.states:
            for obs_state in self.states:
                rslt = self.sim.mary(init_state, obs_state, self.time, k, self.B, H)
                time, MFE, HFE, LFE, MARY, _, _, rho = radpy.MARY(
                    self.spins,
                    init_state,
                    obs_state,
                    self.t_max,
                    self.dt,
                    k,
                    self.B,
                    H,
                )

                # print(f"{MFE=}")
                # print(f"{rslt['MFE']=}")
                # assert np.all(
                #     np.isclose(prob, evol_true[1][:-1])
                # ), ""
