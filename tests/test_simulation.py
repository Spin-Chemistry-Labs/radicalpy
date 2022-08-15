import unittest

import numpy as np
import src.radicalpy as rp

import tests.radpy as radpy


class DummyTests(unittest.TestCase):
    def setUp(self):
        self.data = rp.data.MOLECULE_DATA["adenine"]["data"]
        self.rad_pair = [
            rp.Molecule("adenine", ["N6-H1", "N6-H2"]),
            rp.Molecule("adenine", ["C8-H"]),
        ]
        self.B = np.random.uniform()
        self.sim = rp.simulation.Quantum(self.rad_pair)
        self.spins = self.sim.num_particles
        self.gamma_mT = rp.data.SPIN_DATA["E"]["gamma"] * 0.001
        self.states = ["S", "Tm", "T0", "Tp", "Tpm"]

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
        sim = rp.simulation.Quantum([mol, mol])
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
        B = np.random.uniform()
        sim = rp.simulation.Quantum(rad_pair)
        HZ = sim.zeeman_hamiltonian(B)

        #########################
        # Assume this is correct!
        omega_e = B * self.gamma_mT
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, self.spins) for i in range(2)])
        omega_n = B * gamma_mT
        nuclei = sum(
            [radpy.np_spinop(radpy.np_Sz, i, self.spins) for i in range(2, self.spins)]
        )
        HZ_true = -omega_e * electrons - omega_n * nuclei

        assert np.all(
            np.isclose(HZ, HZ_true)
        ), "Zeeman Hamiltonian not calculated properly."

    def test_HZ(self):
        HZ = self.sim.zeeman_hamiltonian(self.B)

        #########################
        # Assume this is correct!
        omega_e = self.B * self.gamma_mT
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, self.spins) for i in range(2)])
        omega_n = self.B * rp.data.SPIN_DATA["1H"]["gamma"] * 0.001
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
        J = np.random.uniform()
        HE_true = radpy.HamiltonianExchange(self.spins, J, gamma=self.gamma_mT)
        assert np.all(
            np.isclose(self.sim.exchange_hamiltonian(J), HE_true)
        ), "Exchange (J-coupling) Hamiltonian not calculated properly."

    def test_HD(self):
        D = np.random.uniform()
        HD_true = radpy.HamiltonianDipolar(self.spins, D, self.gamma_mT)
        assert np.all(
            np.isclose(self.sim.dipolar_hamiltonian(D), HD_true)
        ), "Dipolar Hamiltonian not calculated properly."

    def test_hilbert_initial(self):
        B = np.random.uniform()
        J = np.random.uniform()
        D = np.random.uniform()
        H = self.sim.total_hamiltonian(B, J, D)
        state = "S"
        for state in self.states:
            rho0 = self.sim.hilbert_initial(state, H)
            rho0_true = radpy.Hilbert_initial(state, self.spins, H)
            assert np.all(
                np.isclose(rho0, rho0_true)
            ), "Initial density not calculated properly."

    def test_hilbert_observable(self):
        for state in self.states:
            obs = self.sim.hilbert_observable(state)
            obs_true = radpy.Hilbert_observable(state, self.spins)
            for pair in zip(obs, obs_true):
                assert np.all(
                    np.isclose(*pair)
                ), "Initial density not calculated properly."

    def test_hilbert_unitary_propagator(self):
        B = np.random.uniform()
        J = np.random.uniform()
        D = np.random.uniform()
        dt = np.random.uniform()
        H = self.sim.total_hamiltonian(B, J, D)
        U_true = radpy.UnitaryPropagator(H, dt, "Hilbert")
        Utensor = self.sim.hilbert_unitary_propagator(H, dt)
        for pair in zip(U_true, Utensor):
            assert np.all(np.isclose(*pair))

    def test_hilbert_time_evolution(self):
        dt = 0.01
        t_max = 1.0
        time = np.arange(0, t_max, dt)
        k = 1e-10

        B = np.random.uniform()
        J = np.random.uniform()
        D = np.random.uniform()
        H = self.sim.total_hamiltonian(B, J, D)
        for init_state in self.states:
            for obs_state in self.states:
                evol = self.sim.hilbert_time_evolution(init_state, obs_state, time, H)
                evol_true = radpy.TimeEvolution(
                    self.spins, init_state, obs_state, t_max, dt, k, 0, H, "Hilbert"
                )
                assert np.all(np.isclose(evol["evol"], evol_true[1]))
                assert np.all(np.isclose(evol["rho"], evol_true[-1]))

    @unittest.skip("Keeping only for the notes from earlier")
    def test_dummy(self):
        # # creates K "Hamiltonian"
        # sim.kinetics(model="Haberkorn", recombination=3e6, escape=1e6)
        # # creates Relaxation "Hamiltonian"
        # sim.relaxation(model="STD", rate=1e6)
        # # create HJ
        # sim.J_coupling(0.0)
        # # create HD
        # sim.dipolar_coupling(0.0)

        # sim.time_evolution(time=np.linspace())
        # sim.mary(time=np.linspace(), magnetic_field=np.linspace())
        # sim.angle(time=np.linspace(), theta=np.linspace(), phi=np.linspace())
        pass
