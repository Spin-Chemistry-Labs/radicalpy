import unittest

import numpy as np
import src.radicalpy as rp

import tests.radpy as radpy


class DummyTests(unittest.TestCase):
    def setUp(self):
        self.data = rp.data.MOLECULE_DATA["adenine"]["data"]
        ################
        # RadicalPy code
        self.rad_pair = [
            rp.Molecule("adenine", ["N6-H1", "N6-H2"]),
            rp.Molecule("adenine", ["C8-H"]),
        ]
        self.B = 0.5
        self.sim = rp.simulation.Quantum(self.rad_pair)

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
        for i, m in enumerate(molecule.multis):
            elem = self.data[molecule.nuclei[i]]["element"]
            assert m == rp.data.SPIN_DATA[elem]["multiplicity"]

    def test_molecule_raw(self):
        hfcs = [0.1, 0.2]
        multis = [2, 3]
        gammas_mT = [3.14, 2.71]

        molecule = rp.Molecule(hfcs=hfcs, multis=multis, gammas_mT=gammas_mT)
        for i in range(2):
            assert hfcs[i] == molecule.hfcs[i]
            assert multis[i] == molecule.multis[i]
            assert gammas_mT[i] == molecule.gammas_mT[i]

    def test_molecule_raw_nohfcs(self):
        multis = [2, 3]
        gammas_mT = [3.14, 2.71]

        molecule = rp.Molecule(multis=multis, gammas_mT=gammas_mT)
        for i in range(2):
            assert multis[i] == molecule.multis[i]
            assert gammas_mT[i] == molecule.gammas_mT[i]

    def test_HZ_raw(self):
        ################
        # RadicalPy code
        gamma_mT = 3.14
        rad_pair = [
            rp.Molecule(hfcs=[1, 2], multis=[2, 2], gammas_mT=[gamma_mT, gamma_mT]),
            rp.Molecule(hfcs=[3], multis=[2], gammas_mT=[gamma_mT]),
        ]
        B = 0.5
        sim = rp.simulation.Quantum(rad_pair)
        spins = sim.num_particles
        HZ = sim.HZ(B)

        #########################
        # Assume this is correct!
        omega_e = B * rp.data.SPIN_DATA["E"]["gamma"] * 0.001
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2)])
        omega_n = B * gamma_mT
        nuclei = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2, spins)])
        HZ_true = -omega_e * electrons - omega_n * nuclei

        assert np.all(
            np.isclose(HZ, HZ_true)
        ), "Zeeman Hamiltonian not calculated properly."

    def test_HZ(self):
        HZ = self.sim.HZ(self.B)

        #########################
        # Assume this is correct!
        spins = self.sim.num_particles
        omega_e = self.B * rp.data.SPIN_DATA["E"]["gamma"] * 0.001
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2)])
        omega_n = self.B * rp.data.SPIN_DATA["1H"]["gamma"] * 0.001
        nuclei = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2, spins)])
        HZ_true = -omega_e * electrons - omega_n * nuclei

        assert np.all(
            np.isclose(HZ, HZ_true)
        ), "Zeeman Hamiltonian not calculated properly."

    def test_HH(self):
        spins = self.sim.num_particles
        couplings = self.sim.coupling
        hfcs = self.sim.hfcs
        gamma_mT = rp.data.SPIN_DATA["E"]["gamma"] * 0.001
        HH_true = sum(
            [
                radpy.HamiltonianHyperfine(spins, ei, 2 + ni, hfcs[ni], gamma_mT)
                for ni, ei in enumerate(couplings)
            ]
        )
        assert np.all(
            np.isclose(self.sim.HH(), HH_true)
        ), "Hyperfine Hamiltonian not calculated properly."

    @unittest.skip("Keeping only for the notes from earlier")
    def test_dummy(self):
        rad_pair = [
            rp.Molecule("adenine", ["N6-H1", "N6-H2"]),
            rp.Molecule("adenine", ["C8-H"]),
        ]
        B = 0.5
        # print(rad_pair[0].hfc)
        spins = 2 + sum([len(t.nuclei_list) for t in rad_pair])

        # calculates HZ, HH
        sim = rp.Quantum(
            rad_pair,
            # hfc1_custom=[0.5, 0.6], # mT
            # atom1_custom=["H", "N"]
            # or
            # multiplicity1_custom=[1, 3] # or spin1_custom?
            # gamma1_custom=[0.1, 0.4] ???
            # mT vs MHz!
            kinetics=dict(model="Haberkorn", recombination=3e6, escape=1e6),
        )
        # SAIA, SAIB, SBIC

        HZ = sim.hamiltonians["HZ"]

        # Assume this is correct!
        omega_e = (B * 1.760859644e8,)
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2)])
        omega_n = B * 267.513e3
        nuclei = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2, spins)])
        HZ_true = omega_e * electrons - omega_n * nuclei

        sim.hyperfine()

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
