import unittest

import numpy as np
import src.radicalpy as rp

import tests.radpy as radpy


class DummyTests(unittest.TestCase):
    def setUp(self):
        self.m = rp.Molecule("adenine", ["N6-H1", "C8-H"])

    # def test_shared_sigmas(self):
    #     self.sim.sigmas[0][0, 1, 1] = 42
    #     assert self.sim.sigmas[2][0, 1, 1] == 42

    def test_molecule_data(self):
        for prop in ["hfc", "element"]:
            for i, h in enumerate(self.m.data_generator(prop)):
                assert h == self.m.get_data(i, prop)

    def test_zeeman(self):
        ################
        # RadicalPy code
        rad_pair = [
            rp.Molecule("adenine", ["N6-H1", "N6-H2"]),
            rp.Molecule("adenine", ["C8-H"]),
        ]
        B = 0.5
        sim = rp.simulation.Quantum(rad_pair)
        spins = sim.num_particles
        HZ = sim.HZ(B)

        #########################
        # Assume this is correct!
        omega_e = B * 1.760859644e8
        electrons = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2)])
        omega_n = B * 267.513e3
        nuclei = sum([radpy.np_spinop(radpy.np_Sz, i, spins) for i in range(2, spins)])
        HZ_true = omega_e * electrons - omega_n * nuclei

        assert np.all(HZ == HZ_true), "Zeeman Hamiltonian not calculated properly."

    @unittest.skip("Keeping only for the notes from earlier")
    def test_dummy(self):
        rad_pair = [
            rp.Molecule("adenine", ["N6-H1", "N6-H2"]),
            rp.Molecule("adenine", ["C8-H"]),
        ]
        B = 0.5
        # print(rad_pair[0].hfc)
        spins = 2 + sum([len(t.hfc) for t in rad_pair])

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

        assert np.all(HZ == HZ_true), "Zeeman Hamiltonian not calculated properly."

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
