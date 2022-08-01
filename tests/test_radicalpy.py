import unittest

import numpy as np
import src.radicalpy as rp


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

    # def test_molecule_prop(self):
    #     for prop in ["multiplicity"]:
    #         for i, h in enumerate(self.m.elemprop_generator(prop)):
    #             assert h == self.m.get_elemprop(i, prop)

    # @unittest.skip("Not complete")
    def test_dummy(self):
        # calculates HZ, HH
        sim = rp.Sim(
            [
                rp.Molecule("adenine", ["N6-H1", "N6-H2"]),
                # hfc1_custom=[0.5, 0.6], # mT
                # atom1_custom=["H", "N"]
                # or
                # multiplicity1_custom=[1, 3] # or spin1_custom?
                # gamma1_custom=[0.1, 0.4] ???
                # mT vs MHz!
                rp.Molecule("adenine", ["C8-H"]),
            ],
            kinetics=dict(model="Haberkorn", recombination=3e6, escape=1e6),
        )
        # SAIA, SAIB, SBIC

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
