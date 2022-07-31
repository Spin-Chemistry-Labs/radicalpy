import unittest

import numpy as np
import src.radicalpy as rp


class DummyTests(unittest.TestCase):
    def setUp(self):
        self.spin_halves = 3
        self.spin_ones = 2
        self.sim = rp.Sim(self.spin_halves, self.spin_ones)

    def test_shared_sigmas(self):
        self.sim.sigmas[0][0, 1, 1] = 42
        assert self.sim.sigmas[2][0, 1, 1] == 42

    def test_dummy(self):
        # calculates HZ, HH
        sim = rp.Sim(
            # instead of
            rad1="Adenine",
            hfc1=["N6-H1", "N6-H2"],
            # hfc1_custom=[0.5, 0.6], # mT
            # atom1_custom=["H", "N"]
            # or
            # multiplicity1_custom=[1, 3] # or spin1_custom?
            # gamma1_custom=[0.1, 0.4] ???
            # mT vs MHz!
            rad2="FAD",
            hfc2=["N5"],
            kinetics=dict(model="Haberkorn", recombination=3e6, escape=1e6),
        )
        # SAIA, SAIB, SBIC

        sim.hyperfine()

        # creates K "Hamiltonian"
        sim.kinetics(model="Haberkorn", recombination=3e6, escape=1e6)
        # creates Relaxation "Hamiltonian"
        sim.relaxation(model="STD", rate=1e6)
        # create HJ
        sim.J_coupling(0.0)
        # create HD
        sim.dipolar_coupling(0.0)

        sim.time_evolution(time=np.linspace())
        sim.mary(time=np.linspace(), magnetic_field=np.linspace())
        sim.angle(time=np.linspace(), theta=np.linspace(), phi=np.linspace())
