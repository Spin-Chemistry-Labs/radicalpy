import unittest

import numpy as np
from src.radicalpy import Sim


class DummyTests(unittest.TestCase):
    def setUp(self):
        self.spin_halves = 3
        self.spin_ones = 2
        self.sim = Sim(self.spin_halves, self.spin_ones)

    def test_shared_sigmas(self):
        self.sim.sigmas[0][0, 1, 1] = 42
        assert self.sim.sigmas[2][0, 1, 1] == 42
