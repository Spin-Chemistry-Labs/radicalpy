import unittest

import numpy as np
from src.radicalpy import Sim


class DummyTests(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.sim = Sim(self.n)

    def test_succeed(self):
        assert np.all(self.sim.data == np.ones(self.n))
