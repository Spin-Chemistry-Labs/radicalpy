#! /usr/bin/env python

import unittest

from src import radicalpy as rp


class HilbertTests(unittest.TestCase):
    def test_rotational_correlation_time_protein(self):
        Mr = 122  # 61
        temp = 310
        t = rp.utils.rotational_correlation_time_protein(Mr, temp)
        gold = 60.65e-9  # 31.93e-9
        self.assertAlmostEqual(t, gold)
