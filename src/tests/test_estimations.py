#! /usr/bin/env python

import unittest

from src import radicalpy as rp


class EstimationsTests(unittest.TestCase):
    def test_Bhalf_theoretical(self):
        flavin = rp.simulation.Molecule("flavin_anion")
        trp = rp.simulation.Molecule("trp_cation")
        sim = rp.simulation.HilbertSimulation([flavin, trp])
        Bhalf_theoretical = rp.estimations.Bhalf_theoretical(sim)
        self.assertAlmostEqual(Bhalf_theoretical, 2.4663924080289092)

    def test_rotational_correlation_time_protein(self):
        gold = 60.65e-9  # 31.93e-9
        Mr = 122  # 61
        temp = 310
        t = rp.estimations.rotational_correlation_time_protein(Mr, temp)
        self.assertAlmostEqual(gold, t)

    def test_k_ST_mixing(self):
        gold = 7.277263377794495e07
        Bhalf = 2.5967338498081585
        k_ST = rp.estimations.k_ST_mixing(Bhalf=Bhalf)
        scale = 1e-7
        self.assertAlmostEqual(gold * scale, k_ST * scale, places=3)
