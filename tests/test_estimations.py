#! /usr/bin/env python

import unittest

import radicalpy as rp


class EstimationsTests(unittest.TestCase):
    def test_Bhalf_theoretical(self):
        flavin = rp.simulation.Molecule.fromdb("flavin_anion")
        trp = rp.simulation.Molecule.fromdb("tryptophan_cation")
        sim = rp.simulation.HilbertSimulation([flavin, trp])
        Bhalf_theoretical = rp.estimations.Bhalf_theoretical(sim)
        self.assertAlmostEqual(Bhalf_theoretical, 2.9692816566569937, places=2)

    def test_T1_relaxation_rate(self):
        gold = 557760.0907618533
        g = [2.00429, 2.00389, 2.00216]
        B = 0.05
        tau_c = 1.3006195732809966e-07
        t1 = rp.estimations.T1_relaxation_rate(g, B, tau_c)
        self.assertAlmostEqual(gold, t1)

    def test_T2_relaxation_rate(self):
        gold = 1138300.4959382531
        g = [2.00429, 2.00389, 2.00216]
        B = 0.05
        tau_c = 1.3006195732809966e-07
        t2 = rp.estimations.T2_relaxation_rate(g, B, tau_c)
        self.assertAlmostEqual(gold, t2)

    def test_aqueous_glycerol_viscosity(self):
        gold = 0.0032191882387078096
        viscosity = rp.estimations.aqueous_glycerol_viscosity(0.2, 5)
        self.assertAlmostEqual(gold, viscosity)

    def test_diffusion_coefficient(self):
        gold = 6.236355316540426e-10
        radius = 3.5e-10
        temperature = 298
        eta = 1e-3
        diff = rp.estimations.diffusion_coefficient(radius, temperature, eta)
        self.assertAlmostEqual(gold, diff)

    def test_g_tensor_relaxation_rate(self):
        gold = 2138022072021.6843
        tau_c = 5e-12
        g1 = [2.0032, 1.9975, 2.0014]
        g2 = [2.00429, 2.00389, 2.00216]
        k_g = rp.estimations.g_tensor_relaxation_rate(tau_c, g1, g2)
        self.assertAlmostEqual(gold, k_g)

    def test_k_STD_microreactor(self):
        gold = 21131271.24997083
        D = 5e-11
        V = 2e-26
        k_std = rp.estimations.k_STD_microreactor(D, V)
        self.assertAlmostEqual(gold, k_std)

    def test_k_ST_mixing(self):
        gold = 7.277263377794495e07
        Bhalf = 2.5967338498081585
        k_ST = rp.estimations.k_ST_mixing(Bhalf=Bhalf)
        scale = 1e-7
        self.assertAlmostEqual(gold * scale, k_ST * scale, places=3)

    def test_k_electron_transfer(self):
        gold = 15135612.484362071
        R = 13.3
        k_et = rp.estimations.k_electron_transfer(R)
        self.assertAlmostEqual(gold, k_et)

    def test_k_excitation(self):
        gold = 52749.44112741747
        P = 290e-6
        wl = 450e-9
        V = 0.54e-15
        l = 900e-9
        epsilon = 12600
        kI = rp.estimations.k_excitation(P, wl, V, l, epsilon)
        self.assertAlmostEqual(gold, kI)

    def test_k_recombination(self):
        gold = 14161120.442179158
        MFE = 0.35
        k_escape = 5.3e6
        k_rec = rp.estimations.k_recombination(MFE, k_escape)
        self.assertAlmostEqual(gold, k_rec)

    def test_k_reencounter(self):
        gold = 623635531.6540426
        encounter_dist = 10e-10
        diff_coeff = 6.236355316540426e-10
        k_reen = rp.estimations.k_reencounter(encounter_dist, diff_coeff)
        self.assertAlmostEqual(gold, k_reen)

    def test_k_triplet_relaxation(self):
        gold = 187966399.9469546
        D = 1.98e9
        E = 0.56e9
        B0 = 0.2e-3
        tau_c = 58e-12
        k_tr = rp.estimations.k_triplet_relaxation(B0, tau_c, D, E)
        self.assertAlmostEqual(gold, k_tr, places=5)

    def test_rotational_correlation_time_for_molecule(self):
        gold = 5.5745926978864795e-11
        a = 0.4e-9
        T = 310
        tau_c = rp.estimations.rotational_correlation_time_for_molecule(a, T)
        self.assertAlmostEqual(gold, tau_c)

    def test_rotational_correlation_time_for_protein(self):
        gold = 3.595786684051539e-08
        tau_c = rp.estimations.rotational_correlation_time_for_protein(61, 278)
        self.assertAlmostEqual(gold, tau_c)
