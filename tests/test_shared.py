#! /usr/bin/env python

import doctest
import unittest

from radicalpy import shared


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(shared))
    return tests


class ConstantTestCase(unittest.TestCase):
    """Test case for the `Constant` class."""

    def test_number_of_constants(self):
        """Check the number of constants to the previous state."""
        previous_number_of_constants = 21
        current_number_of_constants = len(vars(shared.constants))
        self.assertEqual(current_number_of_constants, previous_number_of_constants)

    def test_constants(self):
        """Test the values of all the constants."""
        C = shared.constants
        self.assertEqual(C.c, 299792458.0)
        self.assertEqual(C.h, 6.6260693e-34)
        self.assertEqual(C.epsilon_0, 8.854187817e-12)
        self.assertEqual(C.hbar, 1.05457168e-34)
        self.assertEqual(C.e, 1.60217653e-19)
        self.assertEqual(C.alpha, 0.007297352568)
        self.assertEqual(C.mu_0, 1.25663706212e-06)
        self.assertEqual(C.m_e, 9.1093826e-31)
        self.assertEqual(C.m_p, 1.67262171e-27)
        self.assertEqual(C.mu_B, 9.27400949e-24)
        self.assertEqual(C.mu_N, 5.05078343e-27)
        self.assertEqual(C.mu_e, -9.28476412e-24)
        self.assertEqual(C.a_e, 0.0011596521859)
        self.assertEqual(C.g_e, -2.0023193043718)
        self.assertEqual(C.mu_p, 1.41060671e-26)
        self.assertEqual(C.g_p, 5.5856946893)
        self.assertEqual(C.N_A, 6.0221415e23)
        self.assertEqual(C.R, 8.314472)
        self.assertEqual(C.k_B, 1.3806505e-23)
        self.assertEqual(C.V, 0.00073)
        self.assertEqual(C.rw, 1.6e-10)
