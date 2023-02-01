#! /usr/bin/env python

import doctest
import unittest

import numpy as np
import numpy.testing
from radicalpy import data


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(data))
    return tests


class IsotopeTestCase(unittest.TestCase):
    """Test case for the `Isotope` class."""

    def test_number_of_isotopes(self):
        """Number of isotopes.

        Changes when:

        - multiplicity and gyromagnetic ratio change;
        - isotopes are added or removed from the database.
        """
        previous_number = 293
        available = data.Isotope.available()
        self.assertEqual(available[:3], ["G", "E", "N"])
        current_number = len(available)
        self.assertEqual(current_number, previous_number)

    def test_all_isotope_jsons(self):
        """Load all isotopes."""
        for isotope in data.Isotope.available():
            with self.subTest(isotope):
                data.Isotope(isotope)

    def test_members(self):
        """Test isotope members and methods."""
        iso = data.Isotope("14N")
        self.assertEqual(iso.details, {"source": "NMR Enc. 1996"})
        self.assertEqual(iso.magnetogyric_ratio, 19337792.0)
        self.assertEqual(iso.magnetogyric_ratio, 1000 * iso.gamma_mT)
        self.assertEqual(iso.multiplicity, 3)
        self.assertEqual(iso.spin_quantum_number, 1)

    def test_constructors(self):
        """Test construction of existing and non-existing Isotope."""
        self.assertIsInstance(data.Isotope("15N"), data.Isotope)
        self.assertRaises(ValueError, data.Isotope, "Kryp")


class HfcTestCase(unittest.TestCase):
    """Test case for the `Hfc` class."""

    def setUp(self):
        """Setup for each test."""
        self.isotropic = 42.0
        self.anisotropic = [[float(i * 3 + j + 1) for j in range(3)] for i in range(3)]

    def test_1d_isotropic(self):
        """Query isotropic when constructed from float."""
        h = data.Hfc(self.isotropic)
        self.assertEqual(h.isotropic, self.isotropic)

    def test_1d_anisotropic(self):
        """Query anisotropic when constructed from float."""
        h = data.Hfc(self.isotropic)
        self.assertRaises(ValueError, lambda: h.anisotropic)

    def test_3d_isotropic(self):
        """Query isotropic when constructed from matrix."""
        h = data.Hfc(self.anisotropic)
        expected = np.trace(self.anisotropic) / 3
        self.assertEqual(h.isotropic, expected)

    def test_3d_anisotropic(self):
        """Query anisotropic when constructed from matrix."""
        h = data.Hfc(self.anisotropic)
        np.testing.assert_almost_equal(h.anisotropic, self.anisotropic)

    def test_3d_bad_constructor(self):
        """Bad ways to create HFCs."""
        self.assertRaises(ValueError, data.Hfc, [[1.0, 2.0], [3.0, 4.0]])
        self.assertRaises(ValueError, data.Hfc, 42)


class MoleculeTestCase(unittest.TestCase):
    """Test case for the `Molecule` class."""

    def test_number_of_molecules(self):
        previous_number = 6
        current_number = len(data.Molecule.available())
        self.assertEqual(current_number, previous_number)
