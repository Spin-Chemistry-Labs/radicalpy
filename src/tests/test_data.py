#! /usr/bin/env python

import doctest
import unittest

from src.radicalpy import data


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(data))
    return tests


class IsotopeTestCase(unittest.TestCase):
    """Test case for the `Isotope` class."""

    def test_number_of_isotopes(self):
        previous_number = 293
        current_number = len(data.Isotope.available())
        self.assertEqual(current_number, previous_number)

    def test_all_isotope_jsons(self):
        """Test loading of all isotopes."""
        for isotope in data.Isotope.available():
            with self.subTest(isotope):
                data.Isotope(isotope)


class MoleculeTestCase(unittest.TestCase):
    """Test case for the `Molecule` class."""

    def test_number_of_molecules(self):
        previous_number = 6
        current_number = len(data.Molecule.available())
        self.assertEqual(current_number, previous_number)
