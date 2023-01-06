#! /usr/bin/env python

import unittest

from ..radicalpy.data import constants as C


class DataTests(unittest.TestCase):
    def test_number_of_constants(self):
        previous_number_of_tests = 21
        self.assertEqual(len(vars(C)), previous_number_of_tests)

    def test_constants(self):
        self.assertEqual(C.mu_0, 1.25663706212e-06)
