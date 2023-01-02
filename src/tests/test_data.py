#! /usr/bin/env python

import unittest

from src.radicalpy.data import constants


class DataTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(constants.mu_0, 1.25663706212e-06)
