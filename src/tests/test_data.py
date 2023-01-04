#! /usr/bin/env python

import unittest

from src.radicalpy.data import constants as C


class DataTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(C.mu_0, 1.25663706212e-06)
