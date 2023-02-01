#! /usr/bin/env python

import doctest

from radicalpy import relaxation


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(relaxation))
    return tests


# class RelaxationsTests(unittest.TestCase):
#     def test_constants(self):
#         self.assertEqual(constants.mu_0, 1.25663706212e-06)
