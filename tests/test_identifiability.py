#! /usr/bin/env python

import doctest
import unittest

import numpy as np

from radicalpy import identifiability


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(identifiability))
    return tests


class IdentifiabilityTests(unittest.TestCase):
    """Cases that do not read naturally as doctests."""

    def setUp(self):
        self.times = np.linspace(0.0, 1.0, 50)

    def _sum_only(self, params):
        """Depends on the parameters only through their sum: exactly degenerate."""
        return np.exp(-(params[0] + params[1]) * self.times)

    def test_degeneracy_detected_across_magnitude_ratios(self):
        """Detection must not depend on how the parameters are scaled.

        The default RELATIVE_STEP exists because of this case: unequal
        magnitudes give unequal absolute steps, so the identical Jacobian
        columns differ at O(h^2) instead of cancelling. Too coarse a step lifts
        that floor above DEGENERACY_TOLERANCE and the degeneracy is missed.
        """
        for second in (2.0, 7.0, 13.0, 250.0):
            with self.subTest(ratio=second):
                report = identifiability.analyze_model(
                    self._sum_only, np.array([1.0, second]), ("k_a", "k_b")
                )
                self.assertEqual(report.rank, 1)
                self.assertTrue(report.is_degenerate)

    def test_coarse_step_reproduces_the_historical_false_negative(self):
        """Guards the step-size choice: 1e-4 misses the degeneracy, 1e-5 finds it."""
        coarse = identifiability.analyze_model(
            self._sum_only, np.array([1.0, 2.0]), ("k_a", "k_b"), rel_step=1e-4
        )
        self.assertEqual(coarse.rank, 2)

        shipped = identifiability.analyze_model(
            self._sum_only, np.array([1.0, 2.0]), ("k_a", "k_b")
        )
        self.assertEqual(shipped.rank, 1)

    def test_underdetermined_jacobian_still_reports_null_directions(self):
        """Fewer observables than parameters must not silently drop the null space.

        A 1x2 sensitivity matrix has a genuine second singular value of zero;
        the economy-size SVD would omit its right singular vector.
        """
        report = identifiability.analyze_model(
            lambda p: np.array([p[0] * p[1]]), np.array([2.0, 3.0]), ("a", "b")
        )
        self.assertEqual(report.rank, 1)
        self.assertEqual(len(report.degenerate_directions), 1)

    def test_log_sensitivity_vanishes_for_an_unmeasurably_slow_rate(self):
        """A rate far slower than the observation window carries no information.

        The absolute Jacobian does not show this, because dy/dk saturates as
        k -> 0. The log-sensitivity does.
        """

        def decay(params):
            return np.exp(-params[1] * self.times) * params[0]

        fast = identifiability.analyze_model(
            decay, np.array([1.0, 1.0]), ("A", "k"), log_space=True
        )
        slow = identifiability.analyze_model(
            decay, np.array([1.0, 1e-8]), ("A", "k"), log_space=True
        )
        self.assertGreater(fast.singular_value_ratios[1], 1e-3)
        self.assertLess(slow.singular_value_ratios[1], 1e-6)

    def test_mismatched_parameter_names_are_rejected(self):
        with self.assertRaises(ValueError):
            identifiability.analyze_jacobian(np.zeros((5, 3)), ("only", "two"))

    def test_log_sensitivity_requires_positive_parameters(self):
        with self.assertRaises(ValueError):
            identifiability.log_sensitivity_jacobian(lambda p: p, np.array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
