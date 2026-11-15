"""Regression checks for source-order restoration and abnormal-time fallback."""

import unittest
from unittest.mock import Mock

import numpy as np

from raw_timing_detection import RawReturnGate, features


class RawTimingTests(unittest.TestCase):
    def test_neighbour_features_are_returned_in_original_source_order(self):
        positions = np.array([[.201, .4, .3], [.2, .4, .3], [.208, .4, .3]])
        values = features(positions, np.ones(3), np.array([301, 300, 309]))
        self.assertEqual(values[0, 5], 1)
        self.assertTrue(np.isnan(values[1, 5]))
        self.assertEqual(values[2, 5], 8)

    def test_repeated_timestamps_leave_points_for_existing_association(self):
        model = Mock()
        gate = RawReturnGate(model, .025)
        positions = np.ones((8, 3))
        np.testing.assert_array_equal(gate(positions, np.ones(8), np.zeros(8)), np.ones(8, dtype=bool))
        model.predict.assert_not_called()

    def test_aligned_independent_times_use_the_probability_floor(self):
        model = Mock()
        model.predict.return_value = np.array([.01, .2, .7, .8, .0, .5])
        times = np.arange(100, 106)
        azimuth = times / (5000 / 3)
        positions = np.column_stack((azimuth, np.ones(6), np.ones(6)))
        gate = RawReturnGate(model, .025)
        np.testing.assert_array_equal(gate(positions, np.ones(6), times), [False, True, True, True, False, True])


if __name__ == "__main__":
    unittest.main()
