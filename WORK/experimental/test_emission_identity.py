"""Confidence and ownership regressions for the public emission policy."""

import unittest

import numpy as np

from emission_identity import accepted_emissions


class EmissionPolicyTests(unittest.TestCase):
    def test_raw_sensor_evidence_still_requires_a_plausible_owner(self):
        result = accepted_emissions(
            np.zeros(4, dtype=bool), np.full(4, .15), np.array([.9, .1, .1, .9]),
            np.array([False, False, True, False]), (0., .25, 0.),
            np.array([.98, .98, .98, np.nan]), .97,
        )
        np.testing.assert_array_equal(result, [True, False, True, False])

    def test_detector_floor_applies_to_legacy_as_well_as_rescued_outputs(self):
        result = accepted_emissions(
            np.array([True, True, False]), np.array([.01, .2, .82]), np.full(3, .9),
            np.zeros(3, dtype=bool), (0., .65, .10),
        )
        np.testing.assert_array_equal(result, [False, True, True])


if __name__ == "__main__":
    unittest.main()
