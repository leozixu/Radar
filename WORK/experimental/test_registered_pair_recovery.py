"""Registered-pair continuation preserves normal reservations and 1:1 ownership."""

from types import SimpleNamespace
import unittest

import numpy as np

from paired_returns import ObservedPair, PairScan
from registered_pair_recovery import compatible_recoveries


def measurements(azimuth, circle, first_point):
    return np.array([[azimuth, .20, .37, .44, .90, circle, first_point],
                     [(azimuth + .001) % 1, .201, .371, .421, .90, circle, first_point + 1]])


class TestRegisteredPairRecovery(unittest.TestCase):
    def setUp(self):
        past = measurements(.97, 8, 100)
        self.old = ObservedPair(past, (10, 20), True)
        self.current = measurements(.04, 9, 102)
        self.tracks = [SimpleNamespace(batch_id=batch, position=row[:3], velocity=np.array([.07, 0., 0.]),
                                       circle=8)
                       for batch, row in zip((10, 20), past)]
        self.frame = PairScan(np.array([[0, 1]]), [], [], strict_count=1)

    def test_recovers_wrapped_large_motion_using_original_order(self):
        recovery = compatible_recoveries(self.frame, self.tracks, self.current, 9, [self.old])
        self.assertEqual(len(recovery), 1)
        self.assertIs(recovery[0][0], self.old)
        self.assertEqual(recovery[0][1:], (0, (0, 1)))

    def test_multiple_wide_column_options_are_left_unassigned(self):
        ambiguous = np.concatenate((self.current, measurements(.041, 9, 104)))
        frame = PairScan(np.array([[0, 1], [2, 3]]), [], [], strict_count=2)
        self.assertEqual(compatible_recoveries(frame, self.tracks, ambiguous, 9, [self.old]), [])

    def test_skips_an_arm_already_reserved_by_original_locks(self):
        frame = PairScan(np.array([[0, 1]]), [], [(0, 3, .99)], strict_count=1)
        self.assertEqual(compatible_recoveries(frame, self.tracks, self.current, 9, [self.old]), [])

    def test_unregistered_or_inconsistent_shape_cannot_recover(self):
        unregistered = ObservedPair(self.old.observations, self.old.owners, False)
        self.assertEqual(compatible_recoveries(self.frame, self.tracks, self.current, 9, [unregistered]), [])
        self.current[1, 3] += .0002
        self.assertEqual(compatible_recoveries(self.frame, self.tracks, self.current, 9, [self.old]), [])


if __name__ == "__main__":
    unittest.main()
