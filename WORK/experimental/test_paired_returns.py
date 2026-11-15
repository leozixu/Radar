"""Causal pairing and gap recovery regressions using synthetic raw observations."""

from types import SimpleNamespace
import unittest

import numpy as np

from paired_returns import MAX_GAP, PairReturns, link_features, measured_pairs


class CertainPair:
    def predict(self, features, num_threads):
        return np.full(len(features), .999)


def observations(circle, keys, *, azimuth=.4, amplitude=.5):
    return np.array([[azimuth, .20, .3, amplitude, .99, circle, keys[0]],
                     [(azimuth + .001) % 1, .201, .3005, amplitude + .010, .99, circle, keys[1]]])


class PairReturnsTests(unittest.TestCase):
    def setUp(self):
        self.pairing = PairReturns(CertainPair(), .985, .95)

    def test_empty_and_duplicate_timestamps_do_not_invent_a_pair(self):
        empty = np.empty((0, 7))
        self.assertEqual(measured_pairs(empty, np.empty(0)).shape, (0, 2))
        raw = observations(0, (1, 2))
        self.assertEqual(len(measured_pairs(raw, np.array([10, 10]))), 0)
        self.assertEqual(len(measured_pairs(raw, np.array([10, 12]))), 0)

    def test_raw_order_and_azimuth_wrap_preserve_signed_role(self):
        previous = observations(0, (1, 2), azimuth=.998, amplitude=.4)
        current = observations(1, (3, 4), azimuth=.002, amplitude=.6)[::-1]
        indices = measured_pairs(current, np.array([101, 100]))
        np.testing.assert_array_equal(indices, [[1, 0]])
        a, b, _ = link_features(previous[None], current[indices])
        np.testing.assert_array_equal(a, [0])
        np.testing.assert_array_equal(b, [0])
        self.assertEqual(len(link_features(previous[None], current[None])[0]), 0)

    def test_second_observation_starts_and_retains_both_identities(self):
        previous = observations(0, (1, 2))
        current = observations(1, (3, 4))
        state = {}
        frame = self.pairing.begin([], previous, np.array([0, 1]), 0, state)
        self.assertEqual(list(self.pairing.starts(frame, np.ones(2, bool))), [])
        self.pairing.remember(frame, previous, {}, 0, state)
        frame = self.pairing.begin([], current, np.array([10, 11]), 1, state)
        self.assertEqual(len(list(self.pairing.starts(frame, np.ones(2, bool)))), 1)
        self.pairing.remember(frame, current, {3: 30, 4: 40}, 1, state)
        self.assertEqual(self.pairing.retained(state), {30, 40})
        tracks = [SimpleNamespace(batch_id=40), SimpleNamespace(batch_id=30)]
        later = observations(1 + MAX_GAP, (5, 6), amplitude=.6)
        frame = self.pairing.begin(tracks, later[::-1], np.array([201, 200]), 1 + MAX_GAP, state)
        self.assertEqual([(row, col) for row, col, _ in frame.locked], [(1, 1), (0, 0)])
        stale = observations(2 + MAX_GAP, (7, 8))
        self.assertFalse(self.pairing.begin(tracks, stale, np.array([300, 301]), 2 + MAX_GAP, state).locked)
        self.assertFalse(self.pairing.begin(tracks, stale, np.array([300, 301]), 2 + MAX_GAP, {}).locked)

    def test_competing_pairs_cannot_lock_the_same_current_observation(self):
        raw = np.array([[.4, .2, .3, a, .99, 0, i] for i, a in enumerate((.40, .41, .42), 1)])
        times = np.arange(3)
        state = {}
        frame = self.pairing.begin([], raw, times, 0, state)
        self.pairing.remember(frame, raw, {1: 10, 2: 20, 3: 30}, 0, state)
        later = raw.copy()
        later[:, 5] = 1
        tracks = [SimpleNamespace(batch_id=ident) for ident in (10, 20, 30)]
        frame = self.pairing.begin(tracks, later, times + 10, 1, state)
        self.assertEqual(len(frame.locked), 2)
        self.assertEqual(len({col for _, col, _ in frame.locked}), 2)


if __name__ == "__main__":
    unittest.main()
