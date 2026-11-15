"""Exercise deferred startup alternatives, skipped scans and consumed observations."""

import unittest

import numpy as np

from causal_birth_paths import Arcs, FEATURES, POINT, PathPool, find_stranded_owner, supported_two_hit_pairs
import teacher_assoc as teacher


def points(circle, observations):
    return np.array([[azimuth, .25, .40, .50, .95, circle, ident]
                     for ident, azimuth in observations], dtype=float).reshape(-1, 7)


class CausalBirthPathsTests(unittest.TestCase):
    def test_different_parents_remain_available_until_the_third_observation(self):
        pool = PathPool()
        first = points(0, [(10, .495), (11, .498)])
        second = points(1, [(12, .500)])
        third = points(3, [(13, .510)])
        pool.add(first, 0)
        self.assertIsNone(pool.propose(second, 1))
        pool.add(second, 1)

        head, middle, cols, features = pool.propose(third, 3)
        self.assertEqual(set(head[:, POINT]), {10, 11})
        self.assertEqual(set(middle[:, POINT]), {12})
        self.assertEqual(set(third[cols, POINT]), {13})
        self.assertEqual(features.shape, (2, len(FEATURES)))
        np.testing.assert_array_equal(features[:, 1], 2.)

    def test_confirmation_removes_alternatives_sharing_consumed_observations(self):
        pool = PathPool()
        pool.add(points(0, [(10, .495), (11, .498)]), 0)
        pool.add(points(1, [(12, .500)]), 1)
        now = points(2, [(13, .505)])
        self.assertIsNotNone(pool.propose(now, 2))
        pool.add(points(2, []), 2, np.array([10, 12, 13]))
        self.assertIsNone(pool.propose(now, 2))

    def test_two_hit_pair_requires_distinct_points_and_stable_relative_elevation(self):
        first = points(0, [(10, .495), (11, .498)])
        first[:, 1] = [.24, .25]
        latest = points(1, [(12, .500), (13, .503)])
        latest[:, 1] = [.26, .27]
        arcs = Arcs(first, latest, np.array([.1, .1]))
        self.assertEqual(supported_two_hit_pairs(arcs), [(0, 1)])
        latest[1, 1] += .02
        self.assertEqual(supported_two_hit_pairs(arcs), [])
        latest[1] = latest[0]
        self.assertEqual(supported_two_hit_pairs(arcs), [])

    def test_reanchor_requires_a_unique_compatible_coasting_track(self):
        track = teacher.spawn(np.array([.49, .25, .4]), .5, 0)
        rival = teacher.spawn(np.array([.49, .25, .4]), .5, 0)
        for time in range(1, 6):
            for candidate in (track, rival):
                teacher.update(candidate, np.array([.49 + time*.002, .25, .4]), .5, time)
        path = tuple(points(time, [(10 + time, .49 + time*.002)])[0] for time in (6, 7, 8))
        self.assertIs(find_stranded_owner([track], path), track)
        self.assertIsNone(find_stranded_owner([track, rival], path))


if __name__ == '__main__':
    unittest.main()
