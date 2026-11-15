"""Regression checks for causal ID recovery."""

import unittest

import numpy as np

from identity_rejoin import IdentityRuntime


class MotionEvidence:
    def predict(self, features, num_threads):
        error = np.linalg.norm(features[:, 10:13], axis=1)
        return np.where(error < .50, .99, .02)


class UncertainEvidence:
    def predict(self, features, num_threads):
        return np.full(len(features), .50)


def known_points(scan, source_ids):
    position = np.array([
        [(.993 + .004 * scan) % 1, .30, .30 + .01 * scan],
        [.95 - .003 * scan, .23, .60 - .007 * scan],
    ])[:len(source_ids)]
    velocity = np.array([[.004, 0, .01], [-.003, 0, -.007]])[:len(source_ids)]
    n = len(source_ids)
    proposals = [{"BatchID": bid, "PntID": 2 * scan + i} for i, bid in enumerate(source_ids)]
    return proposals, (position, np.full(n, .5), np.array(source_ids), np.full(n, 10), velocity, np.ones(n))


class IdentityRejoinTest(unittest.TestCase):
    def test_two_observations_predict_across_wrap_gap_and_source_id_swap(self):
        runtime = IdentityRuntime(None, MotionEvidence())
        state = {}
        for scan, source_ids in ((0, (17, 18)), (1, (17, 18)), (5, (18, 17)), (6, (19, 20))):
            proposals, arrays = known_points(scan, source_ids)
            result = runtime.associate(scan, proposals, arrays, state)
            self.assertEqual([p["PntID"] for p in result], [p["PntID"] for p in proposals])
            self.assertEqual([p["BatchID"] for p in result], [1, 2])

    def test_weak_source_id_handoff_keeps_latent_track_without_emission(self):
        runtime = IdentityRuntime(None, UncertainEvidence(), weak_cross_probability=.64)
        state = {}
        for scan, raw, expected in ((0, 17, [1]), (1, 17, [1]), (2, 18, []), (3, 18, [1])):
            proposals, arrays = known_points(scan, (raw,))
            result = runtime.associate(scan, proposals, arrays, state)
            self.assertEqual([p["BatchID"] for p in result], expected)

    def test_dormant_stitch_separates_aliases_when_both_return(self):
        runtime = IdentityRuntime(None, MotionEvidence(), match_probability=.65, lineage_min_gap=4)
        state = {}
        for scan in range(4):
            proposals, arrays = known_points(scan, (17,))
            result = runtime.associate(scan, proposals, arrays, state)
            self.assertEqual([p["BatchID"] for p in result], [1])
        proposals, arrays = known_points(8, (19,))
        self.assertEqual([p["BatchID"] for p in runtime.associate(8, proposals, arrays, state)], [1])
        proposals, arrays = known_points(9, (19, 17))
        self.assertEqual([p["BatchID"] for p in runtime.associate(9, proposals, arrays, state)], [1, 2])


if __name__ == "__main__":
    unittest.main()
