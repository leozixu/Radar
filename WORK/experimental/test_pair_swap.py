"""Regression checks for the scope of live reciprocal corrections."""

import unittest
from unittest.mock import patch

import numpy as np

from pair_swap_probe import FEATURES, PairCorrector, make_features, spawn, update


class Predictor:
    def __init__(self, output):
        self.output = np.asarray(output)

    def predict(self, features, num_threads):
        if len(features) != len(self.output):
            raise AssertionError('unexpected number of evaluated candidates')
        return self.output


class PairSwapTests(unittest.TestCase):
    def setUp(self):
        self.tracks = [spawn(np.array([.50, el, .3]), .5, 0) for el in (.25, .26)]
        for circle in range(1, 5):
            for index, track in enumerate(self.tracks):
                update(track, np.array([.50 + .001*circle, .25 + .01*index + .002*circle, .3]), .5, circle)
        self.positions = np.array([[.505, .26, .3], [.505, .27, .3]])
        self.rows = np.array([0, 0, 1, 1])
        self.cols = np.array([0, 1, 0, 1])
        self.features = np.zeros((4, len(FEATURES)))

    def test_pair_features_compare_both_orientations_in_the_same_coordinates(self):
        lookup = np.array([[0, 1], [2, 3]])
        scores = np.log([.9, .4, .3, .85])
        direct, crossed, difference = make_features(self.tracks, self.positions, self.features, lookup,
                                                     scores, 0, 1, 0, 1, 5)
        flip, flip_cross, flip_difference = make_features(self.tracks, self.positions, self.features,
                                                           lookup, scores, 0, 1, 1, 0, 5)
        np.testing.assert_allclose(direct, flip_cross)
        np.testing.assert_allclose(crossed, flip)
        self.assertAlmostEqual(difference, -flip_difference)

    def test_swap_preserves_one_to_one_and_requires_a_plausible_cross_edge(self):
        with patch('pair_swap_probe.candidate_features', return_value=(self.rows, self.cols, self.features)):
            corrector = PairCorrector(Predictor([.01, .99]), cutoff=.85)
            matches, held = corrector.associate(self.tracks, self.positions, [.5, .5], 5,
                                                Predictor([.95, .4, .3, .9]), .2, 0., .30)
            self.assertEqual({(r, c) for r, c, _ in matches}, {(0, 1), (1, 0)})
            self.assertEqual(list(held), [])
            self.assertEqual(corrector.swaps, 1)
            matches, _ = corrector.associate(self.tracks, self.positions, [.5, .5], 5,
                                              Predictor([.95, .04, .3, .9]), .2, 0., .30)
            self.assertEqual({(r, c) for r, c, _ in matches}, {(0, 0), (1, 1)})
            self.assertEqual(corrector.swaps, 1)


if __name__ == '__main__':
    unittest.main()
