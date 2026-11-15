"""Causal feature invariance and compatibility with the original CV decisions."""

import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

import replay_validation
from tracker_compat import CompatibilityRuntime, decisions_with_features
import train_compat_quality


def scan(circle, offset=0.):
    if circle == 4:
        return []
    return [{
        "Time": circle * 5000 + int(offset * 1000), "CircleID": circle,
        "PntID": circle + int(offset * 1000), "Azimuth": (0.988 + circle * 0.007 + offset) % 1,
        "Elevation": 0.21 + circle * 0.001 + offset, "Distance": 0.32 + circle * 0.001 + offset,
        "Amp": 0.48, "SNR": 0.48,
    }]


class TestCompatibilityQuality(unittest.TestCase):
    def test_traces_keep_baseline_decisions_and_use_only_relative_features(self):
        baseline = replay_validation.load_handler("baseline", None)
        base_state, compat_state, shifted_state = {}, {}, {}
        outputs = 0
        for circle in range(9):
            points = scan(circle)
            records, features = decisions_with_features(points, compat_state)
            shifted, shifted_features = decisions_with_features(scan(circle, .13), shifted_state)
            self.assertEqual(records, baseline(points, base_state))
            self.assertEqual([r["BatchID"] for r in records], [r["BatchID"] for r in shifted])
            np.testing.assert_allclose(features, shifted_features, atol=1.e-6)
            outputs += len(records)
        self.assertGreater(outputs, 4)

    def test_suppressed_output_does_not_destroy_internal_identity(self):
        class Predictor:
            probability = 0.0

            def predict(self, features, num_threads):
                return np.full(len(features), self.probability)

        model = Predictor()
        runtime = CompatibilityRuntime(model, 0.2, {})
        state = {}
        for circle in range(4):
            self.assertEqual(runtime.process_circle(scan(circle), state), [])
        model.probability = 1.0
        self.assertEqual(runtime.process_circle(scan(4), state), [])
        self.assertEqual(runtime.process_circle(scan(5), state)[0]["BatchID"], 1)

    def test_training_rejects_tuning_before_open(self):
        with patch.object(Path, "open") as open_file, self.assertRaises(SystemExit):
            train_compat_quality.main(["--packets", "1", "71"])
        open_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
