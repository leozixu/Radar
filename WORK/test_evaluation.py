"""Scorer and data-boundary regressions, using synthetic rows or mocked opens."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import evaluate_holdout
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEVELOPMENT_PACKETS, HOLDOUT_PACKETS, packet_paths
import replay_validation


class ScoreTests(unittest.TestCase):
    def test_history_keeps_first_hit_and_merge_diagnostics_separate(self):
        labels = {**dict.fromkeys(range(1, 6), 1), **dict.fromkeys(range(6, 9), 2), 9: 0, 10: -1}
        circles = dict(zip(labels, (0, 1, 2, 3, 4, 0, 1, 2, 0, 3)))
        times = {point: circle * 5000 for point, circle in circles.items()}
        result = evaluate_packet(labels, {4: 10, 5: 11, 8: 10, 9: 10, 10: 10}, circles, times)
        self.assertAlmostEqual(result["score"]["total"], 66 + 5/6)
        self.assertAlmostEqual(result["score"]["total_first_hit"], 65 + 5/6)
        self.assertEqual(result["false_alarm_ratio"], 0.25)
        self.assertEqual(result["merged_output_tracks"], 1)
        self.assertEqual(result["merged_truth_points"], 1)
        self.assertAlmostEqual(result["idf1"], 4/13)
        aggregate = aggregate_packets([result, result])
        self.assertAlmostEqual(result["score"]["total"], aggregate["score"]["total"])
        self.assertEqual(aggregate["identity_true_positive_points"], 4)


class BoundaryTests(unittest.TestCase):
    def test_development_rejects_mixed_selection_before_open(self):
        with patch.object(Path, "open") as open_file, redirect_stderr(StringIO()):
            self.assertEqual(replay_validation.main(["--packets", "79", "101", "--jobs", "2"]), 1)
            open_file.assert_not_called()
        self.assertEqual(len(packet_paths(Path("fake"), DEVELOPMENT_PACKETS)), 100)

    def test_reader_guard_rejects_direct_holdout_path(self):
        with patch.object(Path, "open") as open_file:
            circles = replay_validation.iter_circle_payloads(Path("fake/train_101.csv"), {}, {})
            with self.assertRaises(ValueError):
                next(circles)
            open_file.assert_not_called()

    def test_link_to_out_of_scope_filename_is_rejected(self):
        with patch.object(Path, "resolve", return_value=Path("fake/train_101.csv")):
            with self.assertRaises(ValueError):
                packet_paths(Path("fake"), [1])

    def test_manual_only_requests_holdout_after_frozen_check(self):
        argv = ["--data-dir", "fake", "--report", "fake.json"]
        with patch.object(evaluate_holdout, "verify_frozen_candidate") as verify, \
                patch.object(evaluate_holdout, "run_evaluation") as replay:
            self.assertEqual(evaluate_holdout.main(argv), 0)
            verify.assert_called_once_with()
            self.assertEqual(replay.call_args.args[1], HOLDOUT_PACKETS)
            self.assertEqual(replay.call_args.kwargs["allowed"], HOLDOUT_PACKETS)
            self.assertEqual(replay.call_args.kwargs["engine"], "candidate")
        with patch.object(evaluate_holdout, "verify_frozen_candidate", side_effect=ValueError), \
                patch.object(evaluate_holdout, "run_evaluation") as replay, redirect_stderr(StringIO()):
            self.assertEqual(evaluate_holdout.main(argv), 1)
            replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
