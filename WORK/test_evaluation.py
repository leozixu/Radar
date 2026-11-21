"""Scorer and data-boundary regressions, using synthetic rows or mocked opens."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import evaluate_holdout
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEVELOPMENT_PACKETS, MANUAL_FINAL_PACKETS, STAGE_ONE_PACKETS, packet_paths
import replay_validation


class ScoreTests(unittest.TestCase):
    def test_tracking_length_rewards_same_id_across_scan_gaps(self):
        labels = dict.fromkeys(range(1, 9), 1)
        circles = {point: point - 1 for point in labels}
        times = {point: circle * 5000 for point, circle in circles.items()}
        # The same track returns in two disjoint two-scan runs. Its four hits
        # cover four of the six truth observations after the initialization pair.
        result = evaluate_packet(labels, dict.fromkeys((3, 4, 7, 8), 10), circles, times)
        track = result["tracks"][0]
        self.assertEqual(track["longest_same_batch_consecutive_circle_matches"], 2)
        self.assertEqual(track["missed_segments"], 1)
        self.assertAlmostEqual(track["tracking_length"], 2/3)
        self.assertAlmostEqual(result["score"]["weighted_points"]["tracking_length"], 40/3)
        self.assertAlmostEqual(result["score"]["total"], 235/3)
        self.assertEqual(result["score"]["total_literal_max"], 85)

    def test_tracking_length_main_score_gives_no_credit_for_a_missed_track(self):
        labels = dict.fromkeys(range(1, 7), 1)
        circles = {point: point - 1 for point in labels}
        times = {point: circle * 5000 for point, circle in circles.items()}
        missed = evaluate_packet(labels, {}, circles, times)
        recovered = evaluate_packet(labels, dict.fromkeys(labels, 10), circles, times)
        self.assertEqual(missed["score"]["weighted_points"]["tracking_length"], 0)
        self.assertEqual(missed["score"]["total"], 10)  # Only the no-false-alarm term.
        self.assertEqual(missed["score"]["total_literal_max"], 30)  # Legacy diagnosis only.
        self.assertEqual(recovered["score"]["weighted_points"]["tracking_length"], 20)
        self.assertEqual(recovered["score"]["total"], 100)

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

    def test_frozen_replay_requires_an_explicit_phase(self):
        with patch.object(evaluate_holdout, "run_evaluation") as replay, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                evaluate_holdout.main(["--data-dir", "fake", "--report", "fake.json"])
            replay.assert_not_called()

    def test_manual_only_requests_final_packets_after_frozen_check(self):
        argv = ["--phase", "manual-final", "--data-dir", "fake", "--report", "fake.json"]
        with patch.object(evaluate_holdout, "verify_frozen_candidate", return_value="checksum") as verify, \
                patch.object(evaluate_holdout, "run_evaluation") as replay:
            self.assertEqual(evaluate_holdout.main(argv), 0)
            verify.assert_called_once_with(evaluate_holdout.FROZEN_MANIFEST)
            self.assertEqual(replay.call_args.args[1], tuple(range(141, 201)))
            self.assertEqual(replay.call_args.kwargs["allowed"], MANUAL_FINAL_PACKETS)
            self.assertEqual(replay.call_args.kwargs["engine"], "candidate")
        with patch.object(evaluate_holdout, "verify_frozen_candidate", side_effect=ValueError), \
                patch.object(evaluate_holdout, "run_evaluation") as replay, redirect_stderr(StringIO()):
            self.assertEqual(evaluate_holdout.main(argv), 1)
            replay.assert_not_called()

    def test_stage_one_uses_exact_packets_and_unrounded_cutoff(self):
        argv = ["--phase", "stage1", "--data-dir", "fake", "--report", "fake.json",
                "--manifest", "candidate.json"]
        for total, expected_exit in ((97.99999999999999, 2), (98.0, 0)):
            result = {"summary": {"score": {"total": total}}}
            with self.subTest(total=total), redirect_stdout(StringIO()) as console, \
                    patch.object(evaluate_holdout, "verify_frozen_candidate", return_value="checksum") as verify, \
                    patch.object(evaluate_holdout, "run_evaluation", return_value=result) as replay:
                self.assertEqual(evaluate_holdout.main(argv), expected_exit)
                self.assertIn(repr(total), console.getvalue())
                verify.assert_called_once_with(Path("candidate.json"))
                self.assertEqual(replay.call_args.args[1], tuple(range(101, 141)))
                self.assertEqual(replay.call_args.kwargs["allowed"], STAGE_ONE_PACKETS)
                self.assertEqual(replay.call_args.kwargs["report_metadata"], {
                    "evaluation_phase": "stage1", "minimum_score": 98.0, "frozen_manifest_sha256": "checksum",
                })

    def test_stage_one_reader_rejects_reserved_packet(self):
        with patch.object(Path, "open") as open_file:
            with self.assertRaises(ValueError):
                packet_paths(Path("fake"), (101, 140, 141), STAGE_ONE_PACKETS)
            open_file.assert_not_called()

    def test_incomplete_replay_cannot_return_a_score(self):
        with patch.object(replay_validation, "engine_metadata", return_value={}), \
                patch.object(replay_validation, "replay_packets", return_value=iter(())), \
                redirect_stdout(StringIO()), self.assertRaises(ValueError):
            replay_validation.run_evaluation(Path("fake"), (101, 102), jobs=1, engine="candidate",
                                             allowed=STAGE_ONE_PACKETS)


if __name__ == "__main__":
    unittest.main()
