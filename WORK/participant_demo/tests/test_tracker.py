"""Regression tests for packet-local identity and causal radar output."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracker import TrackerSettings, process_candidates


def detection(circle, pid, azimuth, elevation, distance, timestamp_shift=0):
    turn = azimuth % 1.0
    return {"Time": int(circle * 5000 + turn * 5000) + timestamp_shift,
            "Azimuth": turn, "Elevation": elevation, "Distance": distance,
            "Amp": 0.66, "CircleID": circle, "PntID": pid}


class TrackerTests(unittest.TestCase):
    def test_crossing_tracks_keep_identity_when_order_and_point_numbers_change(self):
        settings = TrackerSettings()
        normal_state, shuffled_state = {}, {}
        expected_ids = None
        for circle in range(9):
            points = [detection(circle, 2 * circle + 1, 0.960 + 0.009 * circle, 0.47, 0.6 + 0.001 * circle),
                      detection(circle, 2 * circle + 2, 0.041 - 0.009 * circle, 0.52, 0.611 - 0.001 * circle)]
            result = process_candidates(points, normal_state, settings)
            reversed_points = [dict(point, PntID=1000 - point["PntID"]) for point in reversed(points)]
            shuffled = process_candidates(reversed_points, shuffled_state, settings)
            ids = [row["BatchID"] for row in result]
            self.assertEqual(ids, [row["BatchID"] for row in reversed(shuffled)])
            if circle < 2:
                self.assertEqual(result, [])
            else:
                self.assertEqual([row["PntID"] for row in result], [point["PntID"] for point in points])
                if expected_ids is None:
                    self.assertEqual(len(set(ids)), 2)
                    expected_ids = ids
                self.assertEqual(ids, expected_ids)

    def test_wrapped_track_recovers_same_id_after_empty_scans_and_time_anomaly(self):
        settings, state = TrackerSettings(), {}
        expected_id = None
        for circle in range(11):
            bad_stamp = 500000 if circle == 10 else 0
            point = detection(circle, circle + 1, 0.987 + 0.007 * circle, 0.53,
                              0.45 + 0.001 * circle, bad_stamp)
            result = process_candidates([point] if circle <= 3 or circle == 10 else [], state, settings)
            if circle in (2, 3, 10):
                self.assertEqual(len(result), 1)
                row = result[0]
                self.assertEqual({name: row[name] for name in ("Time", "Azimuth", "CircleID", "PntID")},
                                 {name: point[name] for name in ("Time", "Azimuth", "CircleID", "PntID")})
                if expected_id is None:
                    expected_id = row["BatchID"]
                self.assertEqual(row["BatchID"], expected_id)
            else:
                self.assertEqual(result, [])
        self.assertEqual(process_candidates([detection(0, 1, 0.20, 0.40, 0.50)], {}, settings), [])

    def test_segment_rejoin_withholds_output_until_three_returning_observations(self):
        settings = TrackerSettings(segment_recovery=True, recovery_margin=10.0)
        state = {}
        reports = []
        for circle in range(12):
            point = detection(circle, circle + 1, 0.987 + 0.007 * circle, 0.53, 0.45 + 0.001 * circle)
            reports.extend(process_candidates([point] if circle < 4 or circle >= 9 else [], state, settings))
        self.assertEqual([point["CircleID"] for point in reports], [2, 3, 11])
        self.assertEqual(len({point["BatchID"] for point in reports}), 1)


if __name__ == "__main__":
    unittest.main()
