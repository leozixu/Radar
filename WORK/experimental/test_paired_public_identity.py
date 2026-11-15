"""Public IDs stay stable when a receiver hands two mature tracks to opposite actors."""

import unittest

from paired_public_identity import PublicPairGuard


def observations(circle, exchanged=False, time_offset=1):
    left = {"PntID": 2 * circle, "Time": 5000 * circle, "CircleID": circle,
            "Azimuth": .29 + .0007 * circle, "Elevation": .2 - .012 * (circle % 2),
            "Distance": .4 + .001 * circle, "Amp": .37 + .016 * (circle % 3)}
    right = left | {"PntID": left["PntID"] + 1, "Time": left["Time"] + time_offset,
                    "Azimuth": left["Azimuth"] + .001, "Elevation": left["Elevation"] - .014,
                    "Distance": left["Distance"] - .002, "Amp": left["Amp"] - .017}
    rows = [left | {"BatchID": 2 if exchanged else 1}, right | {"BatchID": 1 if exchanged else 2}]
    return [left, right], rows


class PairedPublicIdentity(unittest.TestCase):
    def test_recovers_and_retains_identity_through_persistent_exchange_and_reversal(self):
        guard, state = PublicPairGuard(), {}
        for circle in range(12):
            points, rows = observations(circle, exchanged=6 <= circle <= 9)
            corrected = guard.correct(points[::-1], rows[::-1], state, circle)
            self.assertEqual([r["BatchID"] for r in corrected], [2, 1])
            self.assertEqual([{k: v for k, v in row.items() if k != "BatchID"} for row in corrected], points[::-1])
        self.assertEqual(guard.statistics["public_pair_swaps"], 2)

    def test_no_relabel_when_the_current_points_do_not_share_the_paired_timing(self):
        guard, state = PublicPairGuard(), {}
        for circle in range(6):
            points, rows = observations(circle)
            guard.correct(points, rows, state, circle)
        points, rows = observations(6, exchanged=True, time_offset=13)
        self.assertEqual(guard.correct(points, rows, state, 6), rows)
        self.assertEqual(guard.statistics["public_pair_swaps"], 0)


if __name__ == "__main__":
    unittest.main()
