"""High-angular paired returns through the public frozen tracker entrypoint."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithm import process_circle
from client import encode_tracks


def paired_scan(circle, azimuth):
    return [
        {"Time": circle * 5000 + 10 + arm, "Azimuth": (azimuth + arm * .0012) % 1,
         "Elevation": .19 + arm * .0011, "Distance": .30 + (circle + arm) * .0004,
         "Amp": .72 - arm * .023, "SNR": .72 - arm * .023,
         "CircleID": circle, "PntID": circle * 10 + arm + 1}
        for arm in range(2)
    ]


class PairedPrecisionTests(unittest.TestCase):
    def process_checked(self, points, state):
        outputs = process_circle(points, state)
        source = {(p["Time"], p["CircleID"], p["PntID"]): (p["Azimuth"], p["Elevation"], p["Distance"])
                  for p in points}
        encode_tracks(outputs, source)
        return {row["PntID"] % 10: row["BatchID"] for row in outputs}

    def test_fast_pair_survives_wrap_and_an_empty_scan(self):
        state = {}
        self.assertEqual(self.process_checked(paired_scan(0, .920), state), {})
        born = self.process_checked(paired_scan(1, .972)[::-1], state)
        self.assertEqual(len(set(born.values())), 2)
        self.assertEqual(self.process_checked(paired_scan(2, .024), state), born)
        self.assertEqual(self.process_checked([], state), {})
        self.assertEqual(self.process_checked(paired_scan(4, .128)[::-1], state), born)

        new_packet = {}
        self.assertEqual(self.process_checked(paired_scan(0, .920), new_packet), {})
        self.assertEqual(self.process_checked(paired_scan(1, .972), new_packet), born)


if __name__ == "__main__":
    unittest.main()
