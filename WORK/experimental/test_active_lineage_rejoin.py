"""A rejoin preserves public identity and never duplicates an occupied raw owner."""

import unittest

import numpy as np

from active_lineage_rejoin import ActiveLineage, CorroboratedActiveLineage
from teacher_assoc import spawn, update


class AlwaysLink:
    @staticmethod
    def predict(features, num_threads):
        return np.full(len(features), .99)


class NeverLink:
    @staticmethod
    def predict(features, num_threads):
        return np.zeros(len(features))


def observation(circle, point, batch, distance):
    return {"CircleID": circle, "PntID": point, "BatchID": batch, "Time": 1000 * circle + point,
            "Amp": .4, "Azimuth": .30, "Elevation": .20, "Distance": distance}


def active_track(batch, start, end):
    track = spawn(np.array([.30, .20, .40 + .001 * start]), .4, start)
    for circle in range(start + 1, end + 1):
        update(track, np.array([.30, .20, .40 + .001 * circle]), .4, circle)
    track.batch_id = batch
    return track


class TestActiveLineage(unittest.TestCase):
    def first_streak(self, model):
        guard = ActiveLineage(model)
        old = active_track(1, 0, 2)
        for circle in range(3):
            row = observation(circle, circle + 1, 1, .40 + .001 * circle)
            guard.remember([row], [row], np.array([.99]), {"circle": circle, "tracks": ([old], [], [])})
        return guard, old

    def test_first_public_id_recovers_unassigned_previous_id(self):
        guard, old = self.first_streak(AlwaysLink())
        new = active_track(2, 3, 4)
        current = observation(4, 5, 2, .404)
        corrected = guard.correct([current], [current], [current], np.array([.99]),
                                  {"circle": 4, "tracks": ([old, new], [], [])})
        self.assertEqual(corrected, [current | {"BatchID": 1}])
        self.assertEqual(current["BatchID"], 2)
        old_next = observation(5, 6, 1, .404)
        new_next = observation(5, 7, 2, .405)
        both = [old_next, new_next]
        corrected = guard.correct(both, both, both, np.array([.99, .99]),
                                  {"circle": 5, "tracks": ([old, new], [], [])})
        self.assertEqual([row["BatchID"] for row in corrected], [2, 1])

    def test_occupied_old_raw_track_cannot_be_reassigned(self):
        guard, old = self.first_streak(AlwaysLink())
        new = active_track(2, 3, 4)
        old_point = observation(4, 6, 1, .4045)
        current = observation(4, 5, 2, .404)
        both = [old_point, current]
        corrected = guard.correct(both, both, [current], np.array([.99]),
                                  {"circle": 4, "tracks": ([old, new], [], [])})
        self.assertEqual(corrected, [current])

    def test_corroboration_uses_unpublished_neighbors_and_a_stable_signature(self):
        for changed_offset in (False, True):
            with self.subTest(changed_offset=changed_offset):
                guard = CorroboratedActiveLineage(NeverLink())
                old = active_track(1, 0, 2)
                for circle in range(3):
                    current = observation(circle, 5 + 2 * circle, 1, .40 + .001 * circle)
                    neighbor = current | {"Time": current["Time"] + 1, "PntID": current["PntID"] + 1,
                                          "Amp": .38, "Elevation": .201}
                    state = {"circle": circle, "tracks": ([old], [], [])}
                    guard.remember([current, neighbor], [current], np.array([.99]), state)
                new = active_track(2, 3, 4)
                current = observation(4, 13, 2, .404)
                neighbor = current | {"Time": current["Time"] + 1, "PntID": 14,
                                      "Amp": .376 if changed_offset else .38, "Elevation": .201}
                state = {"circle": 4, "tracks": ([old, new], [], []), "raw_return_probability": np.array([.9, .9])}
                result = guard.correct([current, neighbor], [current], [current], np.array([.99]), state)
                self.assertEqual(result, [current | {"BatchID": 2 if changed_offset else 1}])


if __name__ == "__main__":
    unittest.main()
