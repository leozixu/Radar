"""Test a narrow pre-birth recovery for isolated established single-target tracks."""

from collections import Counter

import numpy as np

from causal_birth_paths import AMPLITUDE, CLOCK, POSITION
import pair_swap_probe


class IsolatedGapRecovery:
    def __init__(self, minimum):
        self.minimum = minimum
        self.counters = Counter()

    def associate(self, tracks, raw, available, matches, held, paired, circle, model):
        taken = np.zeros(len(tracks), dtype=bool)
        columns = available.copy()
        columns[held] = False
        for row, col, _ in matches:
            taken[row] = True
            columns[col] = False
        targets = np.flatnonzero(columns & (raw[:, CLOCK] >= .85))
        eligible = np.array([track.batch_id not in paired and track.hits >= 6 and len(track.history) >= 4
                             and 2 <= circle - track.circle <= 5 for track in tracks], dtype=bool) & ~taken
        if not eligible.any() or not len(targets):
            return []
        rows, cols, features = pair_swap_probe.candidate_features(
            tracks, raw[targets, POSITION], raw[targets, AMPLITUDE], circle)
        if not len(rows):
            return []
        q = model.predict(features, num_threads=1)
        reference = np.full(len(tracks), -1, dtype=int)
        c_reference = np.full(len(targets), -1, dtype=int)
        second_row = np.zeros(len(tracks))
        second_col = np.zeros(len(targets))
        for index in np.argsort(-q, kind="stable"):
            row, col = rows[index], cols[index]
            if reference[row] < 0:
                reference[row] = index
            else:
                second_row[row] = max(second_row[row], q[index])
            if c_reference[col] < 0:
                c_reference[col] = index
            else:
                second_col[col] = max(second_col[col], q[index])
        candidates = np.flatnonzero((q >= self.minimum) & eligible[rows])
        result = []
        for index in candidates:
            row, col = rows[index], cols[index]
            if reference[row] != index or c_reference[col] != index:
                continue
            if q[index] < 2.5 * max(second_row[row], second_col[col]):
                continue
            # Only small position innovations can override the ordinary score floor.
            azimuth, _, distance = features[index, 13:16]
            if azimuth > 2.7 or distance > 2.5:
                continue
            gap = int(features[index, 0])
            self.counters[f"recovered_gap_{gap}"] += 1
            result.append((int(row), int(targets[col]), float(q[index])))
        return result
