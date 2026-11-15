"""Assign candidate returns and resolve ambiguous reciprocal swaps."""

import numpy as np

from baseline_cv import residual
from tracker_assoc import assign

from .teacher_assoc import FEATURES, NOISE, candidate_features


PAIR_NAMES = tuple(f'{role}_{name}' for role in ('left', 'right', 'left_cross', 'right_cross')
                   for name in FEATURES)
PAIR_NAMES += ('direct_log_probability', 'cross_log_probability', 'common_past_scans')
PAIR_NAMES += tuple(f'{kind}_{axis}' for kind in ('difference', 'expected_difference', 'spread',
                                                    'relative_velocity', 'error', 'cross_error')
                   for axis in ('az', 'el', 'range'))


def relative_history(left, right, circle):
    earlier = {scan: pos for scan, _, pos in left.history}
    overlap = [(scan, earlier[scan], pos) for scan, _, pos in right.history if scan in earlier]
    if not overlap:
        delta = residual(left.position, right.position)
        velocity = left.velocity - right.velocity
        return 0, delta + velocity * max(1, circle - max(left.circle, right.circle)), NOISE, velocity
    t, a, b = map(np.asarray, zip(*overlap))
    relative_positions = residual(a, b)
    mean = relative_positions.mean(axis=0)
    center = t - t.mean()
    velocity = ((relative_positions - mean) * center[:, None]).sum(axis=0) / max(1., np.dot(center, center))
    expected = mean + (circle - t.mean()) * velocity
    return len(t), expected, (relative_positions - mean - center[:, None] * velocity).std(axis=0), velocity


def make_features(tracks, current, individual, index, confidence, left, right, a, b, circle):
    i, j, k, m = index[left, a], index[right, b], index[left, b], index[right, a]
    direct, crossed = confidence[i] + confidence[j], confidence[k] + confidence[m]
    count, mean, spread, velocity = relative_history(tracks[left], tracks[right], circle)
    delta = residual(current[a], current[b])
    context = np.concatenate((mean, spread, velocity)) / np.tile(NOISE, 3)

    def arrangement(first, second, left_cross, right_cross, good, alt, change):
        errors = np.r_[np.abs(change - mean), np.abs(-change - mean)] / np.tile(NOISE, 2)
        return np.r_[individual[[first, second, left_cross, right_cross]].ravel(), good, alt,
                     count, change / NOISE, context, errors]

    return (arrangement(i, j, k, m, direct, crossed, delta),
            arrangement(k, m, i, j, crossed, direct, -delta), direct - crossed)


class PairCorrector:
    def __init__(self, model, cutoff):
        self.model = model
        self.cutoff = cutoff
        self.swaps = 0

    def associate(self, tracks, positions, amplitudes, circle, model, floor, mix, ambiguity, *, edge_data=None):
        if not tracks or not len(positions):
            return [], np.empty(0, dtype=int)
        if edge_data is None:
            rows, cols, x = candidate_features(tracks, positions, amplitudes, circle)
            if not len(rows):
                return [], np.empty(0, dtype=int)
            probabilities = model.predict(x, num_threads=1)
        else:
            rows, cols, x, probabilities = edge_data
            if not len(rows):
                return [], np.empty(0, dtype=int)
        log_p = np.log(np.clip(probabilities, 1e-6, 1.))
        chosen, competing = assign(rows, cols, log_p - np.log(floor) - mix * x[:, 10],
                                    np.ones(len(tracks), dtype=bool), len(tracks), len(positions), ambiguity)
        selected = np.asarray(chosen, dtype=int)
        held = {int(cols[i]) for candidates in competing.values() for i in candidates}
        left, right = np.triu_indices(len(selected), 1)
        kept_rows, kept_cols = rows[selected], cols[selected]
        qualified = np.asarray([len(track.history) >= 4 for track in tracks])[kept_rows]
        left, right = left[qualified[left] & qualified[right]], right[qualified[left] & qualified[right]]
        repaired = set()
        if len(left):
            index = np.full((len(tracks), len(positions)), -1, dtype=np.int32)
            index[rows, cols] = np.arange(len(rows))
            cross1 = index[kept_rows[left], kept_cols[right]]
            cross2 = index[kept_rows[right], kept_cols[left]]
            acceptable = ((cross1 >= 0) & (cross2 >= 0) &
                          (probabilities[cross1] >= floor / 2) & (probabilities[cross2] >= floor / 2))
            pairs = []
            for a, b in zip(left[acceptable], right[acceptable]):
                row_a, row_b = kept_rows[a], kept_rows[b]
                col_a, col_b = kept_cols[a], kept_cols[b]
                features = make_features(tracks, positions, x, index, log_p,
                                         row_a, row_b, col_a, col_b, circle)
                if features[0][4 * len(FEATURES) + 2] >= 3:
                    pairs.append((a, b, features[0], features[1]))
            if pairs:
                features = np.asarray([x for _, _, primary, other in pairs for x in (primary, other)])
                prediction = self.model.predict(features, num_threads=1).reshape(-1, 2)
                # Require both orientations to agree and repair disjoint pairs only.
                confidence = np.minimum(1. - prediction[:, 0], prediction[:, 1])
                for k in np.argsort(-confidence, kind='stable'):
                    if confidence[k] < self.cutoff:
                        break
                    a, b = pairs[k][:2]
                    if a in repaired or b in repaired:
                        continue
                    selected[a] = index[kept_rows[a], kept_cols[b]]
                    selected[b] = index[kept_rows[b], kept_cols[a]]
                    repaired.update((a, b))
                    self.swaps += 1
        matches = [(int(rows[i]), int(cols[i]), float(max(.60, probabilities[i]) if j in repaired
                                                      else probabilities[i]))
                   for j, i in enumerate(selected)]
        return matches, np.fromiter(held, dtype=int)
