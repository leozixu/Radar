"""Admit unowned exact pairs that move beyond the normal birth gate."""

from collections import Counter

import numpy as np

from baseline_cv import residual

from .causal_birth_paths import POSITION, SCAN
from .paired_returns import PairReturns, pair_offset


ORDINARY_WIDTH = np.array([.038, .140, .020])
FAST_WIDTH = np.array([.110, .220, .020])
ROLE_WIDTH = np.array([.004, .0018, .0025])
AMPLITUDE_ERROR = .0000026


def mature_competitor(active, proposed, circle):
    """Detect proposals within the prediction gate of a mature track."""
    track_positions = np.array([track.position + track.velocity * (circle - track.circle)
                                for track in active if track.hits >= 3]).reshape(-1, 3)
    if not len(track_positions):
        return np.zeros(len(proposed), dtype=bool)
    error = residual(proposed[:, None, POSITION], track_positions[None, :, :])
    return (np.square(error / [.011, .075, .0045]).sum(axis=2) <= 1.).any(axis=1)


class FastPairCenter(PairReturns):
    def begin(self, tracks, raw, times, circle, state):
        frame = super().begin(tracks, raw, times, circle, state)
        used_prior = {id(prior) for prior, _ in frame.decisions}
        used_points = {point for _, index in frame.decisions for point in frame.columns[index]}
        early = [prior for prior in state.get("previous_pairs", ())
                 if id(prior) not in used_prior and not any(prior.owners)
                 and prior.observations[0, SCAN] == circle - 1]
        open_pairs = np.array([index for index, (a, b) in enumerate(frame.columns[:frame.strict_count])
                               if a not in used_points and b not in used_points], dtype=int)
        if not early or not len(open_pairs):
            return frame

        before = np.stack([prior.observations for prior in early])
        after = raw[frame.columns[open_pairs]]
        past_amp, current_amp = pair_offset(before), pair_offset(after)
        past_role = residual(before[:, 1, POSITION], before[:, 0, POSITION])
        current_role = residual(after[:, 1, POSITION], after[:, 0, POSITION])
        rows, cols = np.nonzero(
            (np.abs(past_amp[:, None] - current_amp[None, :]) <= AMPLITUDE_ERROR)
            & (past_amp[:, None] < 0.) & (current_amp[None, :] < 0.)
            & (past_role[:, None, 1] >= 0.) & (current_role[None, :, 1] >= 0.))
        if not len(rows):
            return frame
        movement = np.abs(residual(after[cols, :, POSITION], before[rows, :, POSITION]).mean(axis=1))
        shape = np.abs(residual(current_role[cols], past_role[rows]))
        valid = (~(movement <= ORDINARY_WIDTH).all(axis=1) & (movement <= FAST_WIDTH).all(axis=1)
                 & (shape <= ROLE_WIDTH).all(axis=1))
        rows, cols = rows[valid], cols[valid]
        if not len(rows):
            return frame
        occupied = mature_competitor(tracks, after[cols].reshape(-1, 7), circle).reshape(-1, 2).any(axis=1)
        rows, cols = rows[~occupied], cols[~occupied]
        prior_counts = Counter(int(row) for row in rows)
        point_counts = Counter(int(point) for col in cols for point in frame.columns[open_pairs[col]])
        for row, col in zip(rows, cols):
            index = int(open_pairs[col])
            if prior_counts[int(row)] != 1 or any(point_counts[int(point)] != 1 for point in frame.columns[index]):
                continue
            frame.decisions.append((early[row], index))
        return frame
