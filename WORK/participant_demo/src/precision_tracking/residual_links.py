"""Recover unassigned observations for established tracks using causal evidence."""

from collections import deque
import numpy as np

from baseline_cv import residual
from .causal_birth_paths import AMPLITUDE, CLOCK, POSITION, SCAN
from .clock_association import PERIOD_MS, wrap
from .teacher_assoc import NOISE, fit_predict, history_arrays


FEATURES = (
    "gap", "hit_count", "history_count", "raw_probability", "scan_clock",
    "protected", "held", "original_probability", "row_candidates", "column_candidates",
    "geometry_cost", "row_cost_excess", "column_cost_excess",
    "amplitude", "amplitude_mean", "amplitude_std", "amplitude_error",
    "pair_age", "mate_is_observed", "mate_raw_probability", "mapped_amplitude_error",
) + tuple(
    f"{kind}_{axis}"
    for kind in ("state_error", "short_error", "long_error", "last_error", "mate_error", "velocity")
    for axis in ("azimuth", "elevation", "range")
)

SEARCH_WIDTH = np.array((.070, .200, .025))
GEOMETRY_WIDTH = np.array((.010, .090, .004))
TEMPORAL_FEATURES = FEATURES + (
    "trusted_observations", "trusted_age", "phase_error", "clock_position_error", "clock_motion_error",
    "range_acceleration_error", "trusted_amplitude_error", "held_age", "previous_held_confidence",
    "previous_held_azimuth_error", "previous_held_elevation_error", "previous_held_range_error",
    "previous_held_amplitude_error", "previous_held_phase_error",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}


def corroborated_link_floors(features, floor):
    fields = FEATURE_INDEX
    unambiguous = (
        (features[:, fields["protected"]] == 0)
        & (features[:, fields["raw_probability"]] >= .88)
        & (features[:, fields["row_candidates"]] == 1)
        & (features[:, fields["column_candidates"]] == 1)
        & (features[:, fields["gap"]] <= 4)
    )
    paired = (
        (features[:, fields["mate_is_observed"]] == 1)
        & (features[:, fields["mapped_amplitude_error"]] <= 3.e-6)
        & (features[:, fields["mate_raw_probability"]] >= .88)
        & (features[:, fields["raw_probability"]] >= .65)
        & (features[:, fields["mate_error_azimuth"]] <= 2.5)
        & (features[:, fields["mate_error_elevation"]] <= .38)
        & (features[:, fields["mate_error_range"]] <= 2.5)
    )
    floors = np.full(len(features), floor)
    floors[unambiguous] = min(floor, .70)
    floors[paired] = min(floor, .50)
    return floors


def residual_candidates(tracks, raw, probability, circle, frame, previous_pairs, matches, held, rejected):
    used_rows = {row for row, _, _ in matches}
    used_columns = {col for _, col, _ in matches}
    protected = frame.protected if frame else set()
    if frame:
        for _, index in frame.decisions:
            used_columns.update(frame.columns[index])
    live = np.asarray([i for i, track in enumerate(tracks)
                       if i not in used_rows and track.hits >= 4
                       and 0 < circle - track.circle <= (12 if i in protected else 7)], dtype=int)
    columns = np.flatnonzero((probability >= .20) & ~np.isin(np.arange(len(raw)), list(used_columns)))
    if not len(live) or not len(columns):
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty((0, len(FEATURES)), dtype=np.float32)

    missed = [tracks[i] for i in live]
    gaps = np.asarray([circle - track.circle for track in missed])
    velocity = np.stack([track.velocity for track in missed])
    expected = np.stack([track.position for track in missed]) + gaps[:, None] * velocity
    positions = raw[columns, POSITION]
    delta = np.abs(residual(positions[None, :, :], expected[:, None, :]))
    width = SEARCH_WIDTH * (1 + .23 * (gaps - 1))[:, None]
    rows, cols = np.nonzero(((delta / width[:, None]) ** 2).sum(axis=2) <= 1)
    if not len(rows):
        return rows, cols, np.empty((0, len(FEATURES)), dtype=np.float32)

    history, times, amplitudes, present = history_arrays(missed)
    short, _, _ = fit_predict(history[:, -3:], times[:, -3:], present[:, -3:], circle, velocity)
    long, _, _ = fit_predict(history, times, present, circle, velocity)
    count = present.sum(axis=1)
    amp_mean = amplitudes.sum(axis=1) / count
    amp_std = np.sqrt(np.maximum((amplitudes ** 2).sum(axis=1) / count - amp_mean ** 2, 0))
    state_error = delta[rows, cols]
    short_error = np.abs(residual(positions[cols], short[rows]))
    long_error = np.abs(residual(positions[cols], long[rows]))
    last_error = np.abs(residual(positions[cols], history[rows, -1]))
    geometry_cost = (np.minimum(np.minimum(state_error, short_error), long_error) / GEOMETRY_WIDTH).sum(axis=1)
    row_best, col_best = np.full(len(live), np.inf), np.full(len(columns), np.inf)
    np.minimum.at(row_best, rows, geometry_cost)
    np.minimum.at(col_best, cols, geometry_cost)
    original_rows, original_cols = live[rows], columns[cols]

    latest_pair = {}
    for pair in previous_pairs:
        if not pair.confirmed:
            continue
        for side, owner in enumerate(pair.owners):
            age = circle - pair.observations[side, SCAN]
            if 0 < age <= 12 and (owner not in latest_pair or age < latest_pair[owner][0]):
                latest_pair[owner] = age, pair, side
    current_matches = {tracks[row].batch_id: col for row, col, _ in matches}
    prior = {(row, col): confidence for row, col, confidence in rejected}
    pair_context = np.full((len(rows), 7), np.nan)
    for index, (row, col) in enumerate(zip(original_rows, original_cols)):
        entry = latest_pair.get(tracks[row].batch_id)
        if entry is None:
            continue
        age, pair, side = entry
        pair_context[index, :2] = age, 0
        mate_col = current_matches.get(pair.owners[1 - side])
        if mate_col is None:
            continue
        own, mate = pair.observations[side], pair.observations[1 - side]
        mate_error = np.abs(residual(residual(raw[col, POSITION], raw[mate_col, POSITION]),
                                     residual(own[POSITION], mate[POSITION])))
        amp_error = abs((raw[col, AMPLITUDE] - raw[mate_col, AMPLITUDE])
                        - (own[AMPLITUDE] - mate[AMPLITUDE]))
        pair_context[index, 1:] = 1, probability[mate_col], amp_error, *(mate_error / NOISE)

    linked = np.column_stack((
        gaps[rows], [missed[r].hits for r in rows], count[rows], probability[original_cols],
        raw[original_cols, CLOCK], np.isin(original_rows, list(protected)), np.isin(original_cols, held),
        [prior.get((row, col), np.nan) for row, col in zip(original_rows, original_cols)],
        np.bincount(rows)[rows], np.bincount(cols)[cols],
        geometry_cost, geometry_cost - row_best[rows], geometry_cost - col_best[cols],
        raw[original_cols, AMPLITUDE], amp_mean[rows], amp_std[rows],
        np.abs(raw[original_cols, AMPLITUDE] - amp_mean[rows]), pair_context[:, :4],
        state_error / NOISE, short_error / NOISE, long_error / NOISE, last_error / NOISE,
        pair_context[:, 4:], np.abs(velocity[rows]) / NOISE,
    ))
    return original_rows, original_cols, linked.astype(np.float32)


class ResidualHistory:
    def __init__(self):
        self.trusted = {}
        self.held = {}

    def features(self, rows, cols, features, tracks, raw, times, circle):
        if not len(rows):
            return np.empty((0, len(TEMPORAL_FEATURES)), dtype=np.float32)
        supplements = np.full((len(rows), len(TEMPORAL_FEATURES) - len(FEATURES)), np.nan)
        clock = wrap(times / PERIOD_MS)
        phase = wrap(raw[:, 0] - clock)
        for row in np.unique(rows):
            selected = np.flatnonzero(rows == row)
            points = cols[selected]
            track = tracks[row]
            observations = self.trusted.get(track.batch_id)
            if observations:
                sample = np.asarray(observations)
                timing = sample[:, 0] - circle
                supplements[selected, 0] = len(sample)
                supplements[selected, 1] = -timing[-1]
                supplements[selected, 2] = np.abs(wrap(phase[points] - np.median(sample[:, 2])))
                supplements[selected, 3] = np.abs(wrap(clock[points] - sample[-1, 1]
                                                       + timing[-1] * track.velocity[0]))
                supplements[selected, 6] = np.abs(raw[points, AMPLITUDE] - np.median(sample[:, 4]))
                if len(sample) >= 3:
                    centered = timing - timing.mean()
                    angle = wrap(sample[:, 1] - sample[-1, 1])
                    predicted_movement = angle.mean() - timing.mean() * np.dot(centered, angle) / np.dot(centered, centered)
                    supplements[selected, 4] = np.abs(wrap(clock[points] - sample[-1, 1] - predicted_movement))
                    # Stable acceleration often explains a biased constant-velocity range.
                    design = np.column_stack((np.ones(len(timing)), timing, timing * timing))
                    response = sample[:, 3] - sample[-1, 3]
                    estimate = np.linalg.solve(design.T @ design + np.diag((0., 0., 24.)), design.T @ response)
                    supplements[selected, 5] = np.abs(raw[points, 2] - sample[-1, 3] - estimate[0])
            withheld = self.held.get(track.batch_id)
            if withheld is not None and (gap := circle - withheld[0]) <= 3:
                _, confidence, prior, prior_phase = withheld
                movement = np.abs(residual(raw[points, POSITION], prior[POSITION] + gap * track.velocity))
                supplements[selected, 7:] = np.column_stack((
                    np.full(len(points), gap), np.full(len(points), confidence),
                    movement / NOISE, np.abs(raw[points, AMPLITUDE] - prior[AMPLITUDE]),
                    np.abs(wrap(phase[points] - prior_phase)),
                ))
        return np.column_stack((features, supplements)).astype(np.float32)

    def advance(self, tracks, raw, times, probability, circle, matches, rejected):
        owners = {track.batch_id for track in tracks}
        self.trusted = {ident: history for ident, history in self.trusted.items() if ident in owners}
        self.held = {ident: pending for ident, pending in self.held.items()
                     if ident in owners and circle - pending[0] <= 3}
        for row, col, confidence in matches:
            if confidence >= .86:
                clock = wrap(times[col] / PERIOD_MS)
                samples = self.trusted.setdefault(tracks[row].batch_id, deque(maxlen=10))
                samples.append((circle, clock, wrap(raw[col, 0] - clock), raw[col, 2], raw[col, AMPLITUDE]))
        for row, col, confidence in rejected:
            if probability[col] >= .75:
                self.held[tracks[row].batch_id] = (circle, confidence, raw[col].copy(),
                                                  wrap(raw[col, 0] - times[col] / PERIOD_MS))


class ResidualLinker:
    def __init__(self, model, floor, margin, temporal=False, corroborated=False):
        self.model = model
        self.floor = floor
        self.margin = margin
        self.history = ResidualHistory() if temporal else None
        self.corroborated = corroborated

    def recover(self, tracks, raw, times, probability, circle, frame, previous_pairs, matches, held, rejected):
        rows, cols, features = residual_candidates(
            tracks, raw, probability, circle, frame, previous_pairs, matches, held, rejected
        )
        if self.history is not None:
            features = self.history.features(rows, cols, features, tracks, raw, times, circle)
            self.history.advance(tracks, raw, times, probability, circle, matches, rejected)
        if not len(rows):
            return []
        scores = self.model.predict(features, num_threads=1)
        floors = corroborated_link_floors(features, self.floor) if self.corroborated else np.full(len(rows), self.floor)
        # Decide against the best alternative for both the track and the point.
        row_taken, col_taken = set(), set()
        accepted = []
        for index in np.argsort(-scores, kind="stable"):
            row, col, score = rows[index], cols[index], scores[index]
            if score < floors[index] or row in row_taken or col in col_taken:
                continue
            other = ((rows == row) | (cols == col)) & (np.arange(len(rows)) != index)
            if np.any(scores[other] > score - self.margin):
                continue
            accepted.append((row, col, float(min(score, .85))))
            row_taken.add(row)
            col_taken.add(col)
        return accepted
