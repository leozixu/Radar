"""Precision admission and retention of paired radar returns."""

from dataclasses import dataclass, field

import numpy as np

from baseline_cv import residual

from .causal_birth_paths import AMPLITUDE, CLOCK, POINT, POSITION, SCAN

MAX_PAIRS = 800
AMPLITUDE_TOLERANCE = .0000031
MAX_GAP = 12
PAIR_WIDTH = np.array((.018, .012, .008))
FEATURES = (
    "gap", "signature_difference", "signature_magnitude", "prior_clock_min", "current_clock_min",
    "prior_amplitude_mean", "amplitude_mean", "amplitude_change", "amplitude_min",
) + tuple(f"{part}_{axis}" for part in ("prior_width", "current_width", "width_change", "movement")
          for axis in ("azimuth", "elevation", "range"))


def measured_pairs(raw, times, wide=False):
    """Return spatially close measurements with distinct neighbouring radar ticks."""
    order = np.argsort(times, kind="stable")
    stamps = times[order]
    neighbours = np.diff(stamps)
    unique = np.r_[True, neighbours != 0] & np.r_[neighbours != 0, True] if len(stamps) else np.empty(0, bool)
    if wide:
        adjacent = []
        for step in range(1, min(3, len(stamps) - 1) + 1):
            elapsed = stamps[step:] - stamps[:-step]
            left = np.flatnonzero((elapsed >= 1) & (elapsed <= 3) & unique[:-step] & unique[step:])
            adjacent.append(np.column_stack((order[left], order[left + step])))
        pair = np.concatenate(adjacent) if adjacent else np.empty((0, 2), dtype=int)
    else:
        left = np.flatnonzero((neighbours == 1) & unique[:-1] & unique[1:])
        pair = np.column_stack((order[left], order[left + 1]))
    before, after = raw[pair[:, 0]], raw[pair[:, 1]]
    width = np.abs(residual(after[:, POSITION], before[:, POSITION]))
    amp_difference = np.abs(after[:, AMPLITUDE] - before[:, AMPLITUDE])
    clock = np.minimum(before[:, CLOCK], after[:, CLOCK])
    widths = np.array((.046, .065, .020)) if wide else PAIR_WIDTH
    valid = (width <= widths).all(axis=1) & (amp_difference >= .000004) & (amp_difference <= .032)
    if not wide:
        valid &= (clock >= .20) | (np.minimum(before[:, AMPLITUDE], after[:, AMPLITUDE]) >= .20)
    eligible = np.flatnonzero(valid)
    if len(eligible) > MAX_PAIRS:
        eligible = eligible[np.argsort(-clock[eligible], kind="stable")[:MAX_PAIRS]]
    return pair[eligible]


def pair_offset(observations):
    return observations[:, 1, AMPLITUDE] - observations[:, 0, AMPLITUDE]


def link_features(prior, current, tolerance=AMPLITUDE_TOLERANCE):
    """Gate by the signed amplitude difference and local motion before model inference."""
    past_offsets = pair_offset(prior)
    offsets = pair_offset(current)
    order = np.argsort(offsets, kind="stable")
    ranges = np.searchsorted(offsets[order], past_offsets[:, None] + np.array([-tolerance, tolerance]))
    counts = ranges[:, 1] - ranges[:, 0]
    rows = np.repeat(np.arange(len(prior)), counts)
    if not len(rows):
        return rows, rows.copy(), np.empty((0, len(FEATURES)), np.float32)
    starts = np.repeat(ranges[:, 0] - np.r_[0, counts.cumsum()[:-1]], counts)
    cols = order[starts + np.arange(len(rows))]
    former, latter = prior[rows], current[cols]
    gap = latter[:, 0, SCAN] - former[:, 0, SCAN]
    azimuth_range = residual(latter[:, :, POSITION], former[:, :, POSITION])
    motion = np.abs(azimuth_range.mean(axis=1))
    limit = np.array([.038, .14, .020]) * (1 + .6 * np.maximum(0, gap - 1))[:, None]
    valid = (gap > 0) & (gap <= MAX_GAP) & (motion <= limit).all(axis=1)
    rows, cols, former, latter, motion, gap = (array[valid] for array in (rows, cols, former, latter, motion, gap))
    former_delta = residual(former[:, 1, POSITION], former[:, 0, POSITION])
    latter_delta = residual(latter[:, 1, POSITION], latter[:, 0, POSITION])
    prior_amp = former[:, :, AMPLITUDE].mean(axis=1)
    amp = latter[:, :, AMPLITUDE].mean(axis=1)
    features = np.column_stack((
        gap, np.abs(past_offsets[rows] - offsets[cols]), np.abs(offsets[cols]),
        former[:, :, CLOCK].min(axis=1), latter[:, :, CLOCK].min(axis=1), prior_amp, amp,
        np.abs(prior_amp - amp), latter[:, :, AMPLITUDE].min(axis=1), np.abs(former_delta), np.abs(latter_delta),
        np.abs(residual(latter_delta, former_delta)), motion / gap[:, None],
    ))
    return rows, cols, features.astype(np.float32)


@dataclass(slots=True)
class ObservedPair:
    observations: np.ndarray
    owners: tuple[int, int]
    confirmed: bool


@dataclass(slots=True)
class PairScan:
    columns: np.ndarray
    decisions: list
    locked: list
    strict_count: int = 0
    protected: set = field(default_factory=set)


class PairReturns:
    def __init__(self, model, start_floor=.985, link_floor=.95, early=True, *,
                 wide=True, fallback_floor=.96):
        self.model = model
        self.start_floor = start_floor
        self.link_floor = link_floor
        self.early = early
        self.wide = wide
        self.fallback_floor = fallback_floor

    @staticmethod
    def retained(state):
        return {owner for prior in state.get("previous_pairs", ()) if prior.confirmed for owner in prior.owners if owner}

    def begin(self, tracks, raw, times, circle, state):
        columns = measured_pairs(raw, times)
        strict_count = len(columns)
        previous = [prior for prior in state.get("previous_pairs", ())
                    if circle - prior.observations[0, SCAN] <= (MAX_GAP if prior.confirmed else 1)]
        registered_owners = {owner for prior in previous if prior.confirmed for owner in prior.owners}
        if self.wide and registered_owners:
            strict = set(map(tuple, columns))
            extended = [pair for pair in measured_pairs(raw, times, wide=True) if tuple(pair) not in strict]
            if extended:
                columns = np.vstack((columns, extended))
        frame = PairScan(columns, [], [], strict_count)
        active = {track.batch_id: index for index, track in enumerate(tracks)}
        if self.fallback_floor:
            frame.protected = {active[owner] for owner in registered_owners if owner in active}
        if not len(columns) or not previous:
            return frame
        rows, cols, features = link_features(np.stack([r.observations for r in previous]), raw[columns])
        if not len(rows):
            return frame
        scores = self.model.predict(features, num_threads=1)
        registered = np.asarray([prior.confirmed for prior in previous])[rows]
        ordinary = cols < strict_count
        accepted = (scores >= np.where(registered, self.link_floor, self.start_floor)) & (ordinary | registered)
        # A proven pair keeps both identities when a faint duplicate has the same offset.
        priority = scores - .045 * np.minimum(1., np.linalg.norm(features[:, -3:] / [.014, .05, .012], axis=1))
        priority += .05 * registered - .04 * ~ordinary
        used_rows, used_columns, used_owners = set(), set(), set()
        ranked = np.flatnonzero(accepted)
        for index in ranked[np.argsort(-priority[ranked], kind="stable")]:
            row, col = int(rows[index]), int(cols[index])
            prior, dest = previous[row], columns[col]
            owners = tuple((active[owner], int(point)) for owner, point in zip(prior.owners, dest) if owner in active)
            if row in used_rows or any(p in used_columns for p in dest) or any(k in used_owners for k, _ in owners):
                continue
            used_rows.add(row)
            used_columns.update(dest)
            used_owners.update(key for key, _ in owners)
            frame.decisions.append((prior, col))
            frame.locked.extend((key, point, float(max(.90, scores[index]))) for key, point in owners)
        return frame

    def protect_fallback(self, frame, tracks, positions, circle, matches, held):
        if not frame.protected:
            return matches, held
        locks = {row for row, _, _ in frame.locked}
        accepted, added = [], []
        for row, col, probability in matches:
            if row in frame.protected and row not in locks and probability < self.fallback_floor:
                added.append(col)
            else:
                accepted.append((row, col, probability))
        return accepted, np.r_[held, np.array(added, dtype=int)]

    def starts(self, frame, remaining):
        if not self.early:
            return
        for prior, index in frame.decisions:
            current = frame.columns[index]
            # A partial pair goes through the original per-target birth path.
            if any(prior.owners) or not remaining[current].all():
                continue
            yield prior.observations, current

    def remember(self, frame, raw, output, circle, state):
        verified = {index: prior for prior, index in frame.decisions}
        used = {id(prior) for prior in verified.values()}
        observed_owners = set()
        previous = []
        for index, points in enumerate(raw[frame.columns]):
            owners = tuple(output.get(int(point), 0) for point in points[:, POINT])
            confirmed = index in verified and all(owners) and owners[0] != owners[1]
            if index >= frame.strict_count and not confirmed:
                continue
            if confirmed:
                observed_owners.update(owners)
            previous.append(ObservedPair(points, owners, confirmed))
        for prior in state.get("previous_pairs", ()):
            if (prior.confirmed and id(prior) not in used and circle - prior.observations[0, SCAN] < MAX_GAP
                    and not observed_owners.intersection(prior.owners)):
                previous.append(prior)
        state["previous_pairs"] = previous
