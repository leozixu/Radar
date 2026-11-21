"""Confirm two-observation tracks from points left unused by the main tracker."""

from dataclasses import replace

import numpy as np

from baseline_cv import OUTPUT_FIELDS, residual
from tracker_assoc import spatial_edges

from .causal_birth_paths import AMPLITUDE, CLOCK, POINT, POSITION, SCAN


LINK_WIDTH = np.array([.070, .155, .023])
INCUMBENT_WIDTH = np.array([.015, .080, .007])
MIN_RAW = .78
FEATURES = (
    "scan_gap", "movement_cost", "movement_azimuth", "movement_elevation", "movement_range",
    "before_amplitude", "after_amplitude", "amplitude_change", "before_clock", "after_clock",
    "before_raw", "after_raw", "before_pair", "after_pair", "before_incumbent", "after_incumbent",
    "before_degree", "after_degree", "before_cost_excess", "after_cost_excess",
)


def points_with_evidence(rows, evidence):
    probabilities = np.fromiter((evidence.get(int(row[POINT]), np.nan) for row in rows), dtype=float)
    keep = probabilities >= MIN_RAW
    return rows[keep], probabilities[keep]


def incumbent_distance(rows, tracks):
    mature = [track for track in tracks if track.hits >= 3]
    if not mature:
        return np.full(len(rows), 8.)
    position = np.stack([track.position for track in mature])
    velocity = np.stack([track.velocity for track in mature])
    circle = np.fromiter((track.circle for track in mature), dtype=int)
    age = rows[:, SCAN, None] - circle[None, :]
    predicted = position[None, :, :] + age[:, :, None] * velocity[None, :, :]
    distance = np.sum((residual(rows[:, None, POSITION], predicted) / INCUMBENT_WIDTH) ** 2, axis=2)
    distance[abs(age) > 7] = 8.
    return np.minimum(distance.min(axis=1), 8.)


def candidate_edges(before, after, previous_raw, current_raw, tracks, pair_points):
    gap = after[0, SCAN] - before[:, SCAN]
    width = LINK_WIDTH * np.sqrt(gap[:, None])
    rows, cols = spatial_edges(after[:, POSITION], before[:, POSITION], width)
    movement = residual(after[cols, POSITION], before[rows, POSITION])
    cost = np.sum((movement / width[rows]) ** 2, axis=1)
    available = (cost <= 1.) & (cost >= .0005)
    rows, cols, movement, cost = rows[available], cols[available], movement[available], cost[available]
    if not len(rows):
        return rows, cols, np.empty((0, len(FEATURES)), dtype=np.float32)

    seen_before = incumbent_distance(before, tracks)
    seen_after = incumbent_distance(after, tracks)
    paired_before = np.isin(before[:, POINT], pair_points)
    paired_after = np.isin(after[:, POINT], pair_points)
    left_best = np.full(len(before), np.inf)
    right_best = np.full(len(after), np.inf)
    np.minimum.at(left_best, rows, cost)
    np.minimum.at(right_best, cols, cost)
    a, b = before[rows], after[cols]
    features = np.column_stack((
        gap[rows], cost, abs(movement / LINK_WIDTH), a[:, AMPLITUDE], b[:, AMPLITUDE],
        abs(a[:, AMPLITUDE] - b[:, AMPLITUDE]), a[:, CLOCK], b[:, CLOCK],
        previous_raw[rows], current_raw[cols], paired_before[rows], paired_after[cols],
        seen_before[rows], seen_after[cols], np.bincount(rows, minlength=len(before))[rows],
        np.bincount(cols, minlength=len(after))[cols], cost - left_best[rows], cost - right_best[cols],
    )).astype(np.float32)
    return rows, cols, features


def unique_choices(rows, cols, scores, floor, margin):
    """Keep an edge only when its score wins on both endpoints."""
    row_choices, col_choices = {}, {}
    for i in np.argsort(-scores, kind="stable"):
        for choices, key in ((row_choices, rows[i]), (col_choices, cols[i])):
            contenders = choices.setdefault(key, [])
            if len(contenders) < 2:
                contenders.append(i)
    for i in np.flatnonzero(scores >= floor):
        competitors = (row_choices[rows[i]], col_choices[cols[i]])
        if all(rival[0] == i and (len(rival) == 1 or scores[i] - scores[rival[1]] >= margin)
               for rival in competitors):
            yield i


class EarlyBirthTracker:
    def __init__(self, original, model, floor=.81, margin=.12):
        self.original = original
        self.model = model
        self.floor = floor
        self.margin = margin
        self.raw_by_circle = {}
        self.pairs_by_circle = {}

    def track(self, points, state, observe=None):
        outputs = self.original.track(points, state, observe)
        circle = state["circle"]
        self.raw_by_circle = {t: values for t, values in self.raw_by_circle.items() if circle - t <= 2}
        self.pairs_by_circle = {t: values for t, values in self.pairs_by_circle.items() if circle - t <= 2}
        if not points:
            return outputs
        pool = state["birth_path_pool"]
        available = dict(pool.observations)
        recent = state["previous_pairs"]
        self.pairs_by_circle[circle] = {int(pid) for pair in recent if pair.observations[0, SCAN] == circle
                                        for pid in pair.observations[:, POINT]}
        self.raw_by_circle[circle] = {p["PntID"]: q for p, q in zip(points, state["raw_return_probability"])
                                      if q >= MIN_RAW}
        if circle not in available or len(points) > 1600:
            return outputs
        after, current_raw = points_with_evidence(available[circle], self.raw_by_circle[circle])
        previous = sorted(available.keys() - {circle})
        if not len(after) or not previous:
            return outputs
        parts = [points_with_evidence(available[t], self.raw_by_circle.get(t, {})) for t in previous]
        before, previous_raw = np.vstack([rows for rows, _ in parts]), np.concatenate([q for _, q in parts])
        if not len(before) or len(before) + len(after) > 480:
            return outputs

        active, tentative, seeds = state["tracks"]
        pair_points = tuple(set().union(*self.pairs_by_circle.values()))
        rows, cols, features = candidate_edges(before, after, previous_raw, current_raw, active, pair_points)
        if not len(rows):
            return outputs
        scores = self.model.predict(features, num_threads=1)
        matched, claimed = {}, set()
        next_id = state["next_batch_id"]
        for i in unique_choices(rows, cols, scores, self.floor, self.margin):
            head, tail = before[rows[i]], after[cols[i]]
            self.original.confirm((head, tail), active, matched, next_id, observe)
            claimed.update((int(head[POINT]), int(tail[POINT])))
            next_id += 1
        if not matched:
            return outputs

        state["next_batch_id"] = next_id
        pool.add(np.empty((0, 7)), circle, np.fromiter(claimed, dtype=np.int64))
        young = state["young_keys"]
        tentative = [t for t in tentative if not claimed.intersection(young[id(t)])]
        seeds = [t for t in seeds if not claimed.intersection(young[id(t)])]
        state["tracks"] = active, tentative, seeds
        state["young_keys"] = {id(t): young[id(t)] for t in tentative + seeds}
        state["previous_pairs"] = [
            replace(pair, owners=tuple(matched.get(int(pid), owner)
                                       for pid, owner in zip(pair.observations[:, POINT], pair.owners)))
            if pair.observations[0, SCAN] == circle else pair for pair in recent
        ]
        outputs.extend({**{key: p[key] for key in OUTPUT_FIELDS}, "BatchID": matched[p["PntID"]]}
                       for p in points if p["PntID"] in matched)
        return outputs
