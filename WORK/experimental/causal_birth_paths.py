"""Train and benchmark a three-observation birth detector with competing causal paths."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import teacher_assoc as teacher
from algorithm import load_runtime
from clock_association import phase_features
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths, validate_packet_path
from pair_swap_probe import PairCorrector
import pair_swap_probe
import replay_validation as replay
from tracker_assoc import spatial_edges
import tracker_compat
from train_tracker import read_packet


POSITION = slice(0, 3)
AMPLITUDE, CLOCK, SCAN, POINT = 3, 4, 5, 6
BEAM_WIDTH = 4
LINK_WIDTH = np.array([.041, .110, .019])
MOTION_SCALE = np.array([.004, .025, .0016])
CONFIRM_MOVEMENT = np.array([.002, .025, .0012])
FEATURES = (
    'first_gap', 'second_gap', 'amplitude_first', 'amplitude_middle', 'amplitude_current',
    'amplitude_spread', 'clock_first', 'clock_middle', 'clock_current',
    'clock_min', 'clock_mean', 'parent_cost', 'prediction_distance', 'possible_current_links',
    'competing_paths',
) + tuple(f'{group}_{axis}' for group in ('motion_first', 'motion_second', 'velocity_change', 'forecast_error')
          for axis in ('azimuth', 'elevation', 'range'))


def measurements(position, amp, time, point_ids, circle, clock_model):
    if not len(amp):
        return np.empty((0, 7))
    feature = phase_features(position[:, 0], time, amp)
    probability = clock_model.predict(feature, num_threads=1)
    return np.column_stack((position, amp, probability, np.full(len(amp), circle), point_ids))


def eligible(points):
    return points[(points[:, AMPLITUDE] >= .12) | (points[:, CLOCK] >= .42)]


@dataclass(slots=True)
class Arcs:
    first: np.ndarray
    middle: np.ndarray
    cost: np.ndarray

    def subset(self, mask):
        return Arcs(self.first[mask], self.middle[mask], self.cost[mask])


class PathPool:
    def __init__(self, gap=2):
        self.observations = deque()
        self.arcs = deque()
        self.gap = gap

    def expire(self, circle):
        while self.observations and circle - self.observations[0][0] > self.gap:
            self.observations.popleft()
        while self.arcs and circle - self.arcs[0][0] > self.gap:
            self.arcs.popleft()

    def propose(self, points, circle):
        self.expire(circle)
        if not len(points) or not self.arcs:
            return None
        start = np.vstack([arcs.first for _, arcs in self.arcs])
        middle = np.vstack([arcs.middle for _, arcs in self.arcs])
        cost = np.concatenate([arcs.cost for _, arcs in self.arcs])
        valid = (circle - start[:, SCAN]) <= 2 * self.gap
        start, middle, cost = start[valid], middle[valid], cost[valid]
        if not len(start):
            return None
        first_gap = middle[:, SCAN] - start[:, SCAN]
        second_gap = circle - middle[:, SCAN]
        before = teacher.cv.residual(middle[:, POSITION], start[:, POSITION]) / first_gap[:, None]
        forecast = middle[:, POSITION] + before * second_gap[:, None]
        widths = np.array([.028, .120, .012]) * (1. + .4 * (second_gap - 1))[:, None]
        rows, cols = spatial_edges(points[:, POSITION], forecast, widths)
        if not len(rows):
            return None
        errors = teacher.cv.residual(points[cols, POSITION], forecast[rows])
        distance = np.square(errors / widths[rows]).sum(axis=1)
        keep = distance <= 1.
        rows, cols, errors, distance = rows[keep], cols[keep], errors[keep], distance[keep]
        if not len(rows):
            return None
        a, b, c = start[rows], middle[rows], points[cols]
        afterwards = teacher.cv.residual(c[:, POSITION], b[:, POSITION]) / second_gap[rows, None]
        amp = np.column_stack((a[:, AMPLITUDE], b[:, AMPLITUDE], c[:, AMPLITUDE]))
        clock = np.column_stack((a[:, CLOCK], b[:, CLOCK], c[:, CLOCK]))
        x = np.column_stack((first_gap[rows], second_gap[rows], amp, np.ptp(amp, axis=1), clock,
                             clock.min(axis=1), clock.mean(axis=1), cost[rows], distance,
                             np.bincount(rows, minlength=len(start))[rows],
                             np.bincount(cols, minlength=len(points))[cols],
                             np.abs(before[rows]) / MOTION_SCALE, np.abs(afterwards) / MOTION_SCALE,
                             np.abs(afterwards - before[rows]) / MOTION_SCALE, np.abs(errors) / MOTION_SCALE))
        return a, b, cols, x.astype(np.float32)

    def add(self, points, circle, claimed=()):
        self.expire(circle)
        if len(claimed):
            self.observations = deque((t, rows[~np.isin(rows[:, POINT], claimed)])
                                      for t, rows in self.observations)
            self.arcs = deque((t, arcs.subset(~(np.isin(arcs.first[:, POINT], claimed) |
                                                np.isin(arcs.middle[:, POINT], claimed))))
                              for t, arcs in self.arcs)
        if not len(points):
            return
        parents, children, costs = [], [], []
        for t, previous in self.observations:
            if not len(previous):
                continue
            width = LINK_WIDTH * (1. + .45 * (circle - t - 1))
            rows, cols = spatial_edges(points[:, POSITION], previous[:, POSITION], width[None, :])
            error = teacher.cv.residual(points[cols, POSITION], previous[rows, POSITION])
            distance = np.square(error / width).sum(axis=1)
            valid = distance <= 1.
            rows, cols, distance = rows[valid], cols[valid], distance[valid]
            first, second = previous[rows], points[cols]
            cost = (distance + .20 * np.abs(first[:, AMPLITUDE] - second[:, AMPLITUDE])
                    - .28 * first[:, CLOCK])
            parents.append(first)
            children.append(cols)
            costs.append(cost)
        if parents:
            first = np.vstack(parents)
            child, cost = np.concatenate(children), np.concatenate(costs)
            order = np.lexsort((cost, child))
            counts = np.bincount(child, minlength=len(points))
            rank = np.arange(len(order)) - (counts.cumsum() - counts)[child[order]]
            top = order[rank < BEAM_WIDTH]
            if len(top):
                self.arcs.append((circle, Arcs(first[top], points[child[top]], cost[top])))
        self.observations.append((circle, points))


def supported_birth_pairs(head, middle, tail, cols, probability):
    """Propose two disjoint starts with a stable relative elevation over three scans."""
    possible = np.flatnonzero(probability >= .47)
    if len(possible) < 2:
        return []
    order = possible[np.lexsort((-probability[possible], cols[possible]))]
    counts = np.bincount(cols[possible], minlength=len(tail))
    ranks = np.arange(len(order)) - (counts.cumsum() - counts)[cols[order]]
    best = order[ranks < 2]
    best = best[np.argsort(-probability[best], kind='stable')[:400]]
    a, b, c = head[best], middle[best], tail[cols[best]]
    pos = c[:, POSITION]
    left, right = spatial_edges(pos, pos, np.array([[.037, .12, .015]]))
    masks = ((left < right) & (a[left, SCAN] == a[right, SCAN]) & (b[left, SCAN] == b[right, SCAN]) &
             (a[left, POINT] != a[right, POINT]) & (b[left, POINT] != b[right, POINT]) &
             (c[left, POINT] != c[right, POINT]))
    left, right = left[masks], right[masks]
    if not len(left):
        return []
    differences = np.column_stack([values[left, 1] - values[right, 1] for values in (a, b, c)])
    spread = np.ptp(differences, axis=1)
    keep = (spread <= .00055) & (np.abs(differences).min(axis=1) >= .0007)
    first, second, noise = best[left[keep]], best[right[keep]], spread[keep]
    scores = np.minimum(probability[first], probability[second]) - noise / .005
    return [(int(first[i]), int(second[i])) for i in np.argsort(-scores, kind='stable')]


def supported_two_hit_pairs(arcs):
    confidence = np.minimum(arcs.first[:, CLOCK], arcs.middle[:, CLOCK])
    good = np.flatnonzero((confidence >= .85) & (arcs.cost <= .65))
    if len(good) < 2:
        return []
    good = good[np.argsort(-confidence[good], kind='stable')[:450]]
    previous, current = arcs.first[good], arcs.middle[good]
    pos = current[:, POSITION]
    left, right = spatial_edges(pos, pos, np.array([[.034, .10, .014]]))
    distinct = ((left < right) & (previous[left, SCAN] == previous[right, SCAN]) &
                (previous[left, POINT] != previous[right, POINT]) & (current[left, POINT] != current[right, POINT]))
    left, right = left[distinct], right[distinct]
    if not len(left):
        return []
    before = teacher.cv.residual(previous[left, POSITION], previous[right, POSITION])
    after = teacher.cv.residual(current[left, POSITION], current[right, POSITION])
    shift = np.abs(teacher.cv.residual(after, before))
    stable = ((shift <= np.array([.008, .00015, .004])).all(axis=1) & (np.abs(after[:, 1]) >= .0008))
    a, b = good[left[stable]], good[right[stable]]
    score = np.minimum(confidence[a], confidence[b]) - 100. * shift[stable, 1]
    return [(int(a[i]), int(b[i])) for i in np.argsort(-score, kind='stable')]


def confident_two_hit_starts(previous, current, model):
    """Early starts require reciprocal confidence and two plausible detections."""
    from young_assoc import edge_features

    if not len(previous) or not len(current):
        return []
    # Limit work on abnormal scans while retaining the untruncated mature tracker.
    previous = previous[previous[:, CLOCK] >= .80]
    current = current[current[:, CLOCK] >= .80]
    if not len(previous) or not len(current) or len(previous) * len(current) > 1_000_000:
        return []
    rows, cols, feature = edge_features(previous[:, :4], previous[:, :4], current[:, :4], initial=True)
    if not len(rows):
        return []
    score = model.predict(feature, num_threads=1)
    best_row = np.full(len(previous), -np.inf)
    best_col = np.full(len(current), -np.inf)
    np.maximum.at(best_row, rows, score)
    np.maximum.at(best_col, cols, score)
    best = (score == best_row[rows]) & (score == best_col[cols])
    unique_rows = np.bincount(rows[score == best_row[rows]], minlength=len(previous)) == 1
    unique_cols = np.bincount(cols[score == best_col[cols]], minlength=len(current)) == 1
    candidates = np.flatnonzero(best & unique_rows[rows] & unique_cols[cols] & (score >= .98))
    other = np.ones(len(score), dtype=bool)
    other[candidates] = False
    second_row = np.full(len(previous), -np.inf)
    second_col = np.full(len(current), -np.inf)
    np.maximum.at(second_row, rows[other], score[other])
    np.maximum.at(second_col, cols[other], score[other])
    selected = [i for i in candidates
                if score[i] - max(second_row[rows[i]], second_col[cols[i]]) >= .16 and .014 <= feature[i, 5] <= .80]
    return [(previous[rows[i]], current[cols[i]]) for i in selected]


def find_stranded_owner(active, path):
    """Identify a unique coasting ID compatible with a complete new observation path."""
    head, _, last = path
    scans = np.asarray([point[SCAN] for point in path])
    positions = np.asarray([point[POSITION] for point in path])
    observed_velocity = teacher.cv.residual(last[POSITION], head[POSITION]) / (last[SCAN] - head[SCAN])
    noise = np.array([.0055, .034, .0020])
    options = []
    for track in active:
        if track.hits < 6 or not 1 <= int(head[SCAN]) - track.circle <= 3 or len(track.history) < 4:
            continue
        old = list(track.history)[-5:]
        anchor = old[-1][2]
        dt = np.asarray([row[0] - old[-1][0] for row in old], dtype=float)
        y = teacher.cv.residual(np.asarray([row[2] for row in old]), anchor)
        x = dt - dt.mean()
        velocity = (x @ y) / (x @ x)
        pred = anchor + y.mean(axis=0) + (scans - old[-1][0] - dt.mean())[:, None] * velocity
        residual = teacher.cv.residual(positions, pred) / noise
        score = (np.square(residual).sum(axis=1).mean() + .25 * np.square((observed_velocity - velocity) / noise).sum()
                 + .7 * abs(np.mean([point[AMPLITUDE] for point in path]) - track.mean_amplitude))
        options.append((score, track))
    ranked = sorted(options, key=lambda option: option[0])
    if ranked and ranked[0][0] <= .90 and (len(ranked) == 1 or ranked[0][0] + .70 < ranked[1][0]):
        return ranked[0][1]
    return None


def collect_paths(spec):
    path, clock_path = spec
    path = validate_packet_path(path, TRAIN_PACKETS)
    from lightgbm import Booster

    data = read_packet(path)
    clock = Booster(model_file=str(clock_path))
    rng = np.random.default_rng(1417 + int(path.stem[-3:]))
    by_id = {int(p): int(label) for p, label in data[:, 7:9]}
    pool = PathPool()
    features, truth = [], []
    for scan in np.split(data, np.flatnonzero(np.diff(data[:, 6])) + 1):
        circle = int(scan[0, 6])
        p = eligible(measurements(scan[:, 1:4].astype(np.float32).astype(float),
                                  scan[:, 4].astype(np.float32).astype(float), scan[:, 0], scan[:, 7], circle, clock))
        proposal = pool.propose(p, circle)
        if proposal is not None:
            first, middle, cols, x = proposal
            a, b, c = (np.fromiter((by_id[key] for key in rows[:, POINT]), dtype=int, count=len(rows))
                       for rows in (first, middle, p[cols]))
            y = (a > 0) & (a == b) & (a == c)
            near_target = (a > 0) & (b > 0) & (c > 0)
            keep = (a >= 0) & (b >= 0) & (c >= 0) & (y | (rng.random(len(x)) < np.where(near_target, .8, .2)))
            features.append(x[keep])
            truth.append(y[keep])
        pool.add(p, circle)
    x, y = np.concatenate(features), np.concatenate(truth)
    positives, negatives = np.flatnonzero(y), np.flatnonzero(~y)
    sample = np.r_[rng.choice(positives, min(20000, len(positives)), replace=False),
                   rng.choice(negatives, min(42000, len(negatives)), replace=False)]
    print(path.name, 'raw samples', len(x), 'positive', int(y.sum()), flush=True)
    return x[sample], y[sample]


class BirthOnline:
    def __init__(self, teacher_model, pair_model, birth_model, clock_model, floor, mode,
                 paired_birth=False, joint=False, two_hit_pairs=False, reanchor=False, fast_seeds=None,
                 pairing=None, raw_gate=None, gap_recovery=None, birth_gap=2, retain_ambiguous_seeds=False,
                 association_floor=.20, association_margin=.30, supported_birth_floor=None,
                 support_legacy_confirmation=False):
        self.teacher_model = teacher_model
        self.corrector = PairCorrector(pair_model, cutoff=.85)
        self.matcher = self.corrector
        if joint:
            from joint_elevation_pairs import JointElevationMatcher

            self.matcher = JointElevationMatcher(pair_swap_probe, self.corrector)
        self.birth = birth_model
        self.clock = clock_model
        self.floor = floor
        self.mode = mode
        self.paired_birth = paired_birth
        self.two_hit_pairs = two_hit_pairs
        self.reanchor = reanchor
        self.fast_seeds = fast_seeds
        self.pairing = pairing
        self.raw_gate = raw_gate
        self.gap_recovery = gap_recovery
        self.birth_gap = birth_gap
        self.retain_ambiguous_seeds = retain_ambiguous_seeds
        self.association_floor = association_floor
        self.association_margin = association_margin
        self.supported_birth_floor = supported_birth_floor
        self.support_legacy_confirmation = support_legacy_confirmation
        self.confirmations = 0
        self.paired_confirmations = 0
        self.early_confirmations = 0
        self.reanchored = 0
        self.fast_confirmations = 0

    @staticmethod
    def confirm(history, active, output, ident, observe):
        first, *rest = history
        track = teacher.spawn(first[POSITION], first[AMPLITUDE], int(first[SCAN]))
        for point in rest:
            if observe:
                observe(track, point[POSITION], point[AMPLITUDE], int(point[SCAN]))
            teacher.update(track, point[POSITION], point[AMPLITUDE], int(point[SCAN]))
        track.batch_id = ident
        active.append(track)
        output[int(rest[-1][POINT])] = ident

    def promote_legacy(self, points, state, active, position, amp, circle, remaining, output, next_id, observe,
                       reserved_keys):
        _, tentative, seeds = state.get('tracks', ([], [], []))
        history = state.get('young_keys', {})
        next_tentative, pending_keys, taken = [], {}, set()
        for initial, tracks in ((False, tentative), (True, seeds)):
            tracks = [track for track in tracks if circle - track.circle == 1
                      and not reserved_keys.intersection(history[id(track)])]
            rest = np.flatnonzero(remaining)
            matches = teacher.cv.associate(tracks, position[rest], amp[rest], circle, initial=initial)
            for row, col in matches:
                track, index = tracks[row], rest[col]
                ident = points[index]['PntID']
                keys = (*history[id(track)], ident)
                corroborated = False
                if self.support_legacy_confirmation and not initial:
                    if all(key in state['supported_legacy_keys'] for key in keys):
                        forecast = track.position + track.velocity * (circle - track.circle)
                        error = teacher.cv.residual(position[index], forecast) / np.array([.010, .080, .004])
                        corroborated = error @ error <= 1.
                if observe:
                    observe(track, position[index], amp[index], circle)
                teacher.update(track, position[index], amp[index], circle)
                remaining[index] = False
                if initial:
                    next_tentative.append(track)
                    pending_keys[id(track)] = keys
                    continue
                motion = teacher.cv.residual(position[index], track.first_position)
                if (corroborated or (track.mean_amplitude >= teacher.cv.CONFIRM_MIN_MEAN_AMPLITUDE and
                                    np.linalg.norm(motion / CONFIRM_MOVEMENT) >= 1.)):
                    track.batch_id = next_id
                    next_id += 1
                    active.append(track)
                    output[ident] = track.batch_id
                    taken.update(keys)
        return next_tentative, pending_keys, taken, next_id

    def track(self, points, state, observe=None):
        circle = points[0]['CircleID'] if points else state.get('circle', -1) + 1
        state['circle'] = circle
        dormant = self.pairing.retained(state) if self.pairing else set()
        active = [track for track in state.get('tracks', ([], [], []))[0]
                  if circle - track.circle <= (12 if track.batch_id in dormant else 7)]
        if 'birth_path_pool' not in state:
            state['birth_path_pool'] = PathPool(self.birth_gap)
        pool = state['birth_path_pool']
        position = np.array([(p['Azimuth'], p['Elevation'], p['Distance']) for p in points]).reshape(-1, 3)
        amp = np.fromiter((p['Amp'] for p in points), dtype=float)
        times = np.fromiter((p['Time'] for p in points), dtype=np.int64)
        keys = np.fromiter((p['PntID'] for p in points), dtype=np.int64)
        raw = measurements(position, amp, times, keys, circle, self.clock)
        pair_frame = self.pairing.begin(active, raw, times, circle, state) if self.pairing else None
        if self.raw_gate and hasattr(self.raw_gate, 'probabilities'):
            return_probability = self.raw_gate.probabilities(position, amp, times)
            state['raw_return_probability'] = return_probability
            available = np.isnan(return_probability) | (return_probability >= self.raw_gate.floor)
        elif self.raw_gate:
            available = self.raw_gate(position, amp, times)
        else:
            available = np.ones(len(points), dtype=bool)
        locked = pair_frame.locked if pair_frame else []
        if pair_frame:
            for _, current_group in pair_frame.decisions:
                available[pair_frame.columns[current_group]] = True
        trusted_birth_points = None
        if self.supported_birth_floor is not None or self.support_legacy_confirmation:
            recent = state.setdefault('supported_birth_measurements', deque())
            while recent and circle - recent[0][0] > 2 * self.birth_gap:
                recent.popleft()
            minimum = .98 if self.support_legacy_confirmation else .94
            recent.append((circle, set(keys[state['raw_return_probability'] >= minimum])))
            trusted = set().union(*(scan_points for _, scan_points in recent))
            if self.support_legacy_confirmation:
                state['supported_legacy_keys'] = trusted
            if self.supported_birth_floor is not None:
                trusted_birth_points = np.fromiter(trusted, dtype=np.int64)
        if locked or dormant or not available.all():
            open_rows = np.asarray([circle - track.circle <= 7 for track in active], dtype=bool)
            open_cols = available.copy()
            for row, col, _ in locked:
                open_rows[row] = open_cols[col] = False
            rows, cols = np.flatnonzero(open_rows), np.flatnonzero(open_cols)
            loose, held = self.matcher.associate([active[i] for i in rows], position[cols], amp[cols], circle,
                                                 self.teacher_model, self.association_floor, 0., self.association_margin)
            matches = locked + [(rows[row], cols[col], probability) for row, col, probability in loose]
            held = cols[held]
        else:
            matches, held = self.matcher.associate(active, position, amp, circle, self.teacher_model,
                                                   self.association_floor, 0., self.association_margin)
        shadow_seeds = np.empty(0, dtype=int)
        if self.retain_ambiguous_seeds and len(held):
            raw_q = state['raw_return_probability']
            shadow_seeds = held[raw_q[held] >= .97]
        if pair_frame:
            matches, held = self.pairing.protect_fallback(pair_frame, active, position, circle, matches, held)
        if self.gap_recovery:
            matches += self.gap_recovery.associate(active, raw, available, matches, held, dormant,
                                                   circle, self.teacher_model)
        remaining = available
        remaining[held] = False
        output, association = {}, {}
        for row, col, probability in matches:
            track = active[row]
            if observe:
                observe(track, position[col], amp[col], circle)
            teacher.update(track, position[col], amp[col], circle, probability=probability, robust=4.)
            remaining[col] = False
            ident = points[col]['PntID']
            output[ident] = track.batch_id
            association[ident] = probability

        next_tentative, pending_keys = [], {}
        claimed = set()
        next_id = state.get('next_batch_id', 1)
        if pair_frame:
            for previous, current in self.pairing.starts(pair_frame, remaining):
                for first, second in zip(previous, current):
                    self.confirm((first, raw[second]), active, output, next_id, observe)
                    next_id += 1
                    claimed.update((first[POINT], raw[second, POINT]))
                    remaining[second] = False
        candidates = eligible(raw[remaining])
        if self.mode == 'supplement':
            next_tentative, pending_keys, legacy_claimed, next_id = self.promote_legacy(
                points, state, active, position, amp, circle, remaining, output, next_id, observe, claimed)
            claimed.update(legacy_claimed)
            candidates = candidates[~np.isin(candidates[:, POINT], np.fromiter(claimed, dtype=np.int64))]
        proposal = pool.propose(candidates, circle)
        if proposal is not None:
            head, middle, cols, feature = proposal
            probability = self.birth.predict(feature, num_threads=1)
            enough = probability >= self.floor
            if trusted_birth_points is not None and len(trusted_birth_points):
                precise = ((feature[:, -3] < 2.7) & (feature[:, -1] < 2.5)
                           & np.isin(head[:, POINT], trusted_birth_points)
                           & np.isin(middle[:, POINT], trusted_birth_points)
                           & np.isin(candidates[cols, POINT], trusted_birth_points))
                enough |= precise & (probability >= self.supported_birth_floor)
            ready = np.flatnonzero(enough)
            paired = supported_birth_pairs(head, middle, candidates, cols, probability) if self.paired_birth else []
            ordered = ((index,) for index in ready[np.argsort(-probability[ready], kind='stable')])
            for selection in (*paired, *ordered):
                trajectories = [(head[i], middle[i], candidates[cols[i]]) for i in selection]
                keys = {part[POINT] for history in trajectories for part in history}
                if keys & claimed or len(keys) != len(selection) * 3:
                    continue
                claimed.update(keys)
                if len(selection) == 2:
                    self.paired_confirmations += 2
                for history, index in zip(trajectories, selection):
                    origin = find_stranded_owner(active, history) if self.reanchor and probability[index] >= .87 else None
                    if origin is not None:
                        for obs in history:
                            observation_circle = int(obs[SCAN])
                            if observe:
                                observe(origin, obs[POSITION], obs[AMPLITUDE], observation_circle)
                            teacher.update(origin, obs[POSITION], obs[AMPLITUDE], observation_circle)
                        output[int(history[-1][POINT])] = origin.batch_id
                        self.reanchored += 1
                    else:
                        self.confirm(history, active, output, next_id, observe)
                        next_id += 1
                        self.confirmations += 1
        taken = np.fromiter(claimed, dtype=np.int64)
        unused = candidates[~np.isin(candidates[:, POINT], taken)]
        if len(shadow_seeds):
            unused = np.vstack((unused, raw[shadow_seeds]))
        pool.add(unused, circle, taken)
        if self.two_hit_pairs and pool.arcs and pool.arcs[-1][0] == circle:
            arcs = pool.arcs[-1][1]
            if len(shadow_seeds):
                arcs = arcs.subset(~np.isin(arcs.middle[:, POINT], raw[shadow_seeds, POINT]))
            earlier = set()
            for pair in supported_two_hit_pairs(arcs):
                histories = [(arcs.first[i], arcs.middle[i]) for i in pair]
                keys = {p[POINT] for history in histories for p in history}
                if len(keys) != 4 or keys & earlier:
                    continue
                earlier.update(keys)
                for history in histories:
                    self.confirm(history, active, output, next_id, observe)
                    next_id += 1
                    self.early_confirmations += 1
            if earlier:
                claimed.update(earlier)
                taken = np.fromiter(claimed, dtype=np.int64)
                pool.add(np.empty((0, 7)), circle, np.fromiter(earlier, dtype=np.int64))
        if self.fast_seeds is not None:
            prior = next((p for t, p in pool.observations if t == circle - 1), np.empty((0, 7)))
            now = next((p for t, p in pool.observations if t == circle), np.empty((0, 7)))
            if len(shadow_seeds):
                now = now[~np.isin(now[:, POINT], raw[shadow_seeds, POINT])]
            earlier = set()
            for history in confident_two_hit_starts(prior, now, self.fast_seeds):
                keys = {point[POINT] for point in history}
                if len(keys) != 2 or keys & earlier:
                    continue
                earlier.update(keys)
                self.confirm(history, active, output, next_id, observe)
                next_id += 1
                self.fast_confirmations += 1
            if earlier:
                claimed.update(earlier)
                taken = np.fromiter(claimed, dtype=np.int64)
                pool.add(np.empty((0, 7)), circle, np.fromiter(earlier, dtype=np.int64))
        new_seeds = []
        if self.mode == 'supplement':
            next_tentative = [track for track in next_tentative if not claimed.intersection(pending_keys[id(track)])]
            pending_keys = {id(track): pending_keys[id(track)] for track in next_tentative}
            available = remaining & ~np.isin(raw[:, POINT], taken) & (amp >= teacher.cv.SEED_MIN_AMPLITUDE)
            for i in np.flatnonzero(available):
                track = teacher.spawn(position[i], amp[i], circle)
                new_seeds.append(track)
                pending_keys[id(track)] = (points[i]['PntID'],)
        state['young_keys'] = pending_keys
        state['tracks'] = active, next_tentative, new_seeds
        state['next_batch_id'] = next_id
        state['association_confidence'] = association
        if pair_frame:
            self.pairing.remember(pair_frame, raw, output, circle, state)
        return [{**{k: p[k] for k in teacher.cv.OUTPUT_FIELDS}, 'BatchID': output[p['PntID']]}
                for p in points if p['PntID'] in output]


def evaluate(spec):
    path, args = spec
    from lightgbm import Booster

    fast = getattr(args, 'fast_seeds', None)
    fast_model = Booster(model_file=str(fast)) if fast else None
    run = BirthOnline(*(Booster(model_file=str(p)) for p in (args.teacher, args.pair, args.model, args.clock)),
                      args.floor, args.mode, getattr(args, 'paired_birth', False), getattr(args, 'joint', False),
                      getattr(args, 'two_hit_pairs', False), getattr(args, 'reanchor', False), fast_model)
    with (patch.object(tracker_compat, 'baseline_circle', run.track),
          patch.object(replay, 'load_handler', return_value=load_runtime().process_circle)):
        stats = replay.replay_packet(path, allowed=TUNING_PACKETS)
    print(path.name, round(stats['score']['total'], 4), 'births', run.confirmations,
          'paired births', run.paired_confirmations, 'switches', stats['id_switches'], flush=True)
    stats['path_confirmations'] = run.confirmations
    stats['paired_path_confirmations'] = run.paired_confirmations
    stats['two_hit_confirmations'] = run.early_confirmations
    stats['reanchored_paths'] = run.reanchored
    stats['fast_seed_confirmations'] = run.fast_confirmations
    return stats


def parallel(worker, args, jobs):
    if jobs == 1:
        return list(map(worker, args))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(worker, args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'tune'))
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--clock', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, default=Path('experimental/teacher_assoc_wide.txt'))
    parser.add_argument('--pair', type=Path, default=Path('experimental/models/pair_swap_wide.txt'))
    parser.add_argument('--floor', type=float, default=.74)
    parser.add_argument('--mode', choices=('replace', 'supplement'), default='replace')
    parser.add_argument('--paired-birth', action='store_true')
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--two-hit-pairs', action='store_true')
    parser.add_argument('--reanchor', action='store_true')
    parser.add_argument('--fast-seeds', type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1 or not 0 < args.floor < 1:
        parser.error('invalid floor or job count')
    training = args.phase == 'train'
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS if training else TUNING_PACKETS)
    if training:
        from lightgbm import Dataset, train

        examples = parallel(collect_paths, [(p, args.clock) for p in paths], args.jobs)
        x = np.vstack([x for x, _ in examples])
        y = np.concatenate([y for _, y in examples])
        parameters = dict(objective='binary', learning_rate=.07, num_leaves=31, min_data_in_leaf=72,
                          num_threads=args.jobs, force_col_wise=True, deterministic=True, verbosity=-1)
        model = train(parameters, Dataset(x, label=y, feature_name=list(FEATURES)), num_boost_round=180)
        model.save_model(str(args.model))
        result = dict(sampled=len(x), positives=int(y.sum()), features=FEATURES, training_params=parameters,
                      boosting_rounds=180)
    else:
        reports = parallel(evaluate, [(p, args) for p in paths], args.jobs)
        result = dict(summary=aggregate_packets(reports),
                      packets=[{k: v for k, v in p.items() if k not in ('tracks', '_circle_times')} for p in reports])
        summary = result['summary']
        print('total', summary['score']['total'], 'switches', summary['id_switches'],
              'splits', summary['split_truth_tracks'], flush=True)
    asset_paths = ('model', 'clock') if training else ('model', 'clock', 'teacher', 'pair')
    if not training and args.fast_seeds:
        asset_paths += ('fast_seeds',)
    result['metadata'] = dict(packet_ids=args.packets, floor=args.floor, work=args.phase, mode=args.mode,
                              paired_birth=args.paired_birth, joint=args.joint,
                              two_hit_pairs=args.two_hit_pairs,
                              reanchor=args.reanchor,
                              sha256={k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest() for k in asset_paths})
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
