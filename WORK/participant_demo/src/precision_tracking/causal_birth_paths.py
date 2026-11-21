"""Causal track confirmation and the per-scan precision tracking engine."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from tracker_assoc import spatial_edges

from . import teacher_assoc as teacher
from .clock_association import phase_features
from .joint_elevation_pairs import JointElevationMatcher
from .pair_swap_probe import PairCorrector

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


class BirthOnline:
    """Maintain mature trajectories and confirm disjoint causal birth paths."""

    def __init__(self, teacher_model, pair_model, birth_model, clock_model, floor=.86,
                 mode='supplement', paired_birth=True, joint=True, two_hit_pairs=True,
                 pairing=None, raw_gate=None, association_floor=.035, association_margin=.30,
                 residual_linker=None, association_robustness=4.):
        self.teacher_model = teacher_model
        corrector = PairCorrector(pair_model, cutoff=.85)
        self.matcher = JointElevationMatcher(corrector) if joint else corrector
        self.birth = birth_model
        self.clock = clock_model
        self.floor = floor
        self.mode = mode
        self.paired_birth = paired_birth
        self.two_hit_pairs = two_hit_pairs
        self.pairing = pairing
        self.raw_gate = raw_gate
        self.association_floor = association_floor
        self.association_margin = association_margin
        self.association_robustness = association_robustness
        self.residual_linker = residual_linker

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
                if observe:
                    observe(track, position[index], amp[index], circle)
                teacher.update(track, position[index], amp[index], circle)
                remaining[index] = False
                if initial:
                    next_tentative.append(track)
                    pending_keys[id(track)] = keys
                    continue
                motion = teacher.cv.residual(position[index], track.first_position)
                if (track.mean_amplitude >= teacher.cv.CONFIRM_MIN_MEAN_AMPLITUDE and
                        np.linalg.norm(motion / CONFIRM_MOVEMENT) >= 1.):
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
            state['birth_path_pool'] = PathPool()
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
        rejected = ()
        if pair_frame:
            original = matches
            matches, held = self.pairing.protect_fallback(pair_frame, active, position, circle, matches, held)
            if self.residual_linker is not None:
                accepted = {(row, col) for row, col, _ in matches}
                rejected = [(row, col, prob) for row, col, prob in original if (row, col) not in accepted]
        if self.residual_linker is not None and "raw_return_probability" in state:
            recoveries = self.residual_linker.recover(
                active, raw, times, state["raw_return_probability"], circle, pair_frame,
                state.get("previous_pairs", ()), matches, held, rejected,
            )
            state["residual_matches"] = {int(keys[col]) for _, col, _ in recoveries}
            matches += recoveries
        remaining = available
        remaining[held] = False
        output, association = {}, {}
        for row, col, probability in matches:
            track = active[row]
            if observe:
                observe(track, position[col], amp[col], circle)
            teacher.update(track, position[col], amp[col], circle, probability=probability,
                           robust=self.association_robustness)
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
            ready = np.flatnonzero(probability >= self.floor)
            paired = supported_birth_pairs(head, middle, candidates, cols, probability) if self.paired_birth else []
            ordered = ((index,) for index in ready[np.argsort(-probability[ready], kind='stable')])
            for selection in (*paired, *ordered):
                trajectories = [(head[i], middle[i], candidates[cols[i]]) for i in selection]
                keys = {part[POINT] for history in trajectories for part in history}
                if keys & claimed or len(keys) != len(selection) * 3:
                    continue
                claimed.update(keys)
                for history in trajectories:
                    self.confirm(history, active, output, next_id, observe)
                    next_id += 1
        taken = np.fromiter(claimed, dtype=np.int64)
        pool.add(candidates[~np.isin(candidates[:, POINT], taken)], circle, taken)
        if self.two_hit_pairs and pool.arcs and pool.arcs[-1][0] == circle:
            arcs = pool.arcs[-1][1]
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
