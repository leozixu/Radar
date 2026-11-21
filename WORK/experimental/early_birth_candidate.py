"""Train and replay a causal two-observation start on unused precision-tracker points."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import json
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np

from algorithm import load_runtime
from baseline_cv import OUTPUT_FIELDS, residual
from metrics import SCORE_VERSION, aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from precision_tracking.causal_birth_paths import AMPLITUDE, CLOCK, POINT, POSITION, SCAN
from precision_tracking.runtime import PrecisionRuntime
import replay_validation as replay
from tracker_assoc import spatial_edges


LINK_WIDTH = np.array([.070, .155, .023])
INCUMBENT_WIDTH = np.array([.015, .080, .007])
MIN_RAW = .78
PAIR_ROLE_MIN = np.array([-.008, 0., -.0022, -.026])
PAIR_ROLE_MAX = np.array([.008, .006, .0022, -.000004])
PAIR_SIGNATURE_ERROR = .0000026
PAIR_TRANSFER_WIDTH = np.array([.110, .220, .022])
PAIR_TRANSFER_SHAPE = np.array([.012, .004, .0035])
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
    """The closest established prediction at the observation's own circle."""
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
    """Return edges that win both ends against all alternatives by a finite margin."""
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


def observation_key(circle, amplitude, position):
    return int(circle), float(amplitude), tuple(position)


def row_key(row):
    return observation_key(row[SCAN], row[AMPLITUDE], row[POSITION])


def unsupported_pair_starts(groups, unclaimed, probability, tracks, floor, circle):
    """Select complete, isolated pair observations without an assignment in this scan."""
    options = []
    for group in groups:
        rows = group.observations
        if rows[0, SCAN] != circle or any(group.owners):
            continue
        ids = tuple(int(pid) for pid in rows[:, POINT])
        if not unclaimed.issuperset(ids) or any(not probability.get(pid, 0.) >= floor for pid in ids):
            continue
        role = np.r_[residual(rows[1, POSITION], rows[0, POSITION]), rows[1, AMPLITUDE] - rows[0, AMPLITUDE]]
        if (role >= PAIR_ROLE_MIN).all() and (role <= PAIR_ROLE_MAX).all():
            options.append(rows)
    if not options:
        return []
    observations = np.stack(options)
    distant = incumbent_distance(observations.reshape(-1, 7), tracks).reshape(-1, 2) > 1.
    occurrences = Counter(int(pid) for pid in observations[:, :, POINT].ravel())
    return [rows for rows, distance in zip(observations, distant)
            if distance.all() and all(occurrences[int(pid)] == 1 for pid in rows[:, POINT])]


def supported_pair_aliases(provisional, native_groups, circle, public_ids):
    """Match sparse two-return signatures over a gap without reusing any native ID."""
    reverse = {public: internal for internal, public in public_ids.items()}
    choices = []
    for rows, issued in provisional:
        gap = circle - rows[0, SCAN]
        shape = residual(rows[1, POSITION], rows[0, POSITION])
        offset = rows[1, AMPLITUDE] - rows[0, AMPLITUDE]
        for group in native_groups:
            now = group.observations
            internal = group.owners
            if not group.confirmed or now[0, SCAN] != circle or internal[0] == internal[1]:
                continue
            if all(ident in public_ids for ident in internal):
                continue
            if any((ident in public_ids and public_ids[ident] != public)
                   or (public in reverse and reverse[public] != ident) for ident, public in zip(internal, issued)):
                continue
            offset_now = now[1, AMPLITUDE] - now[0, AMPLITUDE]
            if abs(offset_now - offset) > PAIR_SIGNATURE_ERROR:
                continue
            shift = abs(residual(now[:, POSITION], rows[:, POSITION]).mean(axis=0))
            delta = abs(residual(residual(now[1, POSITION], now[0, POSITION]), shape))
            if (shift <= PAIR_TRANSFER_WIDTH * (1. + .45 * (gap - 1))).all() and (delta <= PAIR_TRANSFER_SHAPE).all():
                choices.append((internal, issued))
    occurrences = Counter((kind, ident) for internal, issued in choices
                          for kind, item in enumerate((internal, issued)) for ident in item)
    return {ident: public for internal, issued in choices
            if all(occurrences[kind, ident] == 1 for kind, item in enumerate((internal, issued)) for ident in item)
            for ident, public in zip(internal, issued) if ident not in public_ids}


class EarlyBirthTracker:
    """Supplement ordinary births after their input points and incumbent tracks are reserved."""

    def __init__(self, original, birth_model, floor, margin, collector, mode, pair_floor, pair_transfer):
        self.original = original
        self.model = birth_model
        self.floor = floor
        self.margin = margin
        self.collector = collector
        self.mode = mode
        self.pair_floor = pair_floor
        self.pair_transfer = pair_transfer
        self.raw_by_circle = {}
        self.pairs_by_circle = {}
        self.added = []
        self.proposals = []
        self.pair_proposals = []
        self.provisional = {}
        self.provisional_pairs = []
        self.public_ids = {}
        self.transfers = 0
        self.initial_pair_hits = 0
        self.signature_transfers = 0

    def track(self, points, state, observe=None):
        self.proposals = []
        self.pair_proposals = []
        outputs = self.original.track(points, state, observe)
        circle = state['circle']
        self.raw_by_circle = {t: values for t, values in self.raw_by_circle.items() if circle - t <= 2}
        self.pairs_by_circle = {t: values for t, values in self.pairs_by_circle.items() if circle - t <= 2}
        if not points:
            return outputs
        pool = state['birth_path_pool']
        available = dict(pool.observations)
        recent = state['previous_pairs']
        self.pairs_by_circle[circle] = {int(pid) for pair in recent if pair.observations[0, SCAN] == circle
                                       for pid in pair.observations[:, POINT]}
        self.raw_by_circle[circle] = dict(zip((p['PntID'] for p in points), state['raw_return_probability']))
        if circle not in available or len(points) > 1600:
            return outputs
        active, tentative, seeds = state['tracks']
        if self.pair_floor is not None:
            self.pair_proposals = unsupported_pair_starts(
                recent, set(available[circle][:, POINT].astype(int)), self.raw_by_circle[circle], active,
                self.pair_floor, circle,
            )
        after, current_raw = points_with_evidence(available[circle], self.raw_by_circle[circle])
        previous = list(available.keys() - {circle})
        if not len(after) or not previous:
            return outputs
        parts = [points_with_evidence(available[t], self.raw_by_circle.get(t, {})) for t in sorted(previous)]
        before, previous_raw = np.vstack([rows for rows, _ in parts]), np.concatenate([q for _, q in parts])
        if not len(before) or len(before) + len(after) > 480:
            return outputs

        pair_points = tuple(set().union(*self.pairs_by_circle.values()))
        rows, cols, features = candidate_edges(before, after, previous_raw, current_raw, active, pair_points)
        if not len(rows):
            return outputs
        if self.collector is not None:
            self.collector(before[rows, POINT].astype(int), after[cols, POINT].astype(int), features)
        if self.model is None:
            return outputs

        scores = self.model.predict(features, num_threads=1)
        chosen = list(unique_choices(rows, cols, scores, self.floor, self.margin))
        if self.mode == 'provisional':
            self.proposals = [(before[rows[i]], after[cols[i]]) for i in chosen]
            return outputs
        matched, claimed = {}, set()
        next_id = state['next_batch_id']
        for i in chosen:
            head, tail = before[rows[i]], after[cols[i]]
            self.original.confirm((head, tail), active, matched, next_id, observe)
            ids = (int(head[POINT]), int(tail[POINT]))
            claimed.update(ids)
            self.added.append(ids)
            next_id += 1
        if not matched:
            return outputs

        state['next_batch_id'] = next_id
        pool.add(np.empty((0, 7)), circle, np.fromiter(claimed, dtype=np.int64))
        young = state['young_keys']
        tentative = [t for t in tentative if not claimed.intersection(young[id(t)])]
        seeds = [t for t in seeds if not claimed.intersection(young[id(t)])]
        state['tracks'] = active, tentative, seeds
        state['young_keys'] = {id(t): young[id(t)] for t in tentative + seeds}
        state['previous_pairs'] = [
            replace(pair, owners=tuple(matched.get(int(pid), owner)
                                       for pid, owner in zip(pair.observations[:, POINT], pair.owners)))
            if pair.observations[0, SCAN] == circle else pair for pair in recent
        ]
        outputs.extend({**{key: p[key] for key in OUTPUT_FIELDS}, 'BatchID': matched[p['PntID']]}
                       for p in points if p['PntID'] in matched)
        return outputs

    def publish(self, outputs, points, state):
        """Continue an early public ID only through shared original observations."""
        circle = state['circle']
        self.provisional = {key: value for key, value in self.provisional.items() if circle - key[0] <= 6}
        self.provisional_pairs = [(rows, ids) for rows, ids in self.provisional_pairs if circle - rows[0, SCAN] <= 8]
        active = {t.batch_id: t for t in state['tracks'][0]}
        attached = set()
        if self.pair_transfer and self.provisional_pairs:
            aliases = supported_pair_aliases(self.provisional_pairs, state['previous_pairs'], circle, self.public_ids)
            self.public_ids.update(aliases)
            attached.update(aliases.values())
            self.signature_transfers += len(aliases)
        for item in outputs:
            ident = item['BatchID']
            if ident not in self.public_ids:
                anchors = {self.provisional.get(observation_key(*observation)) for observation in active[ident].history}
                available = anchors - {None} - attached
                public = available.pop() if len(available) == 1 else ident
                self.public_ids[ident] = public
                if public != ident:
                    attached.add(public)
                    self.transfers += 1
            item['BatchID'] = self.public_ids[ident]
        if attached:
            self.provisional = {key: value for key, value in self.provisional.items() if value not in attached}

        next_id = state['next_batch_id']
        emitted = {out['PntID'] for out in outputs}
        used = {out['BatchID'] for out in outputs}
        current = {p['PntID']: p for p in points} if self.proposals or self.pair_proposals else {}
        for before, after in self.proposals:
            point_id = int(after[POINT])
            if point_id in emitted:
                continue
            public = self.provisional.get(row_key(before))
            if public in used:
                continue
            if public is None:
                public = next_id
                next_id += 1
            point = current[point_id]
            outputs.append({**{key: point[key] for key in OUTPUT_FIELDS}, 'BatchID': public})
            used.add(public)
            emitted.add(point_id)
            self.provisional.setdefault(row_key(before), public)
            self.provisional[row_key(after)] = public
            self.added.append((int(before[POINT]), point_id))
        previous = [key for key in self.provisional if key[0] < circle]
        for rows in self.pair_proposals:
            ids = tuple(int(pid) for pid in rows[:, POINT])
            if emitted.intersection(ids) or any(self.conflicts(row, previous) for row in rows):
                continue
            if self.pair_transfer:
                self.provisional_pairs.append((rows, (next_id, next_id + 1)))
            for row, point_id in zip(rows, ids):
                point = current[point_id]
                outputs.append({**{key: point[key] for key in OUTPUT_FIELDS}, 'BatchID': next_id})
                self.provisional[row_key(row)] = next_id
                next_id += 1
                self.initial_pair_hits += 1
        state['next_batch_id'] = next_id
        return outputs

    @staticmethod
    def conflicts(row, previous):
        """Suppress a new provisional ID if an unresolved earlier observation could own it."""
        if not previous:
            return False
        position = np.asarray([key[2] for key in previous])
        gap = row[SCAN] - np.fromiter((key[0] for key in previous), dtype=int)
        error = residual(row[POSITION], position) / (LINK_WIDTH * np.sqrt(gap[:, None]))
        return bool((np.sum(error**2, axis=1) <= 1.).any())


@dataclass(frozen=True, slots=True)
class EarlyBirthRuntime(PrecisionRuntime):
    birth_model: object
    floor: float
    margin: float
    collector: object = None
    mode: str = 'confirm'
    pair_floor: float | None = None
    pair_transfer: bool = False

    def new_packet(self):
        tracker, history = PrecisionRuntime.new_packet(self)
        return EarlyBirthTracker(tracker, self.birth_model, self.floor, self.margin, self.collector, self.mode,
                                 self.pair_floor, self.pair_transfer), history

    def process_circle(self, points, state):
        outputs = PrecisionRuntime.process_circle(self, points, state)
        if self.mode == 'provisional':
            tracker, _ = state['_precision_packet']
            return tracker.publish(outputs, points, state)
        return outputs


def trial_runtime(model=None, floor=.9, margin=.15, collector=None, mode='confirm', pair_floor=None,
                  pair_transfer=False):
    original = load_runtime()
    return EarlyBirthRuntime(original.settings, original.models, original.metadata, model, floor, margin, collector,
                             mode, pair_floor, pair_transfer)


def collect(path):
    labels, circles, state = {}, {}, {}
    seen, data, targets = set(), [], []

    def record(head, tail, features):
        old = np.fromiter((labels[i] for i in head), dtype=int)
        new = np.fromiter((labels[i] for i in tail), dtype=int)
        fresh = np.fromiter((i not in seen for i in new), dtype=bool)
        data.append(features)
        targets.append((old > 0) & (old == new) & fresh)

    runtime = trial_runtime(collector=record)
    for _, payload in replay.iter_circle_payloads(path, labels, circles, TRAIN_PACKETS):
        points = [dict(zip(replay.POINT_KEYS, row)) for row in replay.POINT.iter_unpack(payload)]
        returns = runtime.process_circle(points, state)
        seen.update(labels[p['PntID']] for p in returns if labels[p['PntID']] > 0)
    x = np.concatenate(data) if data else np.empty((0, len(FEATURES)), dtype=np.float32)
    y = np.concatenate(targets) if targets else np.empty(0, dtype=bool)
    print(path.name, 'training edges', len(y), 'new true paths', int(y.sum()), flush=True)
    return x, y


def evaluate(spec):
    path, model_path, floor, margin, mode, pair_floor, pair_transfer = spec
    model = lgb.Booster(model_file=str(model_path)) if floor is not None else None
    runtime = (trial_runtime(model, floor, margin, mode=mode, pair_floor=pair_floor, pair_transfer=pair_transfer)
               if model is not None else load_runtime())
    states = []

    def process(points, state):
        if not states:
            states.append(state)
        return runtime.process_circle(points, state)

    with patch.object(replay, 'load_handler', return_value=process):
        report = replay.replay_packet(path, allowed=TUNING_PACKETS)
    tracker = states[0]['_precision_packet'][0] if model is not None else None
    added = len(tracker.added) if tracker is not None else 0
    report['added_early_paths'] = added
    report['transferred_public_ids'] = tracker.transfers if tracker is not None else 0
    report['initial_pair_hits'] = tracker.initial_pair_hits if tracker is not None else 0
    report['signature_transfers'] = tracker.signature_transfers if tracker is not None else 0
    print(path.name, floor, round(report['score']['total'], 5), 'new paths', added, 'split',
          report['split_truth_tracks'], 'transferred', report['transferred_public_ids'], 'pair initial hits',
          report['initial_pair_hits'], 'signature transfers', report['signature_transfers'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'trial'))
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--floors', type=float, nargs='+', default=(.90,))
    parser.add_argument('--margin', type=float, default=.15)
    parser.add_argument('--mode', choices=('confirm', 'provisional'), default='confirm')
    parser.add_argument('--pair-floor', type=float)
    parser.add_argument('--pair-transfer', action='store_true')
    parser.add_argument('--skip-control', action='store_true', help='Use the recorded identical-packet control.')
    args = parser.parse_args()
    scope = TRAIN_PACKETS if args.phase == 'fit' else TUNING_PACKETS
    paths = packet_paths(args.data_dir, args.packets, scope)
    report = {'metadata': {'packet_ids': args.packets, 'score_version': SCORE_VERSION,
                           'source_sha256': replay.file_sha256(Path(__file__)), 'features': FEATURES}}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if args.phase == 'fit':
            samples = list(pool.map(collect, paths))
            x = np.concatenate([p[0] for p in samples])
            y = np.concatenate([p[1] for p in samples])
            training = lgb.Dataset(x, label=y, feature_name=list(FEATURES))
            model = lgb.train(dict(objective='binary', learning_rate=.075, num_leaves=19,
                                   min_data_in_leaf=35, num_threads=1, verbosity=-1,
                                   deterministic=True, force_col_wise=True, seed=31), training, num_boost_round=170)
            model.save_model(str(args.model))
            report['training'] = dict(edges=len(y), new_true_paths=int(y.sum()))
        else:
            floors = tuple(args.floors) if args.skip_control else (None, *args.floors)
            specs = [(path, args.model, floor, args.margin, args.mode, args.pair_floor, args.pair_transfer)
                     for floor in floors for path in paths]
            results = list(pool.map(evaluate, specs))
            report['metadata']['margin'] = args.margin
            report['metadata']['mode'] = args.mode
            report['metadata']['pair_floor'] = args.pair_floor
            report['metadata']['pair_transfer'] = args.pair_transfer
            report['variants'] = {}
            for index, floor in enumerate(floors):
                variant = results[index * len(paths):(index + 1) * len(paths)]
                name = 'baseline' if floor is None else str(floor)
                summary = aggregate_packets(variant)
                report['variants'][name] = dict(summary=summary, packets=[
                    {k: v for k, v in row.items() if k not in ('tracks', '_circle_times')} for row in variant])
                print(name, json.dumps(summary['score']['weighted_points']), summary['score']['total'], flush=True)
    report['metadata']['model_sha256'] = replay.file_sha256(args.model)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
