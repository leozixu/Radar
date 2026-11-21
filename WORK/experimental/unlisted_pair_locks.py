"""Audit and replay confirmed pair locks omitted by the normal pair detector."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from frozen_miss_audit import (
    DEFAULT_DATA_DIR, POINT, POINT_KEYS, SCORE_VERSION, TRAIN_PACKETS, WORK,
    aggregate_packets, encode_tracks, evaluate_packet, file_sha256, iter_circle_payloads, packet_paths,
)


OFFSET_TOLERANCE = .0000026
CENTER_WIDTH = np.array((.028, .065, .012))
ARM_WIDTH = np.array((.045, .110, .015))
ROLE_WIDTH = np.array((.012, .004, .0035))
MAX_TICK_GAP = 8
PROBABILITY = .97


def paired_candidates(frame, tracks, raw, stamps, circle, priors):
    from baseline_cv import residual
    from precision_tracking.causal_birth_paths import AMPLITUDE, POSITION, SCAN
    from precision_tracking.paired_returns import MAX_GAP

    if len(raw) < 2 or not priors:
        return []
    known = {tuple(sorted(map(int, record))) for record in frame.columns}
    taken_priors = {id(prior) for prior, _ in frame.decisions}
    taken_points = {point for _, index in frame.decisions for point in frame.columns[index]}
    taken_points.update(col for _, col, _ in frame.locked)
    taken_rows = {row for row, _, _ in frame.locked}
    open_rows = {track.batch_id: row for row, track in enumerate(tracks) if row not in taken_rows}
    open_cols = np.ones(len(raw), dtype=bool)
    open_cols[list(taken_points)] = False
    available = np.flatnonzero(open_cols)
    coordinates = raw[available, POSITION]
    choices = []
    for prior in priors:
        if (not prior.confirmed or id(prior) in taken_priors or prior.owners[0] == prior.owners[1]
                or not all(owner in open_rows for owner in prior.owners)
                or not 0 < circle - prior.observations[0, SCAN] <= MAX_GAP):
            continue
        rows = tuple(open_rows[owner] for owner in prior.owners)
        arms = [tracks[row] for row in rows]
        lags = np.array([circle - arm.circle for arm in arms])
        predicted = np.array([arm.position + lag * arm.velocity for arm, lag in zip(arms, lags)])
        scale = 1 + .35 * np.maximum(0, lags - 1)
        proximity = np.abs(residual(coordinates[None, :, :], predicted[:, None, :]))
        left, right = (available[np.all(distance <= ARM_WIDTH * width, axis=1)]
                       for distance, width in zip(proximity, scale))
        if not len(left) or not len(right):
            continue
        before = prior.observations
        offset = before[1, AMPLITUDE] - before[0, AMPLITUDE]
        if abs(offset) <= 2 * OFFSET_TOLERANCE:
            continue
        # Index the second arm's amplitude to avoid Cartesian products on dense scans.
        right = right[np.argsort(raw[right, AMPLITUDE], kind='stable')]
        center = raw[left, AMPLITUDE] + offset
        lower, upper = np.searchsorted(raw[right, AMPLITUDE], center[:, None]
                                       + (-OFFSET_TOLERANCE, OFFSET_TOLERANCE)).T
        old_role = residual(before[1, POSITION], before[0, POSITION])
        for a, start, stop in zip(left, lower, upper):
            for b in right[start:stop]:
                if a == b or not 0 <= stamps[b] - stamps[a] <= MAX_TICK_GAP:
                    continue
                if tuple(sorted((a, b))) in known:
                    continue
                after = raw[[a, b], POSITION]
                role_change = residual(residual(after[1], after[0]), old_role)
                center_error = residual(after, predicted).mean(axis=0)
                if (np.abs(role_change) <= ROLE_WIDTH).all() and (
                        np.abs(center_error) <= CENTER_WIDTH * max(scale)).all():
                    choices.append((prior, rows, (int(a), int(b))))

    owners = Counter(row for _, rows, _ in choices for row in rows)
    points = Counter(col for _, _, columns in choices for col in columns)
    return [(prior, rows, columns) for prior, rows, columns in choices
            if all(owners[row] == 1 for row in rows) and all(points[col] == 1 for col in columns)]


class PairReplay:
    def __init__(self, pairing, labels, variant):
        self.original = pairing.begin
        self.labels = labels
        self.variant = variant
        self.unlisted_oracle = {}
        self.locked = {}
        self.counts = Counter()

    def begin(self, tracks, raw, stamps, circle, state):
        from baseline_cv import residual
        from precision_tracking.causal_birth_paths import AMPLITUDE, POINT, POSITION

        frame = self.original(tracks, raw, stamps, circle, state)
        priors = state.get('previous_pairs', ())
        if self.variant:
            added = paired_candidates(frame, tracks, raw, stamps, circle, priors)
            if added:
                new_columns = np.asarray([columns for _, _, columns in added], dtype=int)
                size = len(frame.columns)
                frame.columns = np.concatenate((frame.columns, new_columns))
                for index, (prior, rows, columns) in enumerate(added, start=size):
                    frame.decisions.append((prior, index))
                    frame.locked.extend((row, col, PROBABILITY) for row, col in zip(rows, columns))
                    self.counts['extra_pairs_locked'] += 1
                    current = [int(raw[col, POINT]) for col in columns]
                    prior_truth = [self.labels[int(pid)] for pid in prior.observations[:, POINT]]
                    self.counts['extra_pair_both_correct'] += all(
                        self.labels[pid] == truth > 0 for pid, truth in zip(current, prior_truth))
                    for pid, truth, row in zip(current, prior_truth, rows):
                        self.locked[pid] = int(tracks[row].batch_id)
                        self.counts['extra_correct_truth_links'] += self.labels[pid] == truth > 0
                        self.counts['extra_false_links'] += self.labels[pid] <= 0
            return frame

        existing = {tuple(sorted(map(int, pair))) for pair in frame.columns}
        used_rows = {row for row, _, _ in frame.locked}
        used_cols = {col for _, col, _ in frame.locked}
        used_cols.update(col for _, index in frame.decisions for col in frame.columns[index])
        available = {track.batch_id: track for i, track in enumerate(tracks) if i not in used_rows}
        hits = defaultdict(list)
        for col, pid in enumerate(raw[:, POINT]):
            truth = self.labels[int(pid)]
            if truth > 0:
                hits[truth].append(col)
        proposals = []
        for prior in priors:
            if not prior.confirmed or not all(owner in available for owner in prior.owners):
                continue
            truth = [self.labels[int(pid)] for pid in prior.observations[:, POINT]]
            if truth[0] == truth[1] or not all(label > 0 and len(hits[label]) == 1 for label in truth):
                continue
            indices = tuple(hits[label][0] for label in truth)
            if tuple(sorted(indices)) in existing or any(col in used_cols for col in indices):
                continue
            proposals.append((prior, indices))
        owner_use = Counter(owner for prior, _ in proposals for owner in prior.owners)
        point_use = Counter(col for _, indices in proposals for col in indices)
        tick_counts = Counter(stamps) if proposals else {}
        for prior, indices in proposals:
            owners = prior.owners
            if any(owner_use[owner] > 1 for owner in owners) or any(point_use[col] > 1 for col in indices):
                continue
            self.counts['unlisted_truth_pairs'] += 1
            delta = int(stamps[indices[1]] - stamps[indices[0]])
            self.counts[f'unlisted_signed_tick_delta_{max(-9, min(delta, 9))}'] += 1
            self.counts['unlisted_nonunique_tick'] += any(tick_counts[stamps[col]] > 1 for col in indices)
            a, b = indices
            before = prior.observations
            after = raw[[a, b]]
            offset_error = abs(after[1, AMPLITUDE] - after[0, AMPLITUDE]
                               - before[1, AMPLITUDE] + before[0, AMPLITUDE])
            self.counts['unlisted_stable_offset_tight'] += int(offset_error <= OFFSET_TOLERANCE)
            self.counts['unlisted_stable_offset_1e_5'] += int(offset_error <= 1e-5)
            self.counts['unlisted_stable_offset_1e_4'] += int(offset_error <= 1e-4)
            old_role = residual(before[1, POSITION], before[0, POSITION])
            role_change = residual(residual(after[1, POSITION], after[0, POSITION]), old_role)
            self.counts['unlisted_stable_relative_role'] += int((np.abs(role_change) <= ROLE_WIDTH).all())
            pair_tracks = [available[owner] for owner in owners]
            lags = np.array([circle - track.circle for track in pair_tracks])
            predicted = np.array([track.position + lag * track.velocity
                                  for track, lag in zip(pair_tracks, lags)])
            scale = 1 + .35 * np.maximum(0, lags - 1)
            error = residual(after[:, POSITION], predicted)
            self.counts['unlisted_plausible_center'] += int((np.abs(error.mean(axis=0))
                                                            <= CENTER_WIDTH * max(scale)).all())
            for owner, col in zip(owners, indices):
                self.unlisted_oracle[int(raw[col, POINT])] = int(owner)
        return frame


def replay(path, participant, variant):
    sys.path.insert(0, str(participant / 'src'))
    from algorithm import load_runtime

    runtime = load_runtime(participant / 'models' / 'frozen.json')
    tracker, history = runtime.new_packet()
    labels, circles, stamps, outputs = {}, {}, {}, {}
    trace = PairReplay(tracker.pairing, labels, variant)
    state = {'_precision_packet': (tracker, history)}
    with patch.object(tracker.pairing, 'begin', trace.begin):
        for _, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            source = {(p['Time'], p['CircleID'], p['PntID']): (p['Azimuth'], p['Elevation'], p['Distance'])
                      for p in points}
            stamps.update((pid, time) for time, _, pid in source)
            tracks = runtime.process_circle(points, state)
            encode_tracks(tracks, source)
            outputs.update((p['PntID'], p['BatchID']) for p in tracks)

    result = {'score': evaluate_packet(labels, outputs, circles, stamps), 'counts': trace.counts}
    if variant:
        trace.counts['extra_locked_emitted'] = sum(outputs.get(pid) == owner for pid, owner in trace.locked.items())
    else:
        trace.counts['unlisted_oracle_missing_points'] = sum(pid not in outputs for pid in trace.unlisted_oracle)
        trace.counts['unlisted_oracle_wrong_id_points'] = sum(
            pid in outputs and outputs[pid] != owner for pid, owner in trace.unlisted_oracle.items())
        result['oracle'] = evaluate_packet(labels, outputs | trace.unlisted_oracle, circles, stamps)
    print(path.name, 'variant' if variant else 'baseline', result['score']['score']['total'], dict(trace.counts),
          flush=True)
    return result


def replay_pair(path, participant, mode):
    result = {'packet': path.name, 'baseline': replay(path, participant, False)}
    if mode == 'compare':
        result['candidate'] = replay(path, participant, True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--submission-zip', type=Path,
                        default=WORK.parents[1] / 'radar_b02_official_1115_code.zip')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--mode', choices=('oracle', 'compare'), default='oracle')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    with TemporaryDirectory(prefix='b02-unlisted-pairs-') as scratch:
        participant = Path(scratch)
        with ZipFile(args.submission_zip) as package:
            package.extractall(participant, [name for name in package.namelist()
                                            if name.startswith(('src/', 'models/'))])
        if args.jobs == 1:
            outputs = [replay_pair(path, participant, args.mode) for path in paths]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as workers:
                outputs = list(workers.map(replay_pair, paths, [participant] * len(paths), [args.mode] * len(paths)))
    report = {
        'metadata': {'packet_ids': args.packets, 'score_version': SCORE_VERSION,
                     'submission_zip_sha256': file_sha256(args.submission_zip)},
        'baseline': aggregate_packets(x['baseline']['score'] for x in outputs),
        'retrospective_oracle': aggregate_packets(x['baseline']['oracle'] for x in outputs),
        'counts': sum((x['baseline']['counts'] for x in outputs), Counter()),
    }
    if args.mode == 'compare':
        report['candidate'] = aggregate_packets(x['candidate']['score'] for x in outputs)
        report['candidate_counts'] = sum((x['candidate']['counts'] for x in outputs), Counter())
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
