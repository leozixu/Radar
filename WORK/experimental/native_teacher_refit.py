"""Refit or compare the native association teacher on closed-loop development scans."""

import argparse
from collections import Counter
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
from packet_scope import TUNING_PACKETS


class ClosedLoopLabels:
    """Make supervised edge labels from past causal assignments only."""

    def __init__(self, compute_edges):
        self.compute_edges = compute_edges
        self.scan_labels = {}
        self.owners = {}
        self.features = []
        self.targets = []

    def observe_input(self, circle, points, truth):
        by_position = {}
        for point in points:
            position = point['Azimuth'], point['Elevation'], point['Distance']
            label = truth[point['PntID']]
            previous = by_position.setdefault(position, label)
            if previous != label:
                by_position[position] = -1
        self.scan_labels[circle] = by_position

    def intended_owner(self, track):
        if track.batch_id in self.owners:
            return self.owners[track.batch_id]
        votes = Counter(self.scan_labels[t].get(tuple(reading), -1) for t, _, reading in track.history)
        votes.pop(-1, None)
        if not votes:
            return -1
        label, count = votes.most_common(1)[0]
        if count >= 2 and count >= .85 * sum(votes.values()) and (label > 0 or count >= 3):
            self.owners[track.batch_id] = label
            return label
        return -1

    def labeled_edges(self, tracks, positions, amplitudes, circle):
        rows, cols, features = self.compute_edges(tracks, positions, amplitudes, circle)
        if not len(rows):
            return rows, cols, features
        source = self.scan_labels[circle]
        labels = np.fromiter((source.get(tuple(point), -1) for point in positions), np.int64)
        owners = np.fromiter((self.intended_owner(track) for track in tracks), np.int64)
        known = (owners[rows] >= 0) & (labels[cols] >= 0)
        target = (owners[rows] == labels[cols]) & (labels[cols] > 0)
        self.features.append(features[known])
        self.targets.append(target[known])
        return rows, cols, features


def collect(path, participant):
    sys.path.insert(0, str(participant / 'src'))
    from algorithm import load_runtime
    from precision_tracking import joint_elevation_pairs

    runtime = load_runtime(participant / 'models' / 'frozen.json')
    journal = ClosedLoopLabels(joint_elevation_pairs.candidate_features)
    labels, circles, state = {}, {}, {}
    with patch.object(joint_elevation_pairs, 'candidate_features', journal.labeled_edges):
        for circle, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            journal.observe_input(circle, points, labels)
            runtime.process_circle(points, state)
    x = np.concatenate(journal.features)
    y = np.concatenate(journal.targets)
    print(path.name, 'known edges', len(y), 'correct edges', int(y.sum()), flush=True)
    return x, y


def train_model(samples, participant, output, rounds):
    from lightgbm import Dataset, train

    old_teacher = participant / 'models' / 'precision' / 'teacher.txt'
    names = teacher_names(old_teacher)
    x, y = (np.concatenate(parts) for parts in zip(*samples))
    data = Dataset(x, label=y, feature_name=names)
    settings = {'objective': 'binary', 'learning_rate': .025, 'num_leaves': 39, 'min_data_in_leaf': 100,
                'num_threads': 2, 'force_col_wise': True, 'deterministic': True, 'verbosity': -1,
                'seed': 9711}
    trained = train(settings, data, init_model=str(old_teacher), num_boost_round=rounds)
    trained.save_model(str(output))
    return {'count': len(y), 'positives': int(y.sum()), 'settings': settings, 'additional_rounds': rounds,
            'training_edges': [{'edges': len(edges), 'positives': int(targets.sum())} for edges, targets in samples]}


def teacher_names(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('feature_names='):
                return line.strip().split('=', 1)[1].split()
    raise ValueError(f'{path} lacks a native teacher feature schema')


def replay(path, participant, teacher):
    sys.path.insert(0, str(participant / 'src'))
    from algorithm import load_runtime
    from lightgbm import Booster
    from precision_tracking.teacher_assoc import FEATURES

    runtime = load_runtime(participant / 'models' / 'frozen.json')
    tracker, history = runtime.new_packet()
    candidate = Booster(model_file=str(teacher))
    if tuple(candidate.feature_name()) != FEATURES:
        raise ValueError('Candidate does not match the native teacher feature schema')
    tracker.teacher_model = candidate
    labels, circles, times, outputs = {}, {}, {}, {}
    state = {'_precision_packet': (tracker, history)}
    for _, payload in iter_circle_payloads(path, labels, circles, allowed=TUNING_PACKETS):
        points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
        source = {(p['Time'], p['CircleID'], p['PntID']): (p['Azimuth'], p['Elevation'], p['Distance'])
                  for p in points}
        times.update((pid, stamp) for stamp, _, pid in source)
        tracks = runtime.process_circle(points, state)
        encode_tracks(tracks, source)
        outputs.update((p['PntID'], p['BatchID']) for p in tracks)
    stats = evaluate_packet(labels, outputs, circles, times)
    print(path.name, stats['score']['total'], stats['id_switches'], stats['split_truth_tracks'], flush=True)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'compare'))
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--submission-zip', type=Path,
                        default=WORK.parents[1] / 'radar_b02_official_1115_code.zip')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--teacher', type=Path, required=True, help='Input for compare, output for train')
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    training = args.phase == 'train'
    paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS if training else TUNING_PACKETS)
    with TemporaryDirectory(prefix='b02-native-teacher-') as scratch:
        participant = Path(scratch)
        with ZipFile(args.submission_zip) as package:
            package.extractall(participant, [name for name in package.namelist()
                                            if name.startswith(('src/', 'models/'))])
        worker = collect if training else replay
        common = (participant,) if training else (participant, args.teacher)
        if args.jobs == 1:
            values = [worker(path, *common) for path in paths]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as workers:
                values = list(workers.map(worker, paths, *([item] * len(paths) for item in common)))
        if training:
            result = train_model(values, participant, args.teacher, args.rounds)
        else:
            result = {'summary': aggregate_packets(values), 'packets': {
                path.name: {k: v for k, v in stats.items() if k != 'tracks'} for path, stats in zip(paths, values)}}
    report = {'metadata': {'packet_ids': args.packets, 'score_version': SCORE_VERSION,
                           'submission_zip_sha256': file_sha256(args.submission_zip),
                           'teacher_sha256': file_sha256(args.teacher)}, **result}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
