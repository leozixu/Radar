"""Train a causal rejection scorer on actual native emissions, then sweep without retracking."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from frozen_miss_audit import (
    DEFAULT_DATA_DIR, POINT, POINT_KEYS, SCORE_VERSION, TRAIN_PACKETS, WORK,
    aggregate_packets, encode_tracks, evaluate_packet, file_sha256, iter_circle_payloads, packet_paths,
)
from packet_scope import TUNING_PACKETS


CUTOFFS = (.70, .80, .90, .95, .975, .990, .995, .999)
EXTRA_NAMES = ('raw_probability', 'native_detection', 'native_ownership', 'confirmed_pair',
               'residual_link', 'native_public_count', 'native_public_age', 'previous_public_gap',
               'forecast_error_azimuth', 'forecast_error_elevation', 'forecast_error_range')


class OutputFeatures:
    def __init__(self, original_build, runtime):
        self.original_build = original_build
        self.runtime = runtime
        self.native_history = {}

    def record_proposals(self, history, points, proposals, quality, probabilities, state, clock):
        value = self.original_build(history, points, proposals, quality, probabilities, state, clock)
        self.proposals = proposals
        self.features = value
        return value

    def for_outputs(self, points, outputs, state):
        from baseline_cv import FOLLOW_GATE, residual
        from precision_tracking.causal_birth_paths import POINT, SCAN
        from precision_tracking import raw_timing_detection

        if not outputs:
            return None
        circle = points[0]['CircleID']
        self.native_history = {key: item for key, item in self.native_history.items() if circle - item[-1][0] <= 12}
        source_by_id = {point['PntID']: i for i, point in enumerate(points)}
        proposal_by_id = {proposal['PntID']: i for i, proposal in enumerate(self.proposals)}
        output_source = [source_by_id[track['PntID']] for track in outputs]
        base = self.features[[proposal_by_id[track['PntID']] for track in outputs]]
        position = np.array([(p['Azimuth'], p['Elevation'], p['Distance']) for p in points])
        amplitude = np.fromiter((p['Amp'] for p in points), dtype=float)
        time = np.fromiter((p['Time'] for p in points), dtype=np.int64)
        raw = raw_timing_detection.features(position, amplitude, time)[output_source]
        quality = state['raw_return_probability'][output_source]
        pair_members = {int(pid) for prior in state.get('previous_pairs', ())
                        if prior.confirmed and prior.observations[0, SCAN] == circle
                        for pid in prior.observations[:, POINT]}
        rescued = state.get('residual_matches', ())
        detection = self.runtime.models['emission_detection'].predict(base, num_threads=1)
        ownership = self.runtime.models['emission_ownership'].predict(base, num_threads=1)
        additions = []
        for i, (track, coordinate) in enumerate(zip(outputs, position[output_source])):
            pid, batch = track['PntID'], track['BatchID']
            past = self.native_history.get(batch, [])
            gap = circle - past[-1][0] if past else np.nan
            age = circle - past[0][0] if past else 0
            error = np.full(3, np.nan)
            if len(past) >= 2 and past[-1][0] > past[-2][0]:
                velocity = residual(past[-1][1], past[-2][1]) / (past[-1][0] - past[-2][0])
                error = residual(coordinate, past[-1][1] + velocity * gap) / FOLLOW_GATE
            additions.append((quality[i], detection[i], ownership[i], pid in pair_members, pid in rescued,
                              len(past), age, gap, *error))
            self.native_history[batch] = [*past[-9:], (circle, coordinate)]
        return np.column_stack((base, raw, additions)).astype(np.float32)


def collect(path, profile, source, training):
    sys.path.insert(0, str(source))
    from algorithm import load_runtime
    from precision_tracking import emission_identity, raw_timing_detection

    runtime = load_runtime(profile)
    trace = OutputFeatures(emission_identity.EmissionFeatures.build, runtime)
    labels, circles, times, outputs, state = {}, {}, {}, {}, {}
    rows, predicted_points = [], []

    def observe(history, points, proposals, quality, probabilities, state, clock):
        return trace.record_proposals(history, points, proposals, quality, probabilities, state, clock)

    allowed = TRAIN_PACKETS if training else TUNING_PACKETS
    with patch.object(emission_identity.EmissionFeatures, 'build', observe):
        for _, payload in iter_circle_payloads(path, labels, circles, allowed=allowed):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            input_scan = {(p['Time'], p['CircleID'], p['PntID']): (p['Azimuth'], p['Elevation'], p['Distance'])
                          for p in points}
            times.update((pid, stamp) for stamp, _, pid in input_scan)
            public = runtime.process_circle(points, state)
            encode_tracks(public, input_scan)
            value = trace.for_outputs(points, public, state)
            if value is None:
                continue
            rows.append(value)
            for track in public:
                pid = track['PntID']
                predicted_points.append(pid)
                outputs[pid] = track['BatchID']
    feature_names = (*emission_identity.FEATURES, *(f'raw_{name}' for name in raw_timing_detection.FEATURES),
                     *EXTRA_NAMES)
    x = np.concatenate(rows)
    truth_classes = np.fromiter((labels[pid] for pid in predicted_points), dtype=np.int64)
    unknown, clutter = int((truth_classes < 0).sum()), int((truth_classes == 0).sum())
    print(path.name, 'emitted', len(x), 'unknown', unknown, 'clutter', clutter, flush=True)
    if training:
        return x, truth_classes, feature_names
    return x, truth_classes, predicted_points, labels, circles, times, outputs


def train(samples, path):
    from lightgbm import Dataset, train as fit

    x = np.concatenate([item[0] for item in samples])
    labels = np.concatenate([item[1] for item in samples])
    settings = {'objective': 'binary', 'learning_rate': .045, 'num_leaves': 23,
                'min_data_in_leaf': 75, 'force_col_wise': True, 'deterministic': True,
                'num_threads': 2, 'verbosity': -1, 'seed': 804}
    model = fit(settings, Dataset(x, label=labels <= 0, feature_name=list(samples[0][2])), num_boost_round=140)
    model.save_model(str(path))
    counts = {'emitted': len(labels), 'unknown': int((labels < 0).sum()), 'clutter': int((labels == 0).sum())}
    return model, {'counts': counts, 'settings': settings, 'trees': model.current_iteration()}


def score_sweep(samples, model):
    thresholds = ('baseline', 'oracle', *CUTOFFS)
    evaluations = {key: [] for key in thresholds}
    removals = {key: Counter() for key in thresholds}
    for x, classes, pids, labels, circles, times, outputs in samples:
        probability = model.predict(x, num_threads=1)
        discard = {'baseline': np.zeros(len(classes), dtype=bool), 'oracle': classes <= 0}
        discard.update({cutoff: probability >= cutoff for cutoff in CUTOFFS})
        for key in thresholds:
            selected = discard[key]
            removed = {pid for pid, drop in zip(pids, selected) if drop}
            current = {pid: owner for pid, owner in outputs.items() if pid not in removed}
            counts = {'truth': int(np.count_nonzero(selected & (classes > 0))),
                      'unknown': int(np.count_nonzero(selected & (classes < 0))),
                      'clutter': int(np.count_nonzero(selected & (classes == 0)))}
            removals[key].update(counts)
            evaluations[key].append(evaluate_packet(labels, current, circles, times))
    return {str(key): {'summary': aggregate_packets(evaluations[key]), 'removed': removals[key],
                       'packet_scores': [item['score']['total'] for item in evaluations[key]]}
            for key in thresholds}


def run_workers(paths, profile, source, training, jobs):
    if jobs == 1:
        return [collect(path, profile, source, training) for path in paths]
    with ProcessPoolExecutor(max_workers=jobs) as workers:
        return list(workers.map(collect, paths, [profile] * len(paths), [source] * len(paths), [training] * len(paths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, nargs='+', required=True)
    parser.add_argument('--tune', type=int, nargs='+', required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    train_paths = packet_paths(args.data_dir, args.train, TRAIN_PACKETS)
    tune_paths = packet_paths(args.data_dir, args.tune, TUNING_PACKETS)
    with TemporaryDirectory(prefix='b02-output-recheck-') as scratch:
        source = Path(scratch) / 'src'
        shutil.copytree(WORK / 'participant_demo' / 'src', source, ignore=shutil.ignore_patterns('__pycache__'))
        source_hashes = {p.relative_to(source).as_posix(): file_sha256(p) for p in sorted(source.rglob('*.py'))}
        fit_data = run_workers(train_paths, args.profile, source, True, args.jobs)
        model, training = train(fit_data, args.model)
        del fit_data
        tune_data = run_workers(tune_paths, args.profile, source, False, args.jobs)
    scores = score_sweep(tune_data, model)
    report = {'metadata': {'train_packets': args.train, 'tune_packets': args.tune,
                           'profile_sha256': file_sha256(args.profile), 'runtime_source_sha256': source_hashes,
                           'model_sha256': file_sha256(args.model), 'score_version': SCORE_VERSION},
              'training': training, 'replay': scores}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for key, item in scores.items():
        print(key, item['summary']['score']['total'], dict(item['removed']), flush=True)


if __name__ == '__main__':
    main()
