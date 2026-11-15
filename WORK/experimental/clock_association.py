"""Evaluate calibrated scan phase and causal motion history as association cues."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

import replay_validation as replay
from algorithm import load_runtime
from baseline_cv import FOLLOW_GATE, START_GATE, residual
import baseline_cv
import tracker_compat
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from train_tracker import read_packet


PERIOD_MS = 5000 / 3
FEATURES = ('phase_abs', 'alignment_fraction', 'amplitude')


def wrap(value):
    return (value + 0.5) % 1.0 - 0.5


def phase_features(azimuth, timestamps, amplitudes):
    """Center phase using the densest observed band in the current scan."""
    phase = wrap(azimuth - timestamps / PERIOD_MS)
    counts = np.bincount(np.rint(500 * (phase + 0.5)).astype(int) % 500, minlength=500)
    peak = np.argmax(counts + np.roll(counts, 1) + np.roll(counts, -1))
    centered = wrap(phase - (peak / 500 - 0.5))
    inliers = np.abs(centered) < 0.004
    share = inliers.mean()
    if share >= 0.3:
        error = np.abs(wrap(centered - np.median(centered[inliers])))
    else:
        error = np.full_like(phase, np.nan)
    return np.column_stack((error, np.full(len(phase), share), amplitudes))


def collect(path):
    points = read_packet(path)
    scans = np.split(points, np.flatnonzero(np.diff(points[:, 6])) + 1)
    xs, ys = [], []
    for scan in scans:
        valid = scan[:, 8] >= 0
        features = phase_features(scan[:, 1], scan[:, 0], scan[:, 4])
        xs.append(features[valid].astype(np.float32))
        ys.append((scan[valid, 8] > 0).astype(np.int8))
    return np.concatenate(xs), np.concatenate(ys)


def fit(args):
    paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    import lightgbm as lgb

    samples = [collect(path) for path in paths]
    x = np.concatenate([features for features, _ in samples])
    y = np.concatenate([labels for _, labels in samples])
    data = lgb.Dataset(x, label=y, feature_name=list(FEATURES))
    model = lgb.train(dict(objective='binary', num_leaves=15, min_data_in_leaf=160, learning_rate=.08,
                           num_threads=args.jobs, force_col_wise=True, deterministic=True, verbosity=-1),
                      data, num_boost_round=100)
    model.save_model(str(args.model))
    print(f'Clock model trained: {len(paths)} packets, {len(x):,} labeled points', flush=True)


def forecast(track, prior, circle):
    """Blend the Kalman forecast with fits over several causal observations."""
    history = prior.get(id(track))
    gap = circle - track.circle
    predicted = track.position + gap * track.velocity
    if history is None or len(history) < 5:
        return predicted
    times, readings = zip(*history)
    x = np.asarray(times) - circle
    pos = predicted + residual(np.asarray(readings), predicted)
    for axis, count in enumerate((5, 8, 5)):
        past = x[-count:]
        values = pos[-count:, axis]
        centered = past - past.mean()
        estimate = values.mean() - past.mean() * np.dot(centered, values) / np.dot(centered, centered)
        innovation = estimate - predicted[axis]
        # A fit based on older measurements cannot completely override a maneuver.
        predicted[axis] += .55 * np.clip(innovation, -FOLLOW_GATE[axis]/2, FOLLOW_GATE[axis]/2)
    return predicted


class Experiment:
    def __init__(self, model, variant, teacher=None, quality_threshold=None):
        self.quality = load_runtime()
        if quality_threshold is not None:
            self.quality = replace(self.quality, threshold=quality_threshold)
        self.model = model
        self.variant = variant
        self.teacher = teacher

    def process_circle(self, points, state):
        with ExitStack() as patches:
            if self.teacher is not None:
                patches.enter_context(patch.object(tracker_compat, 'baseline_circle', self.teacher.track))
            return self.correct_association(points, state)

    def correct_association(self, points, state):
        if not points:
            return self.quality.process_circle(points, state)
        if self.variant == 'reference':
            return self.quality.process_circle(points, state)
        circle = points[0]['CircleID']
        history = state.setdefault('clock_motion', {})
        motion = 'motion' in self.variant
        phase = self.variant.startswith('clock')
        probabilities = {}
        if phase:
            pos = np.asarray([(p['Azimuth'], p['Elevation'], p['Distance']) for p in points])
            clock = np.asarray([p['Time'] for p in points])
            amp = np.asarray([p['Amp'] for p in points])
            features = phase_features(pos[:, 0], clock, amp)
            if features[0, 1] >= .3:
                probabilities = dict(zip(map(tuple, pos), self.model.predict(features, num_threads=1)))

        def association(tracks, positions, amplitudes, circle, initial=False):
            if not tracks or not len(positions):
                return []
            elapsed = np.asarray([circle - t.circle for t in tracks])
            predictions = np.stack([forecast(t, history, circle) if motion else
                                    t.position + dt * t.velocity for t, dt in zip(tracks, elapsed)])
            widths = (START_GATE if initial else FOLLOW_GATE) * (1+.35*(elapsed-1))[:,None]
            innovation = residual(positions[None,:,:], predictions[:,None,:])
            distance = np.square(innovation / widths[:,None,:]).sum(axis=2)
            means = np.asarray([t.mean_amplitude for t in tracks])
            cost = (distance+.45*np.maximum(.35-amplitudes,0)/.35
                    +.22*np.abs(amplitudes[None,:]-means[:,None]))
            if probabilities:
                prob = np.fromiter((probabilities.get(tuple(p), .5) for p in positions), float)
                cost += .30*(1-prob)**3 - .12*prob
            cost[distance>1] = np.inf
            misses = np.full((len(tracks),len(tracks)),np.inf)
            np.fill_diagonal(misses,1.)
            rows, targets = linear_sum_assignment(np.column_stack((cost,misses)))
            return [(r,c) for r,c in zip(rows,targets) if c<len(positions)]

        def update(track, position, amplitude, circle):
            if motion:
                history.setdefault(id(track), deque(maxlen=8)).append((circle,position.copy()))
            raw_update(track, position, amplitude, circle)

        raw_update = baseline_cv.update_track
        with patch.object(baseline_cv, 'associate', association), patch.object(baseline_cv, 'update_track', update):
            result = self.quality.process_circle(points, state)
        if motion:
            state['clock_motion'] = {id(t): history[id(t)] for group in state['tracks'] for t in group if id(t) in history}
        return result


def compare_packet(spec):
    path, model_path, variants, teacher_path, quality_threshold = spec
    from lightgbm import Booster

    booster = Booster(model_file=str(model_path))
    teacher = None
    if teacher_path:
        from teacher_assoc import Online

        teacher = Online(Booster(model_file=str(teacher_path)), floor=.23, mix=.2)
    output = {}
    for variant in variants:
        track = Experiment(booster, variant, teacher, quality_threshold)
        with patch.object(replay, 'load_handler', return_value=track.process_circle):
            output[variant] = replay.replay_packet(path, allowed=TUNING_PACKETS)
        stats = output[variant]
        print(path.name, variant, round(stats['score']['total'], 4), stats['id_switches'], flush=True)
    return output


def evaluate(args):
    paths = packet_paths(args.data_dir, args.packets, TUNING_PACKETS)
    specs = [(path, args.model, args.variants, args.teacher_model, args.quality_threshold) for path in paths]
    if args.jobs == 1:
        reports = list(map(compare_packet, specs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            reports = list(pool.map(compare_packet, specs))
    result = {
        'packet_ids': args.packets,
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'teacher_model_sha256': (hashlib.sha256(args.teacher_model.read_bytes()).hexdigest()
                                 if args.teacher_model else None),
        'quality_threshold': args.quality_threshold,
        'variants': {},
    }
    for variant in args.variants:
        result['variants'][variant] = aggregate_packets(report[variant] for report in reports)
    args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train','evaluate'))
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--teacher-model', type=Path)
    parser.add_argument('--quality-threshold', type=float)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--variants', nargs='+', choices=('reference', 'clock', 'motion', 'clock-motion'),
                        default=('clock', 'motion', 'clock-motion'))
    args = parser.parse_args()
    if args.jobs < 1 or (args.phase == 'evaluate' and not args.report):
        parser.error('positive --jobs and --report for evaluation required')
    if args.quality_threshold is not None and not 0 <= args.quality_threshold <= 1:
        parser.error('--quality-threshold must be within [0, 1]')
    (fit if args.phase=='train' else evaluate)(args)


if __name__ == '__main__':
    main()
