"""Evaluate which previously paired radar arm produced a lone observation."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from paired_returns import AMPLITUDE_TOLERANCE, measured_pairs
from clock_association import PERIOD_MS, wrap
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths, validate_packet_path
from replay_validation import file_sha256
from train_tracker import read_packet


HORIZONS = (4, 9, 15)
AXES = ('azimuth', 'elevation', 'range', 'amplitude', 'clock_azimuth')
NOISE = np.array([.0020, .0090, .00055, .045, .0030])
NEAREST_NOISE = np.array([.0030, .018, .00115])
FEATURES = ('gap', 'history', 'ticks_previous', 'ticks_next', 'same_tick_count', 'adjacent_previous',
            'adjacent_next') + tuple(
    f'{horizon}_{kind}_{axis}' for horizon in HORIZONS
    for kind in ('residual', 'delta', 'spread', 'aligned', 'forecast_step') for axis in AXES
) + tuple(f'{horizon}_{kind}' for horizon in HORIZONS
          for kind in ('odds_diagonal', 'odds_joint', 'phase_relative', 'phase_delta'))


def observed_position(position, amplitude, stamps):
    return np.column_stack((position, amplitude, wrap(stamps / PERIOD_MS)))


def relative(readings, center):
    delta = readings - center
    delta[..., (0, 4)] = wrap(delta[..., (0, 4)])
    return delta


def local_timing(stamps, sorted_stamps, columns, simulate):
    result = []
    for index, column in enumerate(columns):
        stamp = stamps[column]
        left, right = np.searchsorted(sorted_stamps, stamp, side='left'), np.searchsorted(sorted_stamps, stamp, side='right')
        if simulate:
            hidden = stamps[columns[1-index]]
            if left > 0 and sorted_stamps[left-1] == hidden:
                left -= 1
            if right < len(sorted_stamps) and sorted_stamps[right] == hidden:
                right += 1
        previous = min(64, stamp-sorted_stamps[left-1]) if left else 64
        following = min(64, sorted_stamps[right]-stamp) if right < len(sorted_stamps) else 64
        copies = np.searchsorted(sorted_stamps, stamp, side='right') - np.searchsorted(sorted_stamps, stamp, side='left')
        result.append((previous, following, copies, int(previous == 1), int(following == 1)))
    return np.asarray(result)


def predicted_arms(history, observations, circle, local):
    """Fit the shared motion and relative arm offsets using only confirmed prior scans."""
    t = np.asarray([time for time, _ in history], dtype=float)
    before = np.stack([readings for _, readings in history])
    anchor = before[-1, 0]
    values = relative(before, anchor)
    centers = values.mean(axis=1)
    arm_delta = values[:, 1] - values[:, 0]
    observed = relative(observations, anchor)
    gap = circle - t[-1]
    output = [np.tile([gap, len(t)], (len(observed), 1)), local]
    extras = []
    basic = None
    for horizon in HORIZONS:
        time = t[-horizon:]
        elapsed = circle - time.mean()
        shift = time - time.mean()
        base = centers[-horizon:]
        offset = arm_delta[-horizon:]
        denominator = np.dot(shift, shift)
        velocity = shift @ base / denominator
        predicted = base.mean(axis=0) + elapsed * velocity
        expected_delta = np.median(offset, axis=0)
        error = observed - predicted
        adjusted = base - base.mean(axis=0) - shift[:, None] * velocity
        covariance = adjusted.T @ adjusted / max(1, len(time) - 2)
        covariance *= 1. + 1. / len(time) + elapsed**2 / denominator
        covariance += np.diag(NOISE**2)
        scale = np.sqrt(np.diag(covariance))
        normalized = error / scale
        delta = expected_delta / scale
        # Positive means later radar tick. Covariance helps distinguish stable
        # offset direction from measurement jitter shared by both radar arms.
        diagonal = normalized @ delta
        joint = error @ np.linalg.solve(covariance, expected_delta)
        phase = error[:, 0] - error[:, 4]
        phase_scale = np.sqrt(covariance[0, 0] + covariance[4, 4] - 2. * covariance[0, 4])
        output.extend((normalized, np.broadcast_to(delta, error.shape),
                       np.broadcast_to(scale / NOISE, error.shape), normalized * delta,
                       np.broadcast_to(gap * velocity / NOISE, error.shape)))
        extras.append(np.column_stack((diagonal, joint, phase / phase_scale,
                                       np.full(len(error), (expected_delta[0] - expected_delta[4]) / phase_scale))))
        if horizon == 9:
            basic = (error[:, :3] / NEAREST_NOISE) @ (expected_delta[:3] / NEAREST_NOISE)
    return np.column_stack((*output, *extras)).astype(np.float32), basic


def collect(spec):
    path, sim_stride = spec
    path = validate_packet_path(path, TRAIN_PACKETS)
    data = read_packet(path)
    population = defaultdict(list)
    groups = {}
    counts = Counter()
    for scan in np.split(data, np.flatnonzero(np.diff(data[:, 6])) + 1):
        circle = int(scan[0, 6])
        position = scan[:, 1:4].astype('f').astype(float)
        amplitude = scan[:, 4].astype('f').astype(float)
        stamps, labels = scan[:, (0, 8)].astype(np.int64).T
        sorted_stamps = np.sort(stamps)
        # Positive ground truth validates the historical group. A constant
        # quality input avoids using a detection model fitted on held packets.
        raw = np.column_stack((position, amplitude, np.ones(len(scan))))
        observed = observed_position(position, amplitude, stamps)
        pairs = measured_pairs(raw, stamps)
        measured = defaultdict(list)
        pair_by_point = defaultdict(list)
        for a, b in pairs:
            pair_by_point[a].append(amplitude[b] - amplitude[a])
            pair_by_point[b].append(amplitude[b] - amplitude[a])
            if labels[a] > 0 and labels[b] > 0 and labels[a] != labels[b]:
                measured[(labels[a], labels[b])].append((a, b))
        unique, repeated = {}, set()
        for column in np.flatnonzero(labels > 0):
            label = int(labels[column])
            if label in unique:
                repeated.add(label)
            unique[label] = column
        for label in repeated:
            del unique[label]
        groups = {ids: history for ids, history in groups.items() if circle-history[-1][0] <= 12}
        for ids, history in groups.items():
            if len(history) < 4 or circle-history[-1][0] > 5 or any(label in repeated for label in ids):
                continue
            current = [unique.get(key) for key in ids]
            available = [arm for arm, index in enumerate(current) if index is not None]
            if not available:
                continue
            if len(available) == 1:
                arm = available[0]
                offset = history[-1][1][1, 3] - history[-1][1][0, 3]
                alternatives = pair_by_point.get(current[arm], ())
                if any(abs(value-offset) <= AMPLITUDE_TOLERANCE for value in alternatives):
                    counts['single_with_measured_offset'] += 1
                    continue
                mode = 'actual_missing'
            elif circle % sim_stride == 0 and len(measured.get(ids, ())) == 1:
                mode = 'simulated_missing'
            else:
                continue
            columns = [current[arm] for arm in available]
            local = local_timing(stamps, sorted_stamps, columns, mode == 'simulated_missing')
            feature, baseline = predicted_arms(history, observed[columns], circle, local)
            population[mode].append((feature, np.asarray(available), baseline))
        for ids, candidates in measured.items():
            if len(candidates) != 1 or any(label in repeated for label in ids):
                continue
            a, b = candidates[0]
            if ids not in groups:
                groups[ids] = deque(maxlen=max(HORIZONS))
            history = groups[ids]
            if history:
                gap = circle-history[-1][0]
                previous = history[-1][1]
                former = previous[1, 3] - previous[0, 3]
                current = amplitude[b] - amplitude[a]
                translation = np.abs(relative(observed[[a, b]], previous).mean(axis=0))
                if abs(former-current) > AMPLITUDE_TOLERANCE or (translation[:3] > [.07*gap, .15*gap, .035*gap]).any():
                    history.clear()
            history.append((circle, observed[[a, b]]))
    observations = {}
    for mode, batches in population.items():
        features, labels, geometry = zip(*batches)
        observations[mode] = (np.concatenate(features), np.concatenate(labels), np.concatenate(geometry))
    print(path.stem, {key: len(item[1]) for key, item in observations.items()}, dict(counts), flush=True)
    return observations, counts


def fit(samples, scope):
    from lightgbm import Dataset, train

    records = []
    for source, _ in samples:
        for mode, (x, y, _) in source.items():
            if scope != 'combined' and mode != scope:
                continue
            weight = 1. if mode == 'actual_missing' or scope != 'combined' else .25
            records.append((x, y, np.full(len(y), weight)))
    x, y, weight = (np.concatenate(sets) for sets in zip(*records))
    model = train(dict(objective='binary', learning_rate=.06, num_leaves=15, min_data_in_leaf=80,
                       verbosity=-1, deterministic=True, force_col_wise=True, num_threads=1, seed=92),
                  Dataset(x, label=y, weight=weight, feature_name=list(FEATURES)), num_boost_round=125)
    return model, {'samples': len(y), 'roles': np.bincount(y, minlength=2).tolist()}


def statistics(probability, truth):
    correct = (probability >= .5) == truth
    confidence = np.maximum(probability, 1. - probability)
    outcomes = {}
    for threshold in (.50, .65, .75, .85, .90, .95, .97, .99):
        selected = confidence >= threshold
        n = int(selected.sum())
        outcomes[f'{threshold:.2f}'] = {'support': n, 'coverage': n / max(1, len(truth)),
                                         'precision': float(correct[selected].mean()) if n else None}
    return outcomes


def evaluate(held, models):
    modes = ('actual_missing', 'simulated_missing')
    summaries = {}
    for mode in modes:
        batches = [source[mode] for source, _ in held if mode in source]
        features, truth, baseline = (np.concatenate(items) for items in zip(*batches))
        prediction = {key: model.predict(features, num_threads=1) for key, model in models.items()}
        # Use the fixed Gaussian scoring profile only to sweep a comparable
        # coverage curve; it is not fitted on the held train packets.
        prediction['nearest_center'] = 1. / (1. + np.exp(-np.clip(baseline, -50., 50.)))
        prediction['centroid_covariance'] = 1. / (1. + np.exp(-np.clip(features[:, -3], -50., 50.)))
        summary = {key: statistics(value, truth) for key, value in prediction.items()}
        summary['samples'] = len(truth)
        summary['roles'] = np.bincount(truth, minlength=2).tolist()
        summary['majority_accuracy'] = max(summary['roles']) / len(truth)
        summary['local_timing'] = {}
        for earlier in (False, True):
            for later in (False, True):
                selected = (features[:, 5] == earlier) & (features[:, 6] == later)
                summary['local_timing'][f'earlier_{int(earlier)}_later_{int(later)}'] = np.bincount(
                    truth[selected], minlength=2).tolist()
        summary['gaps'] = {}
        for gap in (1, 2, 3, 4, 5):
            mask = features[:, 0] == gap
            summary['gaps'][str(gap)] = {key: statistics(value[mask], truth[mask])
                                         for key, value in prediction.items()}
        summaries[mode] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', nargs='+', type=int, required=True)
    parser.add_argument('--held', nargs='+', type=int, required=True)
    parser.add_argument('--simulation-stride', type=int, default=7)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if set(args.fit) & set(args.held) or args.simulation_stride < 1:
        parser.error('fit and held packets must be disjoint, and simulation stride must be positive')
    fit_paths = packet_paths(DEFAULT_DATA_DIR, args.fit, TRAIN_PACKETS)
    held_paths = packet_paths(DEFAULT_DATA_DIR, args.held, TRAIN_PACKETS)
    tasks = [(path, args.simulation_stride) for path in (*fit_paths, *held_paths)]
    if args.jobs == 1:
        input_samples = list(map(collect, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            input_samples = list(pool.map(collect, tasks))
    training, held = input_samples[:len(fit_paths)], input_samples[len(fit_paths):]
    models = {}
    details = {}
    for scope in ('actual_missing', 'simulated_missing', 'combined'):
        models[scope], details[scope] = fit(training, scope)
    result = {
        'metadata': {'fit': args.fit, 'held': args.held, 'source_sha256': file_sha256(Path(__file__)),
                     'paired_returns_source_sha256': file_sha256(Path(__file__).with_name('paired_returns.py')),
                     'simulation_stride': args.simulation_stride,
                     'causal_history': f'{max(HORIZONS)} prior exact paired measurements, maximum gap 5 scans',
                     'limit': 'Conditional orientation only. True pair ownership and present labels identify examples.'},
        'fit': details,
        'held': evaluate(held, models),
        'held_by_packet': {path.stem: {key: len(value[1]) for key, value in source.items()}
                           for path, (source, _) in zip(held_paths, held)},
    }
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for mode, summary in result['held'].items():
        print(mode, summary['samples'], summary['roles'], {key: summary[key]['0.50'] for key in models}, flush=True)


if __name__ == '__main__':
    main()
