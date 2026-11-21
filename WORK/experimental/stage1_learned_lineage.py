"""Train and replay a causal learned bridge at the first public output of a raw ID."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, encode_tracks, file_sha256, iter_circle_payloads


EXTRA = ('track_public_gap', 'raw_history_gap', 'new_path_span', 'history_overlap', 'old_pub_hit_count') + tuple(
    f'{kind}_{axis}' for kind in ('endpoint_jump', 'new_path_mean_error', 'new_path_peak_error', 'slope_disagreement')
    for axis in ('az', 'el', 'range')
)
SCALE = np.array((.020, .025, .003))
FLOORS = (.55, .68, .80, .90)


def create_guard(source_model=None, floor=.90):
    import active_lineage_rejoin as lineage
    import identity_rejoin

    class LearnedLineage(lineage.CorroboratedActiveLineage):
        minimum_history = 3
        maximum_new_hits = 9

        def candidates(self, points, proposals, rows, qualities, raw_state):
            # The archived CV gate missed most known handoffs. Keep a wide
            # geometrical enumeration, then learn the final bridge confidence.
            with (patch.object(identity_rejoin, 'SCALE', np.array((.080, .036, .008))),
                  patch.object(lineage, 'MIN_OUTPUT_GAP', 1)):
                base = super().candidates(points, proposals, rows, qualities, raw_state)
            if base is None:
                return None
            old, births, ii, jj, features = base
            active = {track.batch_id: track for track in raw_state['tracks'][0]}
            scan = raw_state['circle']
            evidence = []
            for i, j in zip(ii, jj):
                previous = self.state.history[self.state.public.get(old[i], old[i])]
                history = previous.trajectory()
                incoming = active[rows[births[j]]['BatchID']]
                samples = list(incoming.history)
                observations = np.asarray([pos for _, _, pos in samples])
                sample_scans = np.asarray([number for number, _, _ in samples])
                last_scan = history.scans[-1]
                predict = history.short_position + (sample_scans - last_scan)[:, None] * history.short_velocity
                errors = np.abs(identity_rejoin.residual(observations, predict)) / SCALE
                jump = np.abs(identity_rejoin.residual(observations[0], history.positions[-1])) / SCALE
                slope = np.abs(incoming.velocity - history.short_velocity) / SCALE
                evidence.append((scan - last_scan, sample_scans[0] - active[old[i]].circle,
                                 sample_scans[-1] - sample_scans[0], int((sample_scans <= last_scan).sum()),
                                 previous.hits, *jump, *errors.mean(axis=0), *errors.max(axis=0), *slope))
            return old, births, ii, jj, np.column_stack((features, evidence)).astype(np.float32)

        def probabilities(self, features):
            return self.model.predict(features, num_threads=1)

    return LearnedLineage(source_model, floor), lineage.PHYSICAL_FEATURES + EXTRA


def owner_vote(votes):
    if not votes:
        return None
    owner, count = votes.most_common(1)[0]
    return owner if count >= 3 and count >= .85 * votes.total() else None


def run_packet(task):
    path, src, profile, model_path, allowed = task
    sys.path.insert(0, str(src))
    from algorithm import load_runtime
    from frozen_rejoin_compare import CapturedRuntime
    from lightgbm import Booster

    runtime = load_runtime(profile)
    runtime_source = Path(sys.modules[type(runtime).__module__].__file__).resolve()
    if not runtime_source.is_relative_to(src):
        raise RuntimeError(f'Unexpected tracker source {runtime_source}')
    trace = CapturedRuntime(runtime)
    collect = model_path is None
    if collect:
        guard, feature_names = create_guard()
        floor_variants = {}
    else:
        model = Booster(model_file=str(model_path))
        floor_variants = {str(floor): create_guard(model, floor)[0] for floor in FLOORS}
    outputs = {name: {} for name in ('baseline', *floor_variants)}
    votes = {name: defaultdict(Counter) for name in outputs}
    labels, circles, times = {}, {}, {}
    fit_rows, fit_labels = [], []
    with patch.object(trace.tracker, 'track', trace.track), patch.object(trace.quality, 'predict', trace.predict):
        for _, payload in iter_circle_payloads(path, labels, circles, allowed=allowed):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            source = {(p['Time'], p['CircleID'], p['PntID']): (p['Azimuth'], p['Elevation'], p['Distance'])
                      for p in points}
            times.update((pid, stamp) for stamp, _, pid in source)
            public = runtime.process_circle(points, trace.state)
            encode_tracks(public, source)
            quality = trace.public_qualities(public)
            if collect:
                choices = guard.candidates(points, trace.proposals, public, quality, trace.state)
                if choices is not None:
                    old, births, ii, jj, features = choices
                    for i, j, features_row in zip(ii, jj, features):
                        source_owner = owner_vote(votes['baseline'][old[i]])
                        incoming = labels[public[births[j]]['PntID']]
                        if source_owner is not None and incoming >= 0:
                            fit_rows.append(features_row)
                            fit_labels.append(source_owner == incoming)
                guard.remember(points, public, quality, trace.state)
            variants = {'baseline': public}
            for name, variant in floor_variants.items():
                corrected = variant.correct(points, trace.proposals, public, quality, trace.state)
                if variant.last_links:
                    if len({row['BatchID'] for row in corrected}) != len(corrected):
                        raise ValueError('A bridge assigned the same public ID to two points in a scan')
                    encode_tracks(corrected, source)
                for previous, pid in variant.last_links:
                    owner = owner_vote(votes[name][previous])
                    incoming = labels[pid]
                    category = 'unscored' if owner is None or incoming < 0 else (
                        'correct' if owner == incoming else 'wrong'
                    )
                    variant.evidence[f'{category}_alias'] += 1
                variants[name] = corrected
            for name, result in variants.items():
                for row in result:
                    pid, batch = row['PntID'], row['BatchID']
                    outputs[name][pid] = batch
                    if labels[pid] > 0:
                        votes[name][batch][labels[pid]] += 1

    if collect:
        print(path.name, 'train candidates', len(fit_rows), 'positives', sum(fit_labels), flush=True)
        return np.asarray(fit_rows, dtype=np.float32).reshape(-1, len(feature_names)), np.asarray(fit_labels), feature_names
    if any(seen.keys() != outputs['baseline'].keys() for seen in outputs.values()):
        raise ValueError('An identity-only experiment modified which points were returned')
    metrics = {name: evaluate_packet(labels, result, circles, times) for name, result in outputs.items()}
    evidence = {name: variant.evidence for name, variant in floor_variants.items()}
    print(path.name, {name: round(values['score']['total'], 6) for name, values in metrics.items()}, evidence, flush=True)
    return metrics, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--train', type=int, nargs='+', required=True)
    parser.add_argument('--tune', type=int, nargs='+', required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    src = args.src.resolve()
    train_paths = packet_paths(args.data_dir, args.train, TRAIN_PACKETS)
    tune_paths = packet_paths(args.data_dir, args.tune, TUNING_PACKETS)
    metadata = {
        'train_packets': args.train, 'tune_packets': args.tune, 'score_version': SCORE_VERSION,
        'profile_sha256': file_sha256(args.profile),
        'source_sha256': {path.relative_to(src).as_posix(): file_sha256(path) for path in sorted(src.rglob('*.py'))},
    }
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        fit = [(path, src, args.profile, None, TRAIN_PACKETS) for path in train_paths]
        training = list(pool.map(run_packet, fit))
        samples = np.concatenate([sample for sample, _, _ in training])
        classes = np.concatenate([label for _, label, _ in training]).astype(np.int32)
        report = {'metadata': metadata, 'fit': {'candidates': len(classes), 'positives': int(classes.sum())}}
        if classes.sum() < 12 or (classes == 0).sum() < 12:
            args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(f'Too few labelled associations for stable training: {report["fit"]}', flush=True)
            return

        from lightgbm import Dataset, train

        fit_params = {'objective': 'binary', 'learning_rate': .045, 'num_leaves': 7, 'max_depth': 3,
                      'min_data_in_leaf': 8, 'lambda_l2': 2., 'verbosity': -1, 'deterministic': True,
                      'force_col_wise': True, 'num_threads': 1, 'seed': 6303}
        model = train(fit_params, Dataset(samples, label=classes, feature_name=list(training[0][2])),
                      num_boost_round=80)
        model.save_model(str(args.model))
        report['fit'].update(parameters=fit_params, model_sha256=file_sha256(args.model))
        validation = [(path, src, args.profile, args.model, TUNING_PACKETS) for path in tune_paths]
        outcomes = list(pool.map(run_packet, validation))
    report['summary'] = {name: aggregate_packets(metrics[name] for metrics, _ in outcomes)
                         for name in outcomes[0][0]}
    report['aliases'] = {name: sum((evidence[name] for _, evidence in outcomes), Counter())
                         for name in outcomes[0][1]}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print({name: values['score']['total'] for name, values in report['summary'].items()}, flush=True)


if __name__ == '__main__':
    main()
