"""Test early single-track births using two independently plausible raw returns."""

import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paired_single_start import PairFrameCapture, mature_competitor, reciprocal_seeds
from paired_returns import ASSETS, HERE
from causal_birth_paths import POINT, POSITION, BirthOnline, parallel
from algorithm import load_runtime
from baseline_cv import residual
from metrics import SCORE_VERSION, aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from raw_timing_detection import RawReturnGate
import replay_validation as replay
import tracker_compat


class RawGateEvidence:
    """Retain the probabilities from the same invocation as the ordinary raw gate."""

    def __init__(self, model, floor):
        self.model = model
        self.gate = RawReturnGate(self, floor)
        self.probability = None

    def predict(self, sample, num_threads=1):
        self.probability = self.model.predict(sample, num_threads=num_threads)
        return self.probability

    def __call__(self, positions, amplitude, times):
        self.probability = None
        return self.gate(positions, amplitude, times)


class RawAwareBirths(BirthOnline):
    def __init__(self, *upstream, pairing, raw_gate, seed_model, raw_support, seed_floor, margin,
                 association_floor=.20):
        super().__init__(*upstream, .86, "supplement", paired_birth=True, joint=True, two_hit_pairs=True,
                         pairing=pairing, raw_gate=raw_gate, association_floor=association_floor)
        self.seed_model = seed_model
        self.raw_support = raw_support
        self.seed_floor = seed_floor
        self.margin = margin
        self.single_stats = Counter()

    def track(self, points, state, observe=None):
        outputs = super().track(points, state, observe)
        probability = self.raw_gate.probability
        if probability is None:
            state.pop('early_single_raw_evidence', None)
        else:
            supported = self.pairing.raw[probability >= self.raw_support, POINT]
            state['early_single_raw_evidence'] = state['circle'], supported.copy()
        return outputs

    def promote_legacy(self, points, state, active, position, amp, circle, remaining, output, next_id, observe,
                       reserved_keys):
        initially_free = remaining.copy()
        tentative, pending, taken, next_id = super().promote_legacy(
            points, state, active, position, amp, circle, remaining, output, next_id, observe, reserved_keys)
        probability = self.raw_gate.probability
        evidence = state.get('early_single_raw_evidence')
        if (len(points) > 1200 or probability is None or evidence is None or evidence[0] != circle-1
                or not initially_free.any()):
            return tentative, pending, taken, next_id
        raw, frame = self.pairing.raw, self.pairing.frame
        initially_free[frame.columns[:frame.strict_count].ravel()] = False
        claimed = {*reserved_keys, *taken, *output}
        supported = (probability >= self.raw_support) & ~np.isin(raw[:, POINT], tuple(claimed))
        cols = np.flatnonzero(initially_free & supported)
        previous = next((rows for scan, rows in state['birth_path_pool'].observations if scan == circle-1), ())
        if not len(cols) or not len(previous):
            return tentative, pending, taken, next_id
        available = np.isin(previous[:, POINT], evidence[1]) & ~np.isin(previous[:, POINT], tuple(claimed))
        previous = previous[available]
        if not len(previous):
            return tentative, pending, taken, next_id
        current = raw[cols]
        matches = list(reciprocal_seeds(previous, current, self.seed_model, self.seed_floor, self.margin))
        self.single_stats['reciprocal'] += len(matches)
        if not matches:
            return tentative, pending, taken, next_id
        rows, candidates = np.array(matches).T
        first, second = previous[rows], current[candidates]
        displacement = residual(second[:, POSITION], first[:, POSITION])
        too_far = np.square(displacement / [.021, .098, .0078]).sum(axis=1) > 1.
        blocked = mature_competitor(active, first, circle-1) | mature_competitor(active, second, circle)
        self.single_stats['motion_veto'] += int(too_far.sum())
        self.single_stats['mature_veto'] += int((blocked & ~too_far).sum())
        for r in np.flatnonzero(~(blocked | too_far)):
            start, end = first[r], second[r]
            self.confirm((start, end), active, output, next_id, observe)
            remaining[cols[candidates[r]]] = False
            taken.update((start[POINT], end[POINT]))
            self.single_stats['confirmed'] += 1
            next_id += 1
        tentative = [track for track in tentative if not taken.intersection(pending[id(track)])]
        pending = {id(track): pending[id(track)] for track in tentative}
        return tentative, pending, taken, next_id


def evaluate(spec):
    from lightgbm import Booster

    path, args = spec
    upstream = [Booster(model_file=str(asset)) for asset in ASSETS.values()]
    pairing = PairFrameCapture(Booster(model_file=str(args.paired_returns)), .985, .95, wide=True,
                               fallback_floor=.96)
    gate = RawGateEvidence(Booster(model_file=str(args.raw_model)), .025)
    seed = Booster(model_file=str(args.seeds))
    tracker = RawAwareBirths(*upstream, pairing=pairing, raw_gate=gate, seed_model=seed,
                             raw_support=args.raw_support, seed_floor=args.seed_floor, margin=args.margin,
                             association_floor=args.association_floor)
    with (patch.object(tracker_compat, 'baseline_circle', tracker.track),
          patch.object(replay, 'load_handler', return_value=load_runtime().process_circle)):
        result = replay.replay_packet(path, allowed=TUNING_PACKETS)
    result['single_starts'] = dict(tracker.single_stats)
    print(path.name, round(result['score']['total'], 5), result['id_switches'], result['split_truth_tracks'],
          result['single_starts'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', nargs='+', type=int, required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--paired-returns', type=Path, default=HERE / 'models/paired_returns.txt')
    parser.add_argument('--raw-model', type=Path, default=HERE / 'models/raw_timing_detection.txt')
    parser.add_argument('--seeds', type=Path, default=HERE / 'models/young_assoc_seed.txt')
    parser.add_argument('--raw-support', type=float, default=.90)
    parser.add_argument('--seed-floor', type=float, default=.68)
    parser.add_argument('--margin', type=float, default=.12)
    parser.add_argument('--association-floor', type=float, default=.20)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in (args.raw_support, args.seed_floor, args.margin)):
        parser.error('probability thresholds must be between 0 and 1')
    if not 0 < args.association_floor < 1:
        parser.error('association floor must be between 0 and 1')
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    reports = parallel(evaluate, [(path, args) for path in paths], args.jobs)
    summary = aggregate_packets(reports)
    result = {
        'metadata': dict(packets=args.packets, wide=True, fallback_floor=.96, raw_floor=.025,
                         association_floor=args.association_floor,
                         raw_support=args.raw_support, seed_floor=args.seed_floor, margin=args.margin,
                         score_version=SCORE_VERSION,
                         source_sha256={path.name: replay.file_sha256(path) for path in (
                             Path(__file__), HERE / 'paired_returns.py', HERE / 'causal_birth_paths.py',
                             HERE / 'paired_single_start.py', HERE / 'raw_timing_detection.py')},
                         model_sha256={key: replay.file_sha256(path) for key, path in
                                       {**ASSETS, 'paired_returns': args.paired_returns, 'seeds': args.seeds,
                                        'raw_model': args.raw_model}.items()}),
        'summary': summary,
        'packets': [{key: value for key, value in stat.items() if key not in ('tracks', '_circle_times')}
                    for stat in reports],
    }
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('score', summary['score']['total'], 'switches', summary['id_switches'],
          'splits', summary['split_truth_tracks'], 'IDF1', summary['idf1'], flush=True)


if __name__ == '__main__':
    main()
