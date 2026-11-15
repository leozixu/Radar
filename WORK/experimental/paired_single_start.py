"""Test two-hit single births with exclusive geometry, timing and mature-owner vetoes."""

import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paired_returns import ASSETS, HERE, PairReturns
from causal_birth_paths import AMPLITUDE, CLOCK, POINT, POSITION, BirthOnline, parallel
from algorithm import load_runtime
from baseline_cv import residual
from metrics import SCORE_VERSION, aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
import replay_validation as replay
import tracker_compat
from young_assoc import edge_features


class PairFrameCapture(PairReturns):
    def begin(self, tracks, raw, times, circle, state):
        self.raw = raw
        self.frame = super().begin(tracks, raw, times, circle, state)
        return self.frame


def reciprocal_seeds(before, now, model, floor, margin):
    rows, cols, features = edge_features(before[:, :4], before[:, :4], now[:, :4], initial=True)
    if not len(rows):
        return []
    score = model.predict(features, num_threads=1)
    row_best = np.full(len(before), -np.inf)
    col_best = np.full(len(now), -np.inf)
    np.maximum.at(row_best, rows, score)
    np.maximum.at(col_best, cols, score)
    candidate = (score >= floor) & (score == row_best[rows]) & (score == col_best[cols])
    if not candidate.any():
        return []
    # Ties remain competitors. Exclude only a unique reciprocal best edge.
    row_ties = np.bincount(rows[score == row_best[rows]], minlength=len(before))
    col_ties = np.bincount(cols[score == col_best[cols]], minlength=len(now))
    candidate &= (row_ties[rows] == 1) & (col_ties[cols] == 1)
    other = ~candidate
    row_other = np.full(len(before), -np.inf)
    col_other = np.full(len(now), -np.inf)
    np.maximum.at(row_other, rows[other], score[other])
    np.maximum.at(col_other, cols[other], score[other])
    separated = score >= np.maximum(row_other[rows], col_other[cols]) + margin
    valid = candidate & separated & (features[:, 5] >= .010) & (features[:, 5] <= .85)
    return zip(rows[valid], cols[valid])


def mature_competitor(active, proposed, circle):
    track_positions = np.array([track.position + track.velocity * (circle-track.circle)
                                for track in active if track.hits >= 3]).reshape(-1, 3)
    if not len(track_positions):
        return np.zeros(len(proposed), dtype=bool)
    error = residual(proposed[:, None, POSITION], track_positions[None, :, :])
    return (np.square(error / [.011, .075, .0045]).sum(axis=2) <= 1.).any(axis=1)


class EarlySingleBirths(BirthOnline):
    def __init__(self, *upstream, pairing, seed_model, seed_floor, clock_floor, margin):
        super().__init__(*upstream, .86, "supplement", paired_birth=True, joint=True, two_hit_pairs=True,
                         pairing=pairing)
        self.seed_model, self.seed_floor, self.clock_floor, self.margin = seed_model, seed_floor, clock_floor, margin
        self.single_stats = Counter()

    def promote_legacy(self, points, state, active, position, amp, circle, remaining, output, next_id, observe,
                       reserved_keys):
        current_free = remaining.copy()
        tentative, pending, taken, next_id = super().promote_legacy(
            points, state, active, position, amp, circle, remaining, output, next_id, observe, reserved_keys)
        # The ordinary solver continues to handle anomalously large scans.
        if len(points) > 1200 or not current_free.any():
            return tentative, pending, taken, next_id
        raw, pair_frame = self.pairing.raw, self.pairing.frame
        current_free[pair_frame.columns.ravel()] = False
        established = {*reserved_keys, *taken, *output.keys()}
        current_free &= ((raw[:, CLOCK] >= self.clock_floor) & (raw[:, AMPLITUDE] >= .15)
                         & ~np.isin(raw[:, POINT], tuple(established)))
        columns = np.flatnonzero(current_free)
        previous = next((rows for scan, rows in state['birth_path_pool'].observations if scan == circle - 1), ())
        if not len(columns) or not len(previous):
            return tentative, pending, taken, next_id
        previous = previous[(previous[:, CLOCK] >= self.clock_floor) & (previous[:, AMPLITUDE] >= .15)
                            & ~np.isin(previous[:, POINT], tuple(established))]
        if not len(previous):
            return tentative, pending, taken, next_id
        candidates = raw[columns]
        matches = list(reciprocal_seeds(previous, candidates, self.seed_model, self.seed_floor, self.margin))
        self.single_stats['reciprocal'] += len(matches)
        if not matches:
            return tentative, pending, taken, next_id
        blocked = mature_competitor(active, candidates[[c for _, c in matches]], circle)
        for (r, c), conflict in zip(matches, blocked):
            if conflict:
                self.single_stats['mature_veto'] += 1
                continue
            first, second = previous[r], candidates[c]
            self.confirm((first, second), active, output, next_id, observe)
            remaining[columns[c]] = False
            taken.update((first[POINT], second[POINT]))
            self.single_stats['confirmed'] += 1
            next_id += 1
        tentative = [track for track in tentative if not taken.intersection(pending[id(track)])]
        pending = {id(track): pending[id(track)] for track in tentative}
        return tentative, pending, taken, next_id


def evaluate(spec):
    from lightgbm import Booster

    path, args = spec
    upstream = [Booster(model_file=str(asset)) for asset in ASSETS.values()]
    pairing = PairFrameCapture(Booster(model_file=str(args.paired_returns)), .985, .95)
    seed_model = Booster(model_file=str(args.seeds))
    tracker = EarlySingleBirths(*upstream, pairing=pairing, seed_model=seed_model, seed_floor=args.seed_floor,
                                clock_floor=args.clock_floor, margin=args.margin)
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
    parser.add_argument('--seeds', type=Path, default=HERE / 'models/young_assoc_seed.txt')
    parser.add_argument('--seed-floor', type=float, default=.90)
    parser.add_argument('--clock-floor', type=float, default=.83)
    parser.add_argument('--margin', type=float, default=.13)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in (args.seed_floor, args.clock_floor, args.margin)):
        parser.error('probability thresholds must be between 0 and 1')
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    reports = parallel(evaluate, [(path, args) for path in paths], args.jobs)
    summary = aggregate_packets(reports)
    result = {
        'metadata': dict(packets=args.packets, seed_floor=args.seed_floor, clock_floor=args.clock_floor,
                         margin=args.margin, score_version=SCORE_VERSION,
                         source_sha256={path.name: replay.file_sha256(path) for path in (
                             Path(__file__), HERE / 'paired_returns.py', HERE / 'causal_birth_paths.py')},
                         model_sha256={key: replay.file_sha256(path) for key, path in
                                       {**ASSETS, 'paired_returns': args.paired_returns, 'seeds': args.seeds}.items()}),
        'summary': summary,
        'packets': [{key: value for key, value in result.items() if key not in ('tracks', '_circle_times')}
                    for result in reports],
    }
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('score', summary['score']['total'], 'switches', summary['id_switches'],
          'splits', summary['split_truth_tracks'], 'IDF1', summary['idf1'], flush=True)


if __name__ == '__main__':
    main()
