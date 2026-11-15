"""Limit fast-angular births to proposals without evidence of paired returns."""

import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fast_angular_birth as angular
from causal_birth_paths import POINT
from metrics import SCORE_VERSION, aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from paired_returns import ASSETS, HERE, measured_pairs
from paired_single_start import mature_competitor
import replay_validation as replay


BaseAngularBirths = angular.FastAngularBirths


def paired_point_ids(points):
    """Conservatively reserve every return in a plausible neighboring-tick pair."""
    if len(points) < 2:
        return frozenset()
    raw = np.asarray([(p['Azimuth'], p['Elevation'], p['Distance'], p['Amp'], 1.) for p in points])
    times = np.fromiter((p['Time'] for p in points), dtype=np.int64, count=len(points))
    possible = measured_pairs(raw, times, wide=True)
    return frozenset(points[col]['PntID'] for col in np.unique(possible))


class SingletonAngularBirths(BaseAngularBirths):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proposal_counts = Counter()
        self.path_ids = []

    def track(self, points, state, observe=None):
        circle = points[0]['CircleID'] if points else state.get('circle', -1) + 1
        recent = state.setdefault('singleton_angular_pairs', {})
        state['singleton_angular_pairs'] = {scan: ids for scan, ids in recent.items() if circle-scan <= 2}
        # A pressure-scan record with no complete pair enumeration disables
        # the supplementary path for every trio containing this scan.
        state['singleton_angular_pairs'][circle] = paired_point_ids(points) if len(points) <= 1200 else None
        self._scan_state = state
        return super().track(points, state, observe)

    def propose(self, first, second, third):
        state = self._scan_state
        circle = state['circle']
        recent = state['singleton_angular_pairs']
        reserved = [recent.get(scan) for scan in range(circle-2, circle+1)]
        if any(group is None for group in reserved):
            self.proposal_counts['unavailable_pair_context'] += 1
            return []
        # Keep all original rivals in the score margin. Reserving a pair
        # should never expose a previously ambiguous three-hit path.
        proposals = BaseAngularBirths.propose(first, second, third)
        self.proposal_counts['fast_angular_proposals'] += len(proposals)
        filtered = []
        for triple in proposals:
            if any(row[POINT] in group for row, group in zip(triple, reserved)):
                self.proposal_counts['paired_endpoint_veto'] += 1
            else:
                filtered.append(triple)
        if not filtered:
            return []
        columns = tuple(np.stack(endpoints) for endpoints in zip(*filtered))
        active = state['tracks'][0]
        old_owner = (mature_competitor(active, columns[0], circle-2)
                     | mature_competitor(active, columns[1], circle-1))
        self.proposal_counts['older_mature_veto'] += int(old_owner.sum())
        selected = [triple for triple, occupied in zip(filtered, old_owner) if not occupied]
        self.path_ids.extend(tuple(int(row[POINT]) for row in triple) for triple in selected)
        return selected


def compare(spec):
    path, variant = spec
    if variant == 'control':
        result, endpoint = angular.evaluate((path, False, TUNING_PACKETS))
        return path.name, variant, result, endpoint, Counter(), Counter()

    trackers, captured = [], {}
    iterate = replay.iter_circle_payloads

    def record_tracker(*args, **kwargs):
        tracker = SingletonAngularBirths(*args, **kwargs)
        trackers.append(tracker)
        return tracker

    def record_truth(path, labels, circles, allowed):
        captured['labels'] = labels
        return iterate(path, labels, circles, allowed)

    with (patch.object(angular, 'FastAngularBirths', record_tracker),
          patch.object(replay, 'iter_circle_payloads', record_truth)):
        result, endpoint = angular.evaluate((path, True, TUNING_PACKETS))
    tracker = trackers[0]
    labels = captured['labels']
    root = Counter()
    for triple in tracker.path_ids:
        a, b, c = map(labels.__getitem__, triple)
        key = ('same_positive_trajectory' if a == b == c > 0 else 'different_positives'
               if min(a, b, c) > 0 else 'unknown_or_clutter_in_path')
        root[key] += 1
    print(path.name, 'switches', result['id_switches'], 'singleton paths', dict(root),
          'vetoes', dict(tracker.proposal_counts), flush=True)
    return path.name, variant, result, endpoint, tracker.proposal_counts, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    rows = angular.parallel(compare, [(path, variant) for variant in ('control', 'singleton') for path in paths],
                            args.jobs)
    report = {
        'metadata': {'packets': args.packets, 'profile': angular.PROFILE, 'score_version': SCORE_VERSION,
                     'source_sha256': {name: replay.file_sha256(HERE / name) for name in (
                         Path(__file__).name, 'fast_angular_birth.py', 'causal_birth_paths.py', 'paired_returns.py',
                         'raw_birth_stage_audit.py', 'emission_identity.py')},
                     'model_sha256': {key: replay.file_sha256(path) for key, path in (
                         ASSETS | {'paired_returns': angular.PAIR_MODEL, 'raw_timing': angular.RAW_MODEL}
                         | angular.EMISSION_MODELS).items()}},
        'variants': {},
    }
    for name in ('control', 'singleton'):
        selected = [row for row in rows if row[1] == name]
        summary = aggregate_packets(row[2] for row in selected)
        report['variants'][name] = {
            'summary': summary,
            'endpoint_labels': dict(sum((row[3] for row in selected), Counter())),
            'proposals': dict(sum((row[4] for row in selected), Counter())),
            'path_labels': dict(sum((row[5] for row in selected), Counter())),
            'packets': {row[0]: {key: value for key, value in row[2].items() if key != 'tracks'} for row in selected},
        }
        print(name, 'score', summary['score']['total'], 'switches', summary['id_switches'], 'splits',
              summary['split_truth_tracks'], 'IDF1', summary['idf1'], flush=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
