"""Measure a conservative single-observation public start on tuning packets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lightgbm import Booster

from algorithm import load_runtime
from baseline_cv import OUTPUT_FIELDS, residual
from early_birth_candidate import EarlyBirthRuntime, EarlyBirthTracker, row_key
from metrics import SCORE_VERSION, aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from precision_tracking.causal_birth_paths import AMPLITUDE, CLOCK, POINT, POSITION, SCAN
from precision_tracking.runtime import PrecisionRuntime
import replay_validation as replay


CURRENT_ISOLATION = np.array([.018, .055, .010])
INCUMBENT_WIDTH = np.array([.030, .160, .014])


def isolated_from_incumbents(rows, tracks):
    established = [track for track in tracks if track.hits >= 2]
    if not established:
        return np.ones(len(rows), dtype=bool)
    age = rows[:, SCAN, None] - np.array([track.circle for track in established])[None, :]
    predicted = np.stack([track.position for track in established])[None, :, :]
    predicted = predicted + age[:, :, None] * np.stack([track.velocity for track in established])[None, :, :]
    distance = np.sum((residual(rows[:, None, POSITION], predicted) / INCUMBENT_WIDTH) ** 2, axis=2)
    return ~((distance <= 1.) & (abs(age) <= 12)).any(axis=1)


class IsolatedFirstHitTracker(EarlyBirthTracker):
    def __init__(self, native, model, floor, margin, single_floor):
        super().__init__(native, model, floor, margin, None, 'provisional', None, False)
        self.single_floor = single_floor
        self.single_proposals = []
        self.single_ids = []

    def track(self, points, state, observe=None):
        outputs = super().track(points, state, observe)
        self.single_proposals = []
        if not points or len(points) > 1600:
            return outputs
        circle = state['circle']
        pool = dict(state['birth_path_pool'].observations)
        current = pool.get(circle)
        if current is None:
            return outputs
        scores = self.raw_by_circle[circle]
        paired = self.pairs_by_circle[circle]
        available = current[(current[:, CLOCK] >= .70) & (current[:, AMPLITUDE] >= .20)]
        selected = [row for row in available if scores.get(int(row[POINT]), 0.) >= self.single_floor
                    and int(row[POINT]) not in paired]
        if not selected:
            return outputs
        selected = np.stack(selected)
        selected = selected[isolated_from_incumbents(selected, state['tracks'][0])]
        if not len(selected):
            return outputs

        references = [p for p in points if scores.get(p['PntID'], 0.) >= .85]
        if not references:
            return outputs
        position = np.array([(p['Azimuth'], p['Elevation'], p['Distance']) for p in references])
        identifiers = np.fromiter((p['PntID'] for p in references), dtype=np.int64)
        distance = np.sum((residual(selected[:, None, POSITION], position[None, :, :]) / CURRENT_ISOLATION) ** 2,
                          axis=2)
        near = (distance <= 1.) & (selected[:, POINT, None] != identifiers[None, :])
        history = [row_key(row) for t, rows in pool.items() if t < circle
                   for row in rows if self.raw_by_circle.get(t, {}).get(int(row[POINT]), 0.) >= .85]
        self.single_proposals = [row for row in selected[~near.any(axis=1)] if not self.conflicts(row, history)]
        return outputs

    def publish(self, outputs, points, state):
        outputs = super().publish(outputs, points, state)
        if not self.single_proposals:
            return outputs
        circle = state['circle']
        previous = [key for key in self.provisional if key[0] < circle]
        assigned = {row['PntID'] for row in outputs}
        current = {p['PntID']: p for p in points}
        next_id = state['next_batch_id']
        for row in self.single_proposals:
            point_id = int(row[POINT])
            if point_id in assigned or self.conflicts(row, previous):
                continue
            point = current[point_id]
            outputs.append({**{key: point[key] for key in OUTPUT_FIELDS}, 'BatchID': next_id})
            self.provisional[row_key(row)] = next_id
            self.single_ids.append(point_id)
            next_id += 1
        state['next_batch_id'] = next_id
        return outputs


@dataclass(frozen=True, slots=True)
class IsolatedFirstHitRuntime(EarlyBirthRuntime):
    single_floor: float = .997

    def new_packet(self):
        tracker, history = PrecisionRuntime.new_packet(self)
        return IsolatedFirstHitTracker(tracker, self.birth_model, self.floor, self.margin, self.single_floor), history


def compare(spec):
    path, model_path, floor = spec
    original = load_runtime()
    model = Booster(model_file=str(model_path))
    runtime = IsolatedFirstHitRuntime(original.settings, original.models, original.metadata, model,
                                     .90, .12, mode='provisional', single_floor=floor)
    ledger, label_maps = [], []
    def process(points, state):
        if not ledger:
            ledger.append(state)
        return runtime.process_circle(points, state)

    source = replay.iter_circle_payloads
    def collect_truth(path, labels, circles, allowed):
        label_maps.append(labels)
        return source(path, labels, circles, allowed)

    with (patch.object(replay, 'load_handler', return_value=process),
          patch.object(replay, 'iter_circle_payloads', side_effect=collect_truth)):
        report = replay.replay_packet(path, allowed=TUNING_PACKETS)
    added = ledger[0]['_precision_packet'][0].single_ids
    report['single_starts'] = dict(total=len(added), true=sum(label_maps[0][pid] > 0 for pid in added))
    print(path.name, floor, round(report['score']['total'], 5), 'first hits', report['single_starts'],
          'splits', report['split_truth_tracks'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--floors', type=float, nargs='+', default=(.99, .997))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(args.data_dir, args.packets, TUNING_PACKETS)
    specs = [(path, args.model, floor) for floor in args.floors for path in paths]
    with ProcessPoolExecutor(max_workers=args.jobs) as workers:
        results = list(workers.map(compare, specs))
    report = {'metadata': {'packet_ids': args.packets, 'score_version': SCORE_VERSION,
                           'source_sha256': replay.file_sha256(Path(__file__)),
                           'model_sha256': replay.file_sha256(args.model)}, 'variants': {}}
    for index, floor in enumerate(args.floors):
        packets = results[index * len(paths):(index + 1) * len(paths)]
        total = aggregate_packets(packets)
        report['variants'][str(floor)] = dict(summary=total, packets=[
            {k: v for k, v in row.items() if k not in ('tracks', '_circle_times')} for row in packets])
        print(floor, total['score']['total'], flush=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
