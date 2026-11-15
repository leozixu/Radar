"""Compare emission rules on identical causal associations in the tuning packets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from teacher_assoc import Online
from clock_association import phase_features
from algorithm import load_runtime
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, encode_tracks, iter_circle_payloads
import tracker_compat


# (normal quality, healthily high link threshold & relaxed quality, weak link
# cutoff & cautious quality, qualified early emission quality).
RULES = {
    'reference': (.16, 1.01, .16, 0., .16, .16),
    'relaxed': (.12, 1.01, .12, 0., .12, .12),
    'link_supported': (.16, .85, .05, 0., .16, .16),
    'link_conservative': (.16, .85, .05, .55, .28, .16),
    'earlier_start': (.16, .85, .05, 0., .16, .06),
    'conservative_early': (.16, .85, .05, .55, .28, .06),
    'link_priority': (.16, .95, 0., .70, .30, .065),
}


def worker(spec):
    path, teacher_path, clock_path = spec
    from lightgbm import Booster

    teacher = Online(Booster(model_file=str(teacher_path)), floor=.20, mix=0., coast=7,
                     ambiguity=.30, robust=4.)
    clock = Booster(model_file=str(clock_path))
    compat = load_runtime()
    state, labels, circles, times = {}, {}, {}, {}
    emitted = {name: {} for name in RULES}
    with patch.object(tracker_compat, 'baseline_circle', teacher.track):
        for _, payload in iter_circle_payloads(path, labels, circles, allowed=TUNING_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            times.update((p['PntID'], p['Time']) for p in points)
            result, x = tracker_compat.decisions_with_features(points, state)
            if not result:
                continue
            source_points = {(p['Time'], p['CircleID'], p['PntID']): (p['Azimuth'], p['Elevation'], p['Distance'])
                             for p in points}
            # Filtering this valid list preserves the packet output contract for every variant.
            encode_tracks(result, source_points)
            detection = compat.quality_model.predict(x, num_threads=1)
            scores = state['association_confidence']
            links = np.fromiter((scores.get(p['PntID'], np.nan) for p in result), dtype=float)
            newborn = np.isnan(links)
            early = np.zeros(len(result), dtype=bool)
            if newborn.any():
                azimuth = np.fromiter((p['Azimuth'] for p in points), dtype=float)
                timestamp = np.fromiter((p['Time'] for p in points), dtype=float)
                amplitude = np.fromiter((p['Amp'] for p in points), dtype=float)
                clock_x = phase_features(azimuth, timestamp, amplitude)
                if clock_x[0, 1] >= .30:
                    index = {p['PntID']: j for j, p in enumerate(points)}
                    newborn_rows = np.flatnonzero(newborn)
                    selected = [index[result[j]['PntID']] for j in newborn_rows]
                    usable = clock.predict(clock_x[selected], num_threads=1) >= .90
                    early[newborn_rows] = usable
            for name, (base, good, quality_good, weak, quality_weak, startup) in RULES.items():
                threshold = np.full(len(result), base)
                threshold[links >= good] = quality_good
                threshold[links < weak] = quality_weak
                threshold[early] = startup
                emitted[name].update((result[j]['PntID'], result[j]['BatchID'])
                                     for j in np.flatnonzero(detection >= threshold))
    reports = {name: evaluate_packet(labels, by_id, circles, times) for name, by_id in emitted.items()}
    print(path.name, {k: round(v['score']['total'], 4) for k, v in reports.items()}, flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--clock', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    inputs = [(path, args.teacher, args.clock) for path in paths]
    if args.jobs == 1:
        reports = list(map(worker, inputs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            reports = list(pool.map(worker, inputs))
    variants = {name: aggregate_packets(packet[name] for packet in reports) for name in RULES}
    header = dict(packet_ids=args.packets, rules=RULES,
                  teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                  clock_sha256=hashlib.sha256(args.clock.read_bytes()).hexdigest())
    args.report.write_text(json.dumps(dict(metadata=header, variants=variants), indent=2) + '\n', encoding='utf-8')
    for name, variant in variants.items():
        print(name, round(variant['score']['total'], 4), round(variant['idf1'], 5),
              variant['id_switches'], variant['split_truth_tracks'], flush=True)


if __name__ == '__main__':
    main()
