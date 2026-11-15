"""Measure attainable precision and recall of one-scan assignment with clean past histories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from teacher_assoc import candidate_features, spawn, update
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from train_tracker import read_packet


RULES = ((.10, None), (.23, None), (.35, None), (.50, None), (.65, None), (.80, None),
         (.23, 0.), (.23, .20), (.23, .45), (.23, .75), (.23, 1.25))


def selected_scores(track_count, point_count, rows, cols, confidence):
    # First assign with a permissive reject option, then sweep emission confidence.
    cost = np.full((track_count, point_count + track_count), np.inf)
    utility = np.log(np.clip(confidence, 1e-6, 1.0))
    cost[rows, cols] = -utility
    cost[np.arange(track_count), point_count + np.arange(track_count)] = -np.log(.08)
    track, point = linear_sum_assignment(cost)
    selected = point < point_count
    track, point = track[selected], point[selected]
    chosen = np.zeros((track_count, point_count), dtype=bool)
    chosen[track, point] = True
    other = ~chosen[rows, cols]
    row_best = np.full(track_count, -np.inf)
    col_best = np.full(point_count, -np.inf)
    np.maximum.at(row_best, rows[other], utility[other])
    np.maximum.at(col_best, cols[other], utility[other])
    absolute = -cost[track, point]
    margin = absolute - np.maximum(row_best[track], col_best[point])
    return track, point, np.exp(absolute), margin


def probe(path, model):
    data = read_packet(path)
    breaks = np.r_[0, np.flatnonzero(np.diff(data[:, 6])) + 1, len(data)]
    previous = {}
    counts = np.zeros((len(RULES), 4), dtype=np.int64)
    present_count = 0
    for start, end in zip(breaks, breaks[1:]):
        scan = data[start:end]
        circle = int(scan[0, 6])
        positions = scan[:, 1:4].astype('f').astype(float)
        amplitudes = scan[:, 4].astype('f').astype(float)
        labels = scan[:, 8].astype(int)
        active = [(ident, track) for ident, track in previous.items()
                  if circle - track.circle <= 5 and track.hits >= 3]
        if active:
            ids, tracks = zip(*active)
            identities = np.asarray(ids)
            present_count += np.isin(identities, labels).sum()
            rows, cols, features = candidate_features(tracks, positions, amplitudes, circle)
            if len(rows):
                confidence = model.predict(features, num_threads=1)
                r, c, score, gap = selected_scores(len(active), len(positions), rows, cols, confidence)
                same = identities[r] == labels[c]
                for idx, (threshold, margin) in enumerate(RULES):
                    kept = score >= threshold
                    if margin is not None:
                        kept &= gap >= margin
                    counts[idx] += ((kept & same).sum(), (kept & ~same & (labels[c] > 0)).sum(),
                                    (kept & (labels[c] == 0)).sum(), (kept & (labels[c] < 0)).sum())
        for j in np.flatnonzero(labels > 0):
            ident = labels[j]
            track = previous.get(ident)
            if not track or not 0 < circle - track.circle <= 5:
                previous[ident] = spawn(positions[j], amplitudes[j], circle)
            else:
                update(track, positions[j], amplitudes[j], circle)
    return counts, int(present_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    files = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    from lightgbm import Booster

    model = Booster(model_file=str(args.weights))
    total = np.zeros((len(RULES), 4), dtype=np.int64)
    present = 0
    for path in files:
        count, targets = probe(path, model)
        total += count
        present += targets
        print(path.name, targets, count[1].tolist(), flush=True)
    options = []
    for (threshold, margin), (correct, wrong, clutter, unknown) in zip(RULES, total.tolist()):
        predicted = correct + wrong + clutter + unknown
        options.append(dict(threshold=threshold, log_margin=margin, correct=correct, wrong_id=wrong,
                            clutter=clutter, unknown=unknown, recall=correct/present,
                            precision=correct/predicted if predicted else None))
    report = dict(scope='offline oracle with true past measurements', packet_ids=args.packets,
                  weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  present=present, emission_rules=options)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for row in options:
        print(row, flush=True)


if __name__ == '__main__':
    main()
