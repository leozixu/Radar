"""Check raw single-return dropout roles for already-supported true return pairs."""

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from clock_association import phase_features
from paired_returns import AMPLITUDE_TOLERANCE, PAIR_WIDTH
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import POINT, iter_circle_payloads


MAX_PAIR_AGE = 12


def diagnose(path):
    labels, circles = {}, {}
    pair_history = defaultdict(lambda: deque(maxlen=5))
    result = Counter()
    differences = []
    role_features = []
    for circle, payload in iter_circle_payloads(path, labels, circles, TRAIN_PACKETS):
        input_count = Counter()
        truth = []
        raw = list(POINT.iter_unpack(payload))
        if not raw:
            continue
        stamp, azimuth, amplitude = np.asarray(raw)[:, (0, 1, 4)].T
        phase_error = phase_features(azimuth, stamp, amplitude)[:, 0]
        by_truth = {}
        for i, row in enumerate(raw):
            stamp, *_, point = row
            input_count[stamp] += 1
            target = labels[point]
            if target > 0:
                truth.append((target, row))
                by_truth[target] = (phase_error[i], stamp % 2, row[4])
        current = by_truth.keys()
        for pair, history in pair_history.items():
            if len(history) < 3 or circle - history[-1][0] > MAX_PAIR_AGE:
                continue
            earlier = pair[0] in current
            later = pair[1] in current
            if earlier == later:
                continue
            role = "earlier" if earlier else "later"
            result[f"sole_{role}"] += 1
            result[f"sole_{role}_after_1_scan"] += circle - history[-1][0] == 1
            signed_amp = history[-1][1]
            result["sole_louder_previous_arm"] += (signed_amp < 0) == earlier
            role_features.append((earlier, *by_truth[pair[0 if earlier else 1]]))
        truth.sort(key=lambda item: item[1][0])
        for (left, a), (right, b) in zip(truth, truth[1:]):
            if left == right or b[0] - a[0] != 1 or input_count[a[0]] != 1 or input_count[b[0]] != 1:
                continue
            az = (b[1] - a[1] + .5) % 1. - .5
            if abs(az) > PAIR_WIDTH[0] or abs(b[2] - a[2]) > PAIR_WIDTH[1] or abs(b[3] - a[3]) > PAIR_WIDTH[2]:
                continue
            difference = b[4] - a[4]
            if not .000004 <= abs(difference) <= .032:
                continue
            history = pair_history[left, right]
            if history and (circle - history[-1][0] > MAX_PAIR_AGE or
                            abs(difference - history[-1][1]) > AMPLITUDE_TOLERANCE):
                history.clear()
            if len(history) >= 2:
                result["repeated_true_pairs"] += 1
                result["earlier_louder_in_repeat"] += difference < 0
                result["earlier_higher_elevation_in_repeat"] += a[2] > b[2]
                result["earlier_farther_in_repeat"] += a[3] > b[3]
                result["earlier_greater_azimuth_in_repeat"] += az < 0
                differences.append((az, b[2] - a[2], b[3] - a[3], difference))
            history.append((circle, difference))
        for pair, history in list(pair_history.items()):
            if circle - history[-1][0] > MAX_PAIR_AGE:
                del pair_history[pair]
    print(path.stem, dict(result), flush=True)
    return result, np.asarray(differences).reshape(-1, 4), np.asarray(role_features).reshape(-1, 4)


def role_separation(rows):
    """AUC for recognizing the earlier role from the current sole observation."""
    answer = {}
    for index, name in enumerate(("absolute_phase", "timestamp_odd", "absolute_amplitude"), 1):
        observed = rows[np.isfinite(rows[:, index])]
        earlier = observed[:, 0] == 1
        first, second = int(earlier.sum()), int((~earlier).sum())
        if first and second:
            ranks = rankdata(observed[:, index])
            area = (ranks[earlier].sum() - first * (first + 1) / 2) / (first * second)
            answer[name] = {"auc": float(area), "earlier_n": first, "later_n": second,
                            "earlier_median": float(np.median(observed[earlier, index])),
                            "later_median": float(np.median(observed[~earlier, index]))}
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    evaluated = {path.stem: diagnose(path) for path in paths}
    per_packet = {name: counts for name, (counts, _, _) in evaluated.items()}
    differences = np.concatenate([parts for _, parts, _ in evaluated.values()])
    role_features = np.concatenate([features for _, _, features in evaluated.values()])
    percentiles = np.array([0, 1, 10, 50, 90, 99, 100])
    report = {"packet_ids": args.packets, "by_packet": per_packet,
              "totals": sum(per_packet.values(), Counter()),
              "sole_return_role_measurements": role_separation(role_features),
              "late_minus_early_difference_percentiles": {
                  int(q): dict(zip(("azimuth", "elevation", "range", "amplitude"), values))
                  for q, values in zip(percentiles, np.percentile(differences, percentiles, axis=0))
              } if len(differences) else {}}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
