"""Compare a one-outlier history filter for the joint elevation matcher."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

if __package__:
    from . import joint_elevation_pairs as joint
    from .relative_elevation_assoc import HISTORY
else:
    import joint_elevation_pairs as joint
    from relative_elevation_assoc import HISTORY

from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from replay_validation import file_sha256


BASE_PROFILE = dict(spread=.0005, tolerance=.0009, bonus=1.4, min_probability=.065,
                    minimum_gain=.25, max_groups=3, minimum_history=5)


def trimmed_forecast(differences, minimum_count, spread_limit):
    """Fit after excluding exactly one prior observation and reject conflicting fits."""
    mask = np.isfinite(differences)
    delta = np.where(mask, differences, 0.)
    times = HISTORY[None, :]
    count = mask.sum(axis=1, keepdims=True) - 1
    x_sum = (mask * times).sum(axis=1, keepdims=True) - times
    x2_sum = (mask * times**2).sum(axis=1, keepdims=True) - times**2
    y_sum = delta.sum(axis=1, keepdims=True) - delta
    xy_sum = (delta * times).sum(axis=1, keepdims=True) - delta * times
    y2_sum = (delta**2).sum(axis=1, keepdims=True) - delta**2
    denominator = np.maximum(count * x2_sum - x_sum**2, 1.)
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    estimate = (y_sum - slope * x_sum) / np.maximum(count, 1)
    variance = np.maximum(y2_sum - estimate * y_sum - slope * xy_sum, 0.) / np.maximum(count - 2, 1)
    recent = mask[:, -5:].sum(axis=1, keepdims=True) - (times >= -5)
    now_recent = mask[:, -3:].sum(axis=1, keepdims=True) - (times >= -3)
    gap = abs(delta - estimate - slope * times)
    valid = (mask & (count >= minimum_count) & (recent >= 2) & (now_recent >= 1)
             & (variance <= spread_limit**2) & (abs(slope) <= .003) & (gap >= 3 * spread_limit))
    score = np.where(valid, variance, np.inf)
    winner = score.argmin(axis=1)
    difference = np.where(valid, estimate, -np.inf).max(axis=1) - np.where(valid, estimate, np.inf).min(axis=1)
    accepted = valid.any(axis=1) & (difference <= .0004)
    return estimate[np.arange(len(delta)), winner], accepted


class RobustPairHistory:
    def __init__(self, original, minimum_inliers=5):
        self.original = original
        self.minimum_inliers = minimum_inliers
        self.stats = Counter()

    def __call__(self, tracks, circle, spread_limit, available_rows, minimum_history=5):
        clean = self.original(tracks, circle, spread_limit, available_rows, minimum_history)
        minimum = self.minimum_inliers
        eligible = np.asarray([len(track.history) >= minimum + 1 for track in tracks]) & available_rows
        for first, second, _ in clean:
            eligible[[first, second]] = False
        group = np.flatnonzero(eligible)
        if len(group) < 2:
            return clean

        history = joint.recent_elevation([tracks[i] for i in group], circle)
        _, individual_spread, _, _ = joint.fit_relative_elevation(history, np.zeros((1, history.shape[1])))
        dynamic = individual_spread[:, 0] >= .002
        history, group = history[dynamic], group[dynamic]
        if len(group) < 2:
            return clean
        left, right = np.triu_indices(len(group), k=1)
        differences = history[left] - history[right]
        # Keep the leave-one-out fit limited to pairs with enough common history.
        possible = np.isfinite(differences).sum(axis=1) >= minimum + 1
        left, right, differences = left[possible], right[possible], differences[possible]
        if not len(differences):
            return clean

        estimate, accepted = trimmed_forecast(differences, minimum, spread_limit)
        left, right, estimate = left[accepted], right[accepted], estimate[accepted]
        degree = np.bincount(np.r_[left, right], minlength=len(group))
        clear = (degree[left] == 1) & (degree[right] == 1)
        restored = [(int(group[a]), int(group[b]), float(offset))
                    for a, b, offset in zip(left[clear], right[clear], estimate[clear])]
        self.stats["restored_history_pairs"] += len(restored)
        self.stats["discarded_ambiguous_relationships"] += int((~clear).sum())
        return clean + restored


def _test_packet(config, path):
    robust = RobustPairHistory(joint.stable_pairs, config.minimum_inliers)
    with patch.object(joint, "stable_pairs", robust):
        stats, diagnostics = joint._test_packet(config, path)
    print(path.name, dict(robust.stats), flush=True)
    return stats, diagnostics | robust.stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--minimum-inliers", type=int, choices=(5, 6, 7), default=5)
    parser.add_argument("--report", type=Path, required=True)
    parser.set_defaults(**BASE_PROFILE)
    args = parser.parse_args()
    try:
        paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    except ValueError as error:
        parser.error(str(error))
    runner = partial(_test_packet, args)
    if args.jobs == 1:
        results = list(map(runner, paths))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(runner, paths))
    stats, diagnostics = zip(*results)
    total = aggregate_packets(stats)
    metadata = {"packet_ids": args.packets, "source_sha256": file_sha256(Path(__file__)),
                "joint_source_sha256": file_sha256(Path(joint.__file__)),
                "teacher_sha256": file_sha256(args.teacher), "pair_sha256": file_sha256(args.pair),
                "pair_cutoff": .85, "settings": BASE_PROFILE | {"minimum_inliers": args.minimum_inliers}}
    report = {"metadata": metadata, "summary": total, "diagnostics": sum(map(Counter, diagnostics), Counter()),
              "packets": [{k: v for k, v in item.items() if k != "tracks"} for item in stats]}
    print({k: total[k] for k in ("score", "idf1", "id_switches", "split_truth_tracks", "merged_output_tracks")})
    print(report["diagnostics"])
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
