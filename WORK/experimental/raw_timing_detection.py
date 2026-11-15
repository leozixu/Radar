"""Learn per-return evidence from within-scan radar timing and neighboring measurements."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from clock_association import phase_features
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
import replay_validation as replay


FEATURES = ("amplitude", "scan_phase", "phase_alignment", "same_tick_count", "near_tick_count") + tuple(
    f"{neighbour}_{part}"
    for neighbour in ("previous", "previous_2", "next", "next_2")
    for part in ("elapsed", "azimuth_delta", "elevation_delta", "range_delta", "amplitude", "amplitude_difference")
)


class RawReturnGate:
    def __init__(self, model, floor):
        self.model = model
        self.floor = floor

    def probabilities(self, positions, amplitude, times):
        sample = features(positions, amplitude, times)
        # An unaligned or malformed time stream falls back to the regular tracker.
        if len(sample) < 5 or sample[0, 2] < .30 or (sample[:, 3] == 1).mean() < .85:
            return np.full(len(sample), np.nan)
        return self.model.predict(sample, num_threads=1)

    def __call__(self, positions, amplitude, times):
        probability = self.probabilities(positions, amplitude, times)
        return np.isnan(probability) | (probability >= self.floor)


def features(positions, amplitude, time):
    """Use time spacing and source measurements from the available current scan."""
    count = len(time)
    if count == 0:
        return np.empty((0, len(FEATURES)), dtype=np.float32)
    order = np.argsort(time, kind="stable")
    inverse = np.empty(count, dtype=int)
    inverse[order] = np.arange(count)
    positions, amplitude, time = positions[order], amplitude[order], time[order]
    phase = phase_features(positions[:, 0], time, amplitude)
    duplicate = np.searchsorted(time, time, side="right") - np.searchsorted(time, time, side="left")
    local = np.searchsorted(time, time + 8, side="right") - np.searchsorted(time, time - 8, side="left")
    parts = [np.column_stack((amplitude, phase[:, :2], duplicate, local))]
    indices = np.arange(count)
    for step in (-1, -2, 1, 2):
        neighbor = indices + step
        present = (neighbor >= 0) & (neighbor < count)
        neighbor = np.clip(neighbor, 0, count - 1)
        elapsed = abs(time[neighbor] - time)
        delta = positions - positions[neighbor]
        delta[:, 0] = (delta[:, 0] + .5) % 1. - .5
        other_amp = amplitude[neighbor]
        value = np.column_stack((elapsed, delta, other_amp, amplitude - other_amp))
        value[~present] = np.nan
        parts.append(value)
    return np.column_stack(parts)[inverse].astype(np.float32)


def load(path):
    labels, circles = {}, {}
    x, y = [], []
    for _, payload in replay.iter_circle_payloads(path, labels, circles, TRAIN_PACKETS):
        points = np.asarray(list(replay.POINT.iter_unpack(payload)), dtype=float).reshape(-1, len(replay.POINT_KEYS))
        target = np.fromiter((labels[int(point)] for point in points[:, 7]), dtype=np.int64)
        x.append(features(points[:, 1:4], points[:, 4], points[:, 0]))
        y.append(target)
    print(path.stem, len(labels), flush=True)
    return np.concatenate(x), np.concatenate(y)


def report_probabilities(model, data):
    from sklearn.metrics import roc_auc_score

    x, labels = data
    q = model.predict(x, num_threads=1)
    true, clutter, unknown = labels > 0, labels == 0, labels < 0
    stats = {"auc_known": float(roc_auc_score(true[~unknown], q[~unknown]))}
    for floor in (.001, .005, .01, .025, .05, .20, .50, .80, .90, .95, .98, .99):
        selected = q >= floor
        stats[str(floor)] = {
            "truth_recall": float(selected[true].mean()), "clutter_keep_rate": float(selected[clutter].mean()),
            "unknown_keep_rate": float(selected[unknown].mean()) if unknown.any() else None,
            "known_precision": float(true[selected & ~unknown].mean()) if (selected & ~unknown).any() else None,
        }
    return stats


def main():
    from lightgbm import Dataset, train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--check", type=int, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    train_paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    check_paths = packet_paths(DEFAULT_DATA_DIR, args.check, TRAIN_PACKETS)
    if set(train_paths).intersection(check_paths):
        parser.error("use separate training and check packets")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        train_rows = list(pool.map(load, train_paths))
    x, y = (np.concatenate(rows) for rows in zip(*train_rows))
    known = y >= 0
    params = dict(objective="binary", num_threads=args.jobs, num_leaves=63, min_data_in_leaf=120,
                  learning_rate=.065, seed=803, force_col_wise=True, deterministic=True, verbosity=-1)
    model = train(params, Dataset(x[known], label=y[known] > 0, feature_name=list(FEATURES)), num_boost_round=200)
    model.save_model(str(args.model))
    checks = {path.stem: report_probabilities(model, load(path)) for path in check_paths}
    report = {
        "training_packets": args.packets, "check_packets": args.check, "rows_by_class": [int((y < 0).sum()),
                                                                                       int((y == 0).sum()),
                                                                                       int((y > 0).sum())],
        "checks": checks, "parameters": params, "features": FEATURES,
        "source_sha256": replay.file_sha256(Path(__file__)), "model_sha256": replay.file_sha256(args.model),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for packet, stat in checks.items():
        print(packet, stat["auc_known"], "floor .025", stat["0.025"], "floor .95", stat["0.95"], flush=True)


if __name__ == "__main__":
    main()
