"""Fit a recovery scorer on missed associations from the frozen development replay."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "participant_demo" / "src"))

from algorithm import load_runtime
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from precision_tracking.causal_birth_paths import POINT
from precision_tracking.residual_links import FEATURES, TEMPORAL_FEATURES, ResidualHistory, residual_candidates
from replay_validation import POINT as WIRE_POINT, POINT_KEYS, file_sha256, iter_circle_payloads


class TrainingCollector:
    def __init__(self, labels, temporal):
        self.labels = labels
        self.history = defaultdict(lambda: deque(maxlen=7))
        self.features = []
        self.targets = []
        self.context = ResidualHistory() if temporal else None

    def recover(self, tracks, raw, times, probability, circle, frame, previous_pairs, matches, held, rejected):
        rows, cols, features = residual_candidates(
            tracks, raw, probability, circle, frame, previous_pairs, matches, held, rejected
        )
        if self.context is not None:
            features = self.context.features(rows, cols, features, tracks, raw, times, circle)
            self.context.advance(tracks, raw, times, probability, circle, matches, rejected)
        indices = []
        targets = []
        observed_truth = {self.labels[int(raw[col, POINT])] for _, col, _ in matches}
        for index, (row, col) in enumerate(zip(rows, cols)):
            history = self.history.get(tracks[row].batch_id)
            if history is None or len(history) < 3:
                continue
            owner, votes = Counter(history).most_common(1)[0]
            if owner <= 0 or votes < max(3, len(history) - 1):
                continue
            label = self.labels[int(raw[col, POINT])]
            indices.append(index)
            targets.append(label == owner and label not in observed_truth)
        if indices:
            self.features.append(features[indices])
            self.targets.extend(targets)
        return []

    def observe(self, outputs):
        for point in outputs:
            self.history[point["BatchID"]].append(self.labels[point["PntID"]])


def collect(spec):
    path, temporal = spec
    runtime = load_runtime()
    labels, circles = {}, {}
    collector = TrainingCollector(labels, temporal)
    tracker, feature_history = runtime.new_packet()
    tracker.residual_linker = collector
    state = {"_precision_packet": (tracker, feature_history)}
    for _, payload in iter_circle_payloads(path, labels, circles, TRAIN_PACKETS):
        points = [dict(zip(POINT_KEYS, record)) for record in WIRE_POINT.iter_unpack(payload)]
        collector.observe(runtime.process_circle(points, state))
    size = len(TEMPORAL_FEATURES if temporal else FEATURES)
    values = np.concatenate(collector.features) if collector.features else np.empty((0, size))
    targets = np.array(collector.targets, dtype=bool)
    print(f"{path.stem} {targets.size} recovery edges, {targets.sum()} positive", flush=True)
    return values, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--spatial-model", type=Path, help="Also fit the non-temporal feature subset")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.spatial_model is not None and (not args.temporal or args.spatial_model.resolve() == args.model.resolve()):
        parser.error("--spatial-model requires temporal features and a different output path")
    paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        datasets = list(pool.map(collect, ((path, args.temporal) for path in paths)))
    values, targets = [np.concatenate(part) for part in zip(*datasets)]
    params = {
        "objective": "binary", "learning_rate": .055, "num_leaves": 31, "min_data_in_leaf": 55,
        "num_threads": args.jobs, "seed": 3047, "deterministic": True, "force_col_wise": True, "verbosity": -1,
    }
    features = TEMPORAL_FEATURES if args.temporal else FEATURES
    model = lgb.train(params, lgb.Dataset(values, label=targets, feature_name=list(features)),
                      num_boost_round=165)
    model.save_model(str(args.model))
    report = {
        "packets": args.packets, "examples": len(targets), "positives": int(targets.sum()),
        "parameters": params, "features": features, "model_sha256": file_sha256(args.model),
    }
    if args.spatial_model:
        spatial = lgb.Dataset(values[:, :len(FEATURES)], label=targets, feature_name=list(FEATURES))
        lgb.train(params, spatial, num_boost_round=165).save_model(str(args.spatial_model))
        report["spatial_model_sha256"] = file_sha256(args.spatial_model)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
