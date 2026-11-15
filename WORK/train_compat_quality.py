"""Train the compatibility quality classifier on causal baseline output features."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import PARTICIPANT_DIR, POINT, POINT_KEYS, file_sha256, iter_circle_payloads
from tracker_compat import QUALITY_FEATURES, decisions_with_features


DEFAULT_PACKETS = tuple(range(1, 11))
BOOSTING_ROUNDS = 180
TRAIN_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "max_depth": 5,
    "num_leaves": 23,
    "num_threads": 2,
    "verbosity": -1,
    "force_col_wise": True,
    "deterministic": True,
    "seed": 84,
}


def collect_packet(path):
    labels, circles, state = {}, {}, {}
    samples, targets = [], []
    for _, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
        points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
        decisions, features = decisions_with_features(points, state)
        if decisions:
            target = np.array([labels[point["PntID"]] for point in decisions], dtype=np.int32)
            known = target >= 0
            samples.append(features[known])
            targets.append(target[known] > 0)
    if not targets:
        return np.empty((0, len(QUALITY_FEATURES)), dtype=np.float32), np.empty(0, dtype=bool)
    return np.concatenate(samples)[::2], np.concatenate(targets)[::2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--packets", nargs="+", type=int, default=DEFAULT_PACKETS,
                        help="整包划分，限 001–070，默认 001–010")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--model-dir", type=Path, default=PARTICIPANT_DIR / "models")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs 必须为正整数")
    try:
        paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    except ValueError as exc:
        parser.error(str(exc))

    import lightgbm as lgb

    if args.jobs == 1:
        packets = list(map(collect_packet, paths))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            packets = list(pool.map(collect_packet, paths))
    x = np.concatenate([features for features, _ in packets])
    y = np.concatenate([labels for _, labels in packets])
    if len(y) == 0 or not y.any() or y.all():
        parser.error("模型训练需要至少一个正例和负例")
    data = lgb.Dataset(x, label=y, feature_name=list(QUALITY_FEATURES))
    model = lgb.train(TRAIN_PARAMS, data, num_boost_round=BOOSTING_ROUNDS)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    weights = args.model_dir / "quality.txt"
    model.save_model(str(weights))
    descriptor = {"file": weights.name, "sha256": file_sha256(weights)}
    profile = {
        "feature_version": 1, "backend": "compat",
        "models": {"quality": descriptor}, "settings": {"threshold": 0.20},
    }
    (args.model_dir / "compat.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report = {
            "packet_ids": list(args.packets), "examples": len(y), "positives": int(y.sum()),
            "boosting_rounds": BOOSTING_ROUNDS, "training_params": TRAIN_PARAMS,
            "features": list(QUALITY_FEATURES), "lightgbm": lgb.__version__, "model": descriptor,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"质量模型样本 {len(y):,}，正例 {y.sum():,}，SHA-256 {descriptor['sha256']}")


if __name__ == "__main__":
    main()
