"""Train and evaluate causal identity re-association using only train_001–085."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, iter_circle_payloads
from tracker_compat import decisions_with_features

from identity_rejoin import FEATURES, History, IdentityRuntime, candidate_features, load_quality_reference, proposal_arrays


class TrainingState:
    def __init__(self, clutter_histories=False):
        self.history = {}
        self.features = []
        self.targets = []
        self.available = 0
        self.recalled = 0
        self.clutter_histories = clutter_histories

    def update(self, scan, proposals, arrays, labels):
        candidate_labels = np.asarray([labels[point["PntID"]] for point in proposals])
        positions, amps, raw_ids, _, velocity, quality = arrays
        self.history = {bid: h for bid, h in self.history.items() if scan - h.scans[-1] <= 16}
        truth_ids = np.asarray(list(self.history), dtype=int)
        rr, cc, features = candidate_features(list(self.history.values()), scan, arrays)
        trusted = candidate_labels[cc] >= 0
        true_edge = truth_ids[rr] == candidate_labels[cc]
        self.features.append(features[trusted])
        self.targets.append(true_edge[trusted])
        self.available += (np.isin(candidate_labels, truth_ids) & (candidate_labels > 0)).sum()
        self.recalled += (true_edge & trusted).sum()
        for index in np.argsort(-quality, kind="stable"):
            target = candidate_labels[index]
            if target == 0 and self.clutter_histories:
                target = -raw_ids[index]
            elif target <= 0:
                continue
            history = self.history.setdefault(target, History())
            if history.scans and history.scans[-1] == scan:
                continue
            history.add(scan, positions[index], amps[index], raw_ids[index], quality[index], velocity[index])

    def result(self):
        return np.concatenate(self.features), np.concatenate(self.targets), int(self.recalled), int(self.available)


VARIANTS = {
    "low": {"match_probability": .24},
    "medium": {"match_probability": .40},
    "high": {"match_probability": .57},
    "conservative_birth": {"match_probability": .40, "confirm_hits": 2},
    "same_raw": {"match_probability": .38, "same_raw_bonus": .85},
    "same_raw_hold": {"match_probability": .38, "same_raw_bonus": .85, "hold_raw": True},
    "defer_birth": {"match_probability": .24, "hold_nearby_births": .12},
    "defer_uncertain": {"match_probability": .24, "publish_probability": .72, "hold_nearby_births": .12},
    "defer_crossing": {"match_probability": .24, "publish_probability": .87, "hold_nearby_births": .08},
    "weak_cross": {"match_probability": .24, "weak_cross_probability": .55},
    "trusted_cross": {"match_probability": .18, "weak_cross_probability": .64, "hold_nearby_births": .35},
    "stitch": {"match_probability": .62, "lineage_min_gap": 4},
    "stitch_high": {"match_probability": .78, "lineage_min_gap": 4},
}


def run_packet(args):
    path, command, model_path, variant_names, teacher_model, teacher_settings = args
    allowed = TRAIN_PACKETS if command == "train" else TUNING_PACKETS
    runtime = load_quality_reference()
    if teacher_model is not None:
        from lightgbm import Booster
        from teacher_assoc import Online
        import tracker_compat

        teacher = Booster(model_file=str(teacher_model))
        tracker_compat.baseline_circle = Online(teacher, **teacher_settings).track
    labels, times, circles = {}, {}, {}
    raw = {}
    if command == "train":
        training = TrainingState(clutter_histories=teacher_model is not None)
    else:
        from lightgbm import Booster

        model = Booster(model_file=str(model_path))
        if tuple(model.feature_name()) != FEATURES:
            raise ValueError("identity model feature schema disagrees with runtime")
        base = IdentityRuntime(runtime.quality_model, model, threshold=runtime.threshold)
        variants = {name: replace(base, **VARIANTS[name]) for name in variant_names}
        states = {name: {} for name in variants}
        upstream = "teacher" if teacher_model is not None else "frozen"
        outputs = {name: {} for name in (*variants, upstream)}
    started = time.perf_counter()
    for scan, payload in iter_circle_payloads(path, labels, circles, allowed=allowed):
        records = tuple(POINT.iter_unpack(payload))
        points = [dict(zip(POINT_KEYS, record)) for record in records]
        if command != "train":
            times.update((record[-1], record[0]) for record in records)
        proposals, features = decisions_with_features(points, raw)
        if not proposals:
            continue
        probabilities = runtime.quality_model.predict(features, num_threads=1)
        keep = probabilities >= runtime.threshold
        proposals = [row for row, selected in zip(proposals, keep) if selected]
        if not proposals:
            continue
        arrays = proposal_arrays(points, proposals, probabilities[keep], raw)
        if command == "train":
            training.update(scan, proposals, arrays, labels)
        else:
            for name, candidate in variants.items():
                result = candidate.associate(scan, proposals, arrays, states[name])
                outputs[name].update((row["PntID"], row["BatchID"]) for row in result)
            outputs[upstream].update((row["PntID"], row["BatchID"]) for row in proposals)
    print(f"{path.stem} {time.perf_counter() - started:.1f}s", flush=True)
    if command == "train":
        return training.result()
    return {name: evaluate_packet(labels, mapping, circles, times) for name, mapping in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "tune"))
    parser.add_argument("--packets", type=int, nargs="+")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--variants", choices=tuple(VARIANTS), nargs="+", default=("low", "medium", "same_raw"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--teacher-model", type=Path)
    parser.add_argument("--teacher-floor", type=float, default=.23)
    parser.add_argument("--teacher-mix", type=float, default=.20)
    parser.add_argument("--teacher-coast", type=int, default=5)
    parser.add_argument("--teacher-ambiguity", type=float, default=0.)
    parser.add_argument("--teacher-robust", type=float, default=0.)
    args = parser.parse_args()
    if args.model.suffix.lower() != ".txt":
        parser.error("--model must be a .txt file")
    if args.report is not None and args.report.suffix.lower() != ".json":
        parser.error("--report must be a .json file")
    allowed = TRAIN_PACKETS if args.command == "train" else TUNING_PACKETS
    default = range(1, 13) if args.command == "train" else (71, 76, 79, 85)
    indices = args.packets or tuple(default)
    paths = packet_paths(DEFAULT_DATA_DIR, indices, allowed=allowed)
    teacher_settings = {name: getattr(args, f"teacher_{name}") for name in ("floor", "mix", "coast", "ambiguity", "robust")}
    jobs = [(path, args.command, args.model, args.variants, args.teacher_model, teacher_settings) for path in paths]
    if args.jobs == 1:
        results = list(map(run_packet, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_packet, jobs))
    if args.command == "train":
        import lightgbm as lgb

        features, targets, recalled, available = zip(*results)
        x, y = np.concatenate(features), np.concatenate(targets)
        print(f"Rows {len(y)}, positive {int(y.sum())}, geometric recall {sum(recalled) / sum(available):.4f}", flush=True)
        weights = None
        if args.teacher_model is not None:
            same = x[:, 3] > .5
            weights = np.where(y & ~same, 2.1, np.where(~y & same, 1.7, 1.))
        data = lgb.Dataset(x, label=y, weight=weights, feature_name=list(FEATURES))
        model = lgb.train({"objective": "binary", "num_threads": args.jobs, "learning_rate": .075,
                           "num_leaves": 41, "min_data_in_leaf": 30, "verbosity": -1,
                           "seed": 20261115, "force_col_wise": True, "deterministic": True},
                          data, num_boost_round=145)
        model.save_model(args.model)
        report = {"rows": len(y), "positives": int(y.sum()), "geometric_recall": sum(recalled) / sum(available)}
    else:
        names = (*args.variants, "teacher" if args.teacher_model is not None else "frozen")
        summaries = {key: aggregate_packets(result[key] for result in results) for key in names}
        for name, result in summaries.items():
            print(name, round(result["score"]["total"], 4), "IDF1", round(result["idf1"], 4),
                  "ID switches", result["id_switches"], "split", result["split_truth_tracks"],
                  "merged", result["merged_truth_points"], flush=True)
        report = {
            "summary": summaries,
            "by_packet": dict(zip(indices, ({key: {name: value for name, value in summary.items() if name != "tracks"}
                                              for key, summary in result.items()} for result in results))),
            "variants": {name: VARIANTS[name] for name in args.variants},
        }
    if args.report:
        report.update({"packets": indices, "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()})
        if args.teacher_model:
            report.update({"teacher_settings": teacher_settings,
                           "teacher_sha256": hashlib.sha256(args.teacher_model.read_bytes()).hexdigest()})
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
