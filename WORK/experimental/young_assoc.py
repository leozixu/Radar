"""Experimental association for one- and two-hit tracks, trained on causal windows."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__:
    from . import teacher_assoc as teacher
else:
    import teacher_assoc as teacher
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths, validate_packet_path
from metrics import aggregate_packets
from replay_validation import file_sha256, replay_packet
from train_tracker import read_packet


cv = teacher.cv
STAGES = ("seed", "tentative")
MOVEMENT_SCALE = np.array([0.002, 0.025, 0.0012])
FEATURES = (
    "amplitude", "previous_amplitude", "mean_amplitude", "amplitude_change",
    "first_amplitude_change", "geometric_cost", "amplitude_cost", "row_degree",
    "column_degree", "row_rank", "column_rank", "row_margin", "column_margin", "total_motion",
) + tuple(
    f"{group}_{axis}"
    for group in ("error", "motion", "previous_motion", "signed_residual")
    for axis in ("azimuth", "elevation", "range")
)


def _values(rows):
    # Match the float32 observation layout in the TCP client.
    return rows[:, 1:5].astype(np.float32).astype(float)


def _legacy_terms(first, last, points, initial):
    gate = cv.START_GATE if initial else cv.FOLLOW_GATE
    velocity = cv.residual(last[:, :3], first[:, :3])
    expected = last[:, :3] + velocity
    delta = cv.residual(points[None, :, :3], expected[:, None, :]) / gate
    dist = np.sum(delta**2, axis=2)
    mean_amp = (last[:, 3] + first[:, 3]) / 2
    amp = points[:, 3]
    amp_cost = 0.45 * np.maximum(0.35 - amp, 0) / 0.35 + 0.22 * abs(amp - mean_amp[:, None])
    return dist, amp_cost, delta, velocity, mean_amp


def _rank(r, c, values, size):
    count = np.bincount(r, minlength=size)
    best = np.full(size, np.inf)
    np.minimum.at(best, r, values)
    order = np.lexsort((c, values, r))
    ranks = np.empty(len(r), dtype=int)
    ranks[order] = np.arange(len(r)) - (count.cumsum() - count)[r[order]]
    return count[r], ranks, values - best[r]


def edge_features(first, last, points, initial):
    dist, amp_cost, delta, velocity, mean_amp = _legacy_terms(first, last, points, initial)
    r, c = np.nonzero(dist <= 1)
    if not len(r):
        return r, c, np.empty((0, len(FEATURES)), dtype=np.float32)
    cost = dist[r, c] + amp_cost[r, c]
    row_degree, row_rank, row_margin = _rank(r, c, cost, len(first))
    col_degree, col_rank, col_margin = _rank(c, r, cost, len(points))
    amp = points[c, 3]
    total = cv.residual(points[c, :3], first[r, :3]) / MOVEMENT_SCALE
    movement = cv.residual(points[c, :3], last[r, :3]) / cv.START_GATE
    return r, c, np.column_stack((
        amp, last[r, 3], mean_amp[r], abs(amp - last[r, 3]), abs(amp - first[r, 3]),
        dist[r, c], amp_cost[r, c], row_degree, col_degree, row_rank, col_rank,
        row_margin, col_margin, np.linalg.norm(total, axis=1), abs(delta[r, c]),
        movement, velocity[r] / cv.START_GATE, delta[r, c],
    )).astype(np.float32)


def _assign(rows, cols, penalty, left, right, miss):
    cost = np.full((left, right + left), np.inf)
    cost[rows, cols] = penalty
    indices = np.arange(left)
    cost[indices, right + indices] = miss
    r, c = linear_sum_assignment(cost)
    keep = c < right
    return r[keep], c[keep]


def _provisional(old, previous, old_truth, previous_truth):
    # Start with candidates a geometry-only tracker could actually produce.
    dist, amp_cost, *_ = _legacy_terms(old, old, previous, True)
    r, c = np.nonzero(dist <= 1)
    r, c = _assign(r, c, (dist + amp_cost)[r, c], len(old), len(previous), 1.0)
    children_to_parent = dict(zip(c, r))

    # Add clean 2-hit windows, including windows from existing true trajectories
    # so the scorer sees the motion that would follow a genuine restart.
    prior_truth = {label: i for i, label in enumerate(old_truth) if label > 0}
    for i, label in enumerate(previous_truth):
        if label in prior_truth:
            children_to_parent[i] = prior_truth[label]
    if not children_to_parent:
        return None
    children = np.array(sorted(children_to_parent))
    parents = np.array([children_to_parent[i] for i in children])
    old_labels, child_labels = old_truth[parents], previous_truth[children]
    known = (old_labels >= 0) & (child_labels >= 0)
    return old[parents], previous[children], np.where(
        known, np.where(old_labels == child_labels, child_labels, 0), -1,
    )


def _collect_pairs(features, r, c, context_truth, truth, rng):
    label = truth[c]
    context = context_truth[r]
    known = (label >= 0) & (context >= 0)
    positive = (context > 0) & (label == context)
    nearest = (features[:, 9] < 2) | (features[:, 10] < 2)
    # Keep close competitors at a higher rate than easy rows while retaining
    # both wrong-target and clutter edges for realistic missing-link decisions.
    keep = known & (positive | (rng.random(len(r)) < np.where(nearest, 0.60, 0.10)))
    return features[keep], positive[keep]


def collect(path):
    path = validate_packet_path(path, TRAIN_PACKETS)
    data = read_packet(path)
    boundaries = np.r_[0, np.flatnonzero(np.diff(data[:, 6])) + 1, len(data)]
    rng = np.random.default_rng(int(path.stem[-3:]) + 8419)
    previous = older = None
    features = {stage: [] for stage in STAGES}
    labels = {stage: [] for stage in STAGES}
    for lo, hi in zip(boundaries, boundaries[1:]):
        points = _values(data[lo:hi])
        truth = data[lo:hi, 8].astype(np.int32)
        circle = int(data[lo, 6])
        if previous is not None and circle == previous[0] + 1:
            _, last, last_truth = previous
            live = last[:, 3] >= cv.SEED_MIN_AMPLITUDE
            seeds = last[live]
            contexts = {"seed": (seeds, seeds, last_truth[live])}
            if older is not None and previous[0] == older[0] + 1:
                _, first, first_truth = older
                seeded = first[:, 3] >= cv.SEED_MIN_AMPLITUDE
                provisional = _provisional(first[seeded], last, first_truth[seeded], last_truth)
                if provisional is not None:
                    contexts["tentative"] = provisional
            for name, (head, tail, context_truth) in contexts.items():
                rows, cols, x = edge_features(head, tail, points, initial=name == "seed")
                x, y = _collect_pairs(x, rows, cols, context_truth, truth, rng)
                features[name].append(x)
                labels[name].append(y)
        older, previous = previous, (circle, points, truth)
    return {stage: (np.concatenate(features[stage]), np.concatenate(labels[stage])) for stage in STAGES}


class YoungAssociation:
    def __init__(self, models, floors, geometric_mix, original=None):
        self.models = models
        self.floors = floors
        self.mix = geometric_mix
        self.original = original if original is not None else cv.associate

    def __call__(self, tracks, positions, amplitudes, circle, initial=False):
        if not tracks or not len(positions):
            return []
        name = "seed" if initial else "tentative"
        model = self.models.get(name)
        if model is None or any(track.batch_id or circle - track.circle != 1 for track in tracks):
            return self.original(tracks, positions, amplitudes, circle, initial=initial)
        latest = np.array([(*track.history[-1][2], track.history[-1][1]) for track in tracks])
        first = latest if initial else np.array([(*track.history[-2][2], track.history[-2][1]) for track in tracks])
        rows, cols, features = edge_features(first, latest, np.column_stack((positions, amplitudes)), initial)
        if not len(rows):
            return []
        confidence = model.predict(features, num_threads=1)
        penalty = -np.log(np.maximum(confidence, 1e-6)) + self.mix * (features[:, 5] + features[:, 6])
        r, c = _assign(rows, cols, penalty, len(tracks), len(positions), -np.log(self.floors[name]))
        return list(zip(r, c))


def evaluate(config, path):
    from lightgbm import Booster

    models = {
        stage: Booster(model_file=str(config.model_dir / f"young_assoc_{stage}.txt")) for stage in config.stages
    }
    for model in models.values():
        if tuple(model.feature_name()) != FEATURES:
            raise ValueError("Young association weights and feature schema disagree.")
    mature = teacher.Online(Booster(model_file=str(config.teacher_weights)),
                            config.floor, config.mix, config.coast, config.ambiguity, config.robust)
    original_young = cv.associate
    original_core = teacher.compat.baseline_circle
    floors = {"seed": config.seed_floor, "tentative": config.tentative_floor}
    cv.associate = YoungAssociation(models, floors, config.geometric_mix, original_young)
    teacher.compat.baseline_circle = mature.track
    try:
        report = replay_packet(path, allowed=TUNING_PACKETS)
    finally:
        cv.associate = original_young
        teacher.compat.baseline_circle = original_core
    report.pop("_circle_times", None)
    print(f"{path.name}: {report['score']['total']:.4f}, switches={report['id_switches']}, splits={report['split_truth_tracks']}", flush=True)
    return report


def _train(paths, config):
    from lightgbm import Dataset, train

    samples = {stage: [] for stage in STAGES}
    for path in paths:
        result = collect(path)
        print(path.name, {stage: (len(result[stage][1]), int(result[stage][1].sum())) for stage in STAGES}, flush=True)
        for stage in STAGES:
            samples[stage].append(result[stage])
    config.model_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    params = {
        "objective": "binary", "learning_rate": .09, "num_leaves": 31, "min_child_samples": 45,
        "verbosity": -1, "num_threads": config.jobs, "seed": 7519, "force_col_wise": True,
    }
    for stage in STAGES:
        features = np.concatenate([x for x, _ in samples[stage]])
        labels = np.concatenate([y for _, y in samples[stage]])
        model = train(params, Dataset(features, label=labels, feature_name=list(FEATURES)), num_boost_round=90)
        path = config.model_dir / f"young_assoc_{stage}.txt"
        model.save_model(str(path))
        models[stage] = {"sha256": file_sha256(path), "samples": len(labels), "positives": int(labels.sum())}
    return {"models": models, "parameters": params, "boosting_rounds": 90}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "tune"))
    parser.add_argument("--packets", nargs="+", type=int, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--teacher-weights", type=Path)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--stages", nargs="*", choices=STAGES, default=STAGES)
    parser.add_argument("--floor", type=float, default=.20)
    parser.add_argument("--mix", type=float, default=0.)
    parser.add_argument("--coast", type=int, default=7)
    parser.add_argument("--ambiguity", type=float, default=.3)
    parser.add_argument("--robust", type=float, default=4.)
    parser.add_argument("--seed-floor", type=float, default=.12)
    parser.add_argument("--tentative-floor", type=float, default=.18)
    parser.add_argument("--geometric-mix", type=float, default=.15)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    allowed = TRAIN_PACKETS if args.phase == "train" else TUNING_PACKETS
    try:
        paths = packet_paths(DEFAULT_DATA_DIR, args.packets, allowed)
    except ValueError as error:
        parser.error(str(error))
    if args.phase == "tune" and args.teacher_weights is None:
        parser.error("--teacher-weights is required during tuning")
    if not all(0 < floor < 1 for floor in (args.floor, args.seed_floor, args.tentative_floor)):
        parser.error("probability thresholds must be between zero and one")

    started = time.monotonic()
    settings = {name: getattr(args, name) for name in (
        "stages", "floor", "mix", "coast", "ambiguity", "robust", "seed_floor", "tentative_floor", "geometric_mix",
    )}
    report = {"metadata": {"packet_ids": args.packets, "settings": settings, "source_sha256": file_sha256(Path(__file__))}}
    if args.phase == "train":
        report.update(_train(paths, args))
    else:
        runner = partial(evaluate, args)
        if args.jobs == 1:
            stats = list(map(runner, paths))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                stats = list(pool.map(runner, paths))
        totals = aggregate_packets(stats)
        report["summary"] = totals
        report["packets"] = [{key: value for key, value in item.items() if key != "tracks"} for item in stats]
        report["metadata"]["teacher_weights_sha256"] = file_sha256(args.teacher_weights)
        for stage in args.stages:
            report["metadata"][f"{stage}_weights_sha256"] = file_sha256(args.model_dir / f"young_assoc_{stage}.txt")
        print(json.dumps({k: totals[k] for k in (
            "score", "idf1", "mean_track_coverage", "false_alarm_ratio", "id_switches",
            "split_truth_tracks", "merged_output_tracks",
        )}), flush=True)
    report["wall_seconds"] = time.monotonic() - started
    if args.report:
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
