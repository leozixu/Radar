"""Calibrate public point emissions with causal measurement and ownership history."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from causal_birth_paths import BirthOnline
from algorithm import load_runtime
from baseline_cv import FOLLOW_GATE, residual
from clock_association import phase_features
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from paired_returns import PairReturns
import replay_validation as replay
import tracker_compat


HERE = Path(__file__).resolve().parent
INPUT_MODELS = {
    "teacher": HERE / "teacher_assoc_wide.txt",
    "pair": HERE / "models/pair_swap_wide.txt",
    "birth": HERE / "models/causal_birth_paths.txt",
    "clock": HERE / "models/scan_clock.txt",
}
HISTORY = 10
FEATURES = tracker_compat.QUALITY_FEATURES + (
    "legacy_quality", "link_probability", "clock_probability", "scan_phase_error",
    "snr", "previous_proposals", "trusted_proposals", "previous_gap", "trusted_gap",
    "recent_link_mean", "recent_clock_mean", "recent_quality_mean", "amplitude_deviation",
) + tuple(f"{group}_{axis}" for group in ("step", "trusted_forecast", "recent_forecast")
          for axis in ("azimuth", "elevation", "range"))


class EmissionFeatures:
    def __init__(self):
        self.history = {}

    @staticmethod
    def forecast_error(past, position, circle):
        if len(past) < 2:
            return np.full(3, np.nan)
        older, previous = past[-2:]
        velocity = residual(previous[1], older[1]) / (previous[0] - older[0])
        predicted = previous[1] + velocity * (circle - previous[0])
        return np.abs(residual(position, predicted)) / FOLLOW_GATE

    def build(self, points, proposals, quality, probabilities, state, clock):
        live = {track.batch_id for track in state["tracks"][0]}
        self.history = {batch: entries for batch, entries in self.history.items() if batch in live}
        if not proposals:
            return np.empty((0, len(FEATURES)), dtype=np.float32)
        measurements = [(p["Azimuth"], p["Time"], p["Amp"]) for p in points]
        phase = phase_features(*np.array(measurements).T)
        scan_probability = clock.predict(phase, num_threads=1)
        indices = {p["PntID"]: i for i, p in enumerate(points)}
        associations = state["association_confidence"]
        extra = []
        for result, probability in zip(proposals, probabilities):
            ident = result["PntID"]
            index = indices[ident]
            point = points[index]
            past = self.history.setdefault(result["BatchID"], deque(maxlen=HISTORY))
            circle, amplitude = point["CircleID"], point["Amp"]
            position = np.array((point["Azimuth"], point["Elevation"], point["Distance"]))
            link = associations.get(ident, np.nan)
            phase_prob = scan_probability[index]
            trusted = [row for row in past if row[3] >= .80 and row[4] >= .50]
            if past:
                previous_gap = circle - past[-1][0]
                step = np.abs(residual(position, past[-1][1])) / (FOLLOW_GATE * previous_gap)
                prior_links = [row[3] for row in past if np.isfinite(row[3])]
                mean_link = np.mean(prior_links) if prior_links else np.nan
                mean_phase = np.mean([row[4] for row in past])
                mean_quality = np.mean([row[5] for row in past])
                amp_change = abs(amplitude - np.median([row[2] for row in past]))
            else:
                previous_gap = mean_link = mean_phase = mean_quality = amp_change = np.nan
                step = np.full(3, np.nan)
            trusted_gap = circle - trusted[-1][0] if trusted else np.nan
            extra.append((probability, link, phase_prob, phase[index, 0], point["SNR"], len(past),
                          len(trusted), previous_gap, trusted_gap, mean_link, mean_phase, mean_quality,
                          amp_change, *step, *self.forecast_error(trusted, position, circle),
                          *self.forecast_error(list(past), position, circle)))
            past.append((circle, position, amplitude, link, phase_prob, probability))
        return np.column_stack((quality, extra)).astype(np.float32)


class TrainingLabels:
    def __init__(self):
        self.history = {}

    def labels(self, outputs, labels, live):
        self.history = {batch: votes for batch, votes in self.history.items() if batch in live}
        detection, ownership = [], []
        for output in outputs:
            truth = labels[output["PntID"]]
            votes = self.history.setdefault(output["BatchID"], Counter())
            owner = votes.most_common(1)
            if truth > 0 and owner and owner[0][1] >= 3 and owner[0][1] >= .80 * votes.total():
                belongs = int(truth == owner[0][0])
            else:
                belongs = -1
            detection.append(-1 if truth < 0 else int(truth > 0))
            ownership.append(belongs)
            if truth > 0:
                votes[truth] += 1
        return detection, ownership


def make_models():
    from lightgbm import Booster

    return {name: Booster(model_file=str(path)) for name, path in INPUT_MODELS.items()}


def outputs(path, allowed, pair_model=None, wide_pairs=False, fallback_floor=0., raw_model=None, raw_floor=.025,
            association_floor=.20):
    models = make_models()
    paired = None
    gate = None
    if pair_model or raw_model:
        from lightgbm import Booster

    if pair_model:
        paired = PairReturns(Booster(model_file=str(pair_model)), .985, .95,
                             wide=wide_pairs, fallback_floor=fallback_floor)
    if raw_model:
        from raw_timing_detection import RawReturnGate

        gate = RawReturnGate(Booster(model_file=str(raw_model)), raw_floor)
    tracker = BirthOnline(models["teacher"], models["pair"], models["birth"], models["clock"], .86,
                          "supplement", paired_birth=True, joint=True, two_hit_pairs=True, pairing=paired, raw_gate=gate,
                          association_floor=association_floor)
    legacy = load_runtime()
    labels, circles = {}, {}
    state, history = {}, EmissionFeatures()
    with patch.object(tracker_compat, "baseline_circle", tracker.track):
        for _, payload in replay.iter_circle_payloads(path, labels, circles, allowed):
            points = [dict(zip(replay.POINT_KEYS, record)) for record in replay.POINT.iter_unpack(payload)]
            proposals, quality = tracker_compat.decisions_with_features(points, state)
            probabilities = legacy.quality_model.predict(quality, num_threads=1) if proposals else np.empty(0)
            features = history.build(points, proposals, quality, probabilities, state, models["clock"])
            yield points, proposals, probabilities, features, labels, circles, state


def collect(path):
    feature_rows, detection, ownership = [], [], []
    teacher = TrainingLabels()
    for _, proposals, _, features, truth, _, state in outputs(path, TRAIN_PACKETS):
        live = {track.batch_id for track in state["tracks"][0]}
        detection_rows, ownership_rows = teacher.labels(proposals, truth, live)
        feature_rows.append(features)
        detection.extend(detection_rows)
        ownership.extend(ownership_rows)
    print(path.name, len(detection), flush=True)
    return np.concatenate(feature_rows), np.array(detection), np.array(ownership)


def run_workers(worker, specs, jobs):
    if jobs == 1:
        return list(map(worker, specs))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(worker, specs))


def fit(args):
    import lightgbm as lgb

    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    data = run_workers(collect, paths, args.jobs)
    features, detection, ownership = (np.concatenate(parts) for parts in zip(*data))
    counts = {}
    for kind, target in (("detection", detection), ("ownership", ownership)):
        known = target >= 0
        counts[kind] = np.bincount(target[known], minlength=2).tolist()
        sample = lgb.Dataset(features[known], label=target[known], feature_name=list(FEATURES))
        model = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 2, "seed": 130,
                           "learning_rate": .045, "num_leaves": 39, "min_data_in_leaf": 90},
                          sample, num_boost_round=220)
        model.save_model(str(args.models / f"emission_{kind}.txt"))
    report = {"packets": args.packets, "samples_by_class": counts, "features": FEATURES,
              "source_sha256": replay.file_sha256(Path(__file__)),
              "input_models": {name: replay.file_sha256(path) for name, path in INPUT_MODELS.items()}}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(counts, flush=True)


DECISIONS = {
    "baseline": (0., 1., 0.),
    "rescue_02": (0., .02, 0.),
    "rescue_10": (0., .10, 0.),
    "rescue_25": (0., .25, 0.),
    "rescue_35": (0., .35, 0.),
    "rescue_45": (0., .45, 0.),
    "rescue_55": (0., .55, 0.),
    "rescue_65": (0., .65, 0.),
    "rescue_80": (0., .80, 0.),
    "rescue_92": (0., .92, 0.),
    "rescue65_drop40": (.40, .65, 0.),
    "rescue65_drop60": (.60, .65, 0.),
    "rescue65_drop75": (.75, .65, 0.),
    "rescue65_drop85": (.85, .65, 0.),
    "rescue65_clutter10": (0., .65, .10),
    "rescue65_clutter25": (0., .65, .25),
    "rescue65_clutter40": (0., .65, .40),
    "rescue65_clutter55": (0., .65, .55),
    "rescue65_drop60_clutter25": (.60, .65, .25),
    "rescue65_drop75_clutter40": (.75, .65, .40),
}
RAW_RESCUE = {
    "rescue10_raw80": (.10, .80),
    "rescue25_raw80": (.25, .80),
    "rescue25_raw85": (.25, .85),
    "rescue25_raw90": (.25, .90),
    "rescue25_raw97": (.25, .97),
    "rescue65_raw96": (.65, .96),
}


def decision_policies(raw_model):
    return DECISIONS | ({name: (0., floor, 0.) for name, (floor, _) in RAW_RESCUE.items()} if raw_model else {})


def current_raw_probabilities(points, proposals, state):
    source_q = state.get("raw_return_probability")
    if source_q is None:
        return None
    scores_by_point = dict(zip((p["PntID"] for p in points), source_q))
    return np.fromiter((scores_by_point[p["PntID"]] for p in proposals), dtype=float)


def accepted_emissions(baseline, detection, ownership, young, thresholds, raw_probability=None, raw_floor=None):
    reject, rescue, detect_floor = thresholds
    confident_owner = (ownership >= .80) | young
    keep = baseline & ((ownership >= reject) | young)
    keep |= (detection >= rescue) & confident_owner
    if raw_floor is not None:
        keep |= (raw_probability >= raw_floor) & confident_owner
    return keep & (detection >= detect_floor)


def compare(spec):
    from lightgbm import Booster

    path, model_dir, args = spec
    models = {kind: Booster(model_file=str(model_dir / f"emission_{kind}.txt"))
              for kind in ("detection", "ownership")}
    legacy = load_runtime()
    times = {}
    policies = decision_policies(args.raw_model)
    selections = {name: {} for name in policies}
    for points, proposals, probabilities, features, labels, circles, state in outputs(
        path, TUNING_PACKETS, args.paired, args.wide_pairs, args.fallback_floor, args.raw_model, args.raw_floor,
        args.association_floor,
    ):
        times.update((p["PntID"], p["Time"]) for p in points)
        if not proposals:
            continue
        detection = models["detection"].predict(features, num_threads=1)
        ownership = models["ownership"].predict(features, num_threads=1)
        young = features[:, FEATURES.index("trusted_proposals")] < 3
        baseline = probabilities >= legacy.threshold
        source = {(p["Time"], p["CircleID"], p["PntID"]): (p["Azimuth"], p["Elevation"], p["Distance"])
                  for p in points}
        frame = replay.encode_tracks(proposals, source)
        validated = [(row[-1], row[-3]) for row in replay.TRACK.iter_unpack(memoryview(frame)[5:-4])]
        source_q = current_raw_probabilities(points, proposals, state)
        for name, thresholds in policies.items():
            raw_floor = RAW_RESCUE[name][1] if name in RAW_RESCUE else None
            keep = accepted_emissions(baseline, detection, ownership, young, thresholds, source_q, raw_floor)
            selections[name].update(pair for pair, accepted in zip(validated, keep) if accepted)
    results = {name: evaluate_packet(labels, decisions, circles, times) for name, decisions in selections.items()}
    print(path.name, {name: round(result["score"]["total"], 3) for name, result in results.items()}, flush=True)
    return results


def tune(args):
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    sources = {name: replay.file_sha256(HERE / name) for name in (
        "emission_identity.py", "causal_birth_paths.py", "paired_returns.py", "raw_timing_detection.py"
    )}
    fingerprints = {name: replay.file_sha256(path) for name, path in INPUT_MODELS.items()}
    results = run_workers(compare, [(path, args.models, args) for path in paths], args.jobs)
    policies = decision_policies(args.raw_model)
    summaries = {name: aggregate_packets([result[name] for result in results]) for name in policies}
    report = {"packets": args.packets, "score_version": SCORE_VERSION, "thresholds": policies,
              "raw_rescue_thresholds": RAW_RESCUE if args.raw_model else {},
              "source_sha256": sources["emission_identity.py"], "source_sha256_by_file": sources,
              "tracker_models_sha256": fingerprints,
              "paired_returns_sha256": replay.file_sha256(args.paired) if args.paired else None,
              "wide_pairs": args.wide_pairs, "fallback_floor": args.fallback_floor,
              "raw_model_sha256": replay.file_sha256(args.raw_model) if args.raw_model else None,
              "raw_floor": args.raw_floor if args.raw_model else None,
              "association_floor": args.association_floor,
              "models": {name: replay.file_sha256(args.models / f"emission_{name}.txt")
                         for name in ("detection", "ownership")}, "summaries": summaries,
              "packet_scores": [{name: metric["score"] for name, metric in result.items()} for result in results]}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, summary in summaries.items():
        print(name, summary["score"]["total"], summary["id_switches"], summary["split_truth_tracks"],
              summary["idf1"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "tune"))
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--models", type=Path, default=HERE / "models")
    parser.add_argument("--paired", type=Path, help="apply the emission detector after exact paired-return tracking")
    parser.add_argument("--wide-pairs", action="store_true")
    parser.add_argument("--fallback-floor", type=float, default=0.)
    parser.add_argument("--raw-model", type=Path)
    parser.add_argument("--raw-floor", type=float, default=.025)
    parser.add_argument("--association-floor", type=float, default=.20)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.stage == "fit" and (args.paired or args.raw_model):
        parser.error("two-stage profiles are supported only when tuning")
    if not 0 <= args.fallback_floor <= 1 or (args.wide_pairs or args.fallback_floor) and not args.paired:
        parser.error("paired fallback settings require --paired and a floor between 0 and 1")
    if not 0 < args.raw_floor < 1:
        parser.error("expected raw return score floor between 0 and 1")
    if not 0 < args.association_floor < 1:
        parser.error("expected an association threshold between 0 and 1")
    (fit if args.stage == "fit" else tune)(args)


if __name__ == "__main__":
    main()
